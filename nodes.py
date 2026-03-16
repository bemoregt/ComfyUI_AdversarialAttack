import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms.functional as TF
import numpy as np

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])


def normalize(x: torch.Tensor) -> torch.Tensor:
    """x: [B, C, H, W], values in [0, 1]"""
    mean = IMAGENET_MEAN.to(x.device).view(1, 3, 1, 1)
    std  = IMAGENET_STD.to(x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def denormalize(x: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(x.device).view(1, 3, 1, 1)
    std  = IMAGENET_STD.to(x.device).view(1, 3, 1, 1)
    return x * std + mean


def get_class_name(model_weights, idx: int) -> str:
    try:
        return model_weights.meta["categories"][idx]
    except Exception:
        return str(idx)


# ── FGSM ──────────────────────────────────────────────────────────────────────

def fgsm(model, x_norm, true_label, epsilon, targeted=False):
    """
    x_norm : normalized image tensor [1, 3, H, W], requires_grad=True
    Returns perturbed normalized tensor (clamped in pixel space).
    """
    with torch.enable_grad():
        x_adv = x_norm.clone().detach().requires_grad_(True)
        logits = model(x_adv)
        loss = nn.CrossEntropyLoss()(logits, true_label)
        if targeted:
            loss = -loss
        loss.backward()
        grad_sign = x_adv.grad.sign()

    # perturb in normalised space; clamp back to valid pixel range
    x_adv_denorm = denormalize(x_adv) + epsilon * grad_sign * IMAGENET_STD.to(x_adv.device).view(1,3,1,1)
    x_adv_denorm = x_adv_denorm.clamp(0, 1)
    return normalize(x_adv_denorm).detach()


# ── PGD ───────────────────────────────────────────────────────────────────────

def pgd(model, x_norm, true_label, epsilon, alpha, iterations, targeted=False):
    """
    Projected Gradient Descent (Madry et al. 2017).
    All perturbation arithmetic is done in pixel space; model receives normalised input.
    """
    x_orig = denormalize(x_norm).clamp(0, 1).detach()          # pixel space reference
    x_adv  = x_orig.clone() + torch.empty_like(x_orig).uniform_(-epsilon, epsilon)
    x_adv  = x_adv.clamp(0, 1)

    for _ in range(iterations):
        with torch.enable_grad():
            x_adv = x_adv.detach().requires_grad_(True)
            logits = model(normalize(x_adv))
            loss = nn.CrossEntropyLoss()(logits, true_label)
            if targeted:
                loss = -loss
            loss.backward()
            grad_sign = x_adv.grad.sign()

        x_adv = (x_adv + alpha * grad_sign).detach()
        # project back onto epsilon-ball
        delta = (x_adv - x_orig).clamp(-epsilon, epsilon)
        x_adv = (x_orig + delta).clamp(0, 1)

    return normalize(x_adv).detach()


# ══════════════════════════════════════════════════════════════════════════════
# Node 1 – LoadResNet18
# ══════════════════════════════════════════════════════════════════════════════

class LoadResNet18:
    WEIGHTS = models.ResNet18_Weights.IMAGENET1K_V1

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pretrained": (["pretrained", "random"],),
            }
        }

    RETURN_TYPES  = ("RESNET_MODEL",)
    RETURN_NAMES  = ("model",)
    FUNCTION      = "load_model"
    CATEGORY      = "AdversarialAttack"
    DESCRIPTION   = "ImageNet pretrained ResNet18 모델을 로드합니다."

    def load_model(self, pretrained: str):
        weights = self.WEIGHTS if pretrained == "pretrained" else None
        model   = models.resnet18(weights=weights)
        model.eval()
        # attach weights meta for class-name lookup later
        model._aa_weights = self.WEIGHTS
        return (model,)


# ══════════════════════════════════════════════════════════════════════════════
# Node 2 – AdversarialAttack
# ══════════════════════════════════════════════════════════════════════════════

class AdversarialAttack:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":         ("RESNET_MODEL",),
                "image":         ("IMAGE",),
                "attack_method": (["FGSM", "PGD"],),
                "epsilon": (
                    "FLOAT",
                    {"default": 0.03, "min": 0.001, "max": 0.3,
                     "step": 0.001, "display": "number"},
                ),
                "iterations": (
                    "INT",
                    {"default": 40, "min": 1, "max": 500, "step": 1},
                ),
                "alpha": (
                    "FLOAT",
                    {"default": 0.007, "min": 0.001, "max": 0.1,
                     "step": 0.001, "display": "number"},
                ),
                "target_class": (
                    "INT",
                    {"default": -1, "min": -1, "max": 999, "step": 1},
                ),
                "resize_to_224": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES  = ("IMAGE", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES  = (
        "adversarial_image",
        "original_class_name",
        "adversarial_class_name",
        "original_class_idx",
        "adversarial_class_idx",
    )
    FUNCTION      = "attack"
    CATEGORY      = "AdversarialAttack"
    DESCRIPTION   = (
        "FGSM 또는 PGD를 이용해 ResNet18이 다른 클래스로 분류하는 "
        "적대적 이미지를 생성합니다."
    )

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _comfy_to_bchw(image: torch.Tensor, resize: bool) -> torch.Tensor:
        """ComfyUI IMAGE [B, H, W, C] float32 → [B, C, H, W], optionally resized."""
        x = image.permute(0, 3, 1, 2)          # [B, C, H, W]
        if resize:
            x = TF.resize(x, [224, 224], antialias=True)
        return x.float()

    @staticmethod
    def _bchw_to_comfy(x: torch.Tensor, original_hw=None) -> torch.Tensor:
        """[B, C, H, W] → ComfyUI IMAGE [B, H, W, C], optionally resized back."""
        if original_hw is not None:
            x = TF.resize(x, list(original_hw), antialias=True)
        return x.permute(0, 2, 3, 1).clamp(0, 1)

    # ── main ──────────────────────────────────────────────────────────────────

    def attack(
        self,
        model,
        image: torch.Tensor,
        attack_method: str,
        epsilon: float,
        iterations: int,
        alpha: float,
        target_class: int,
        resize_to_224: bool,
    ):
        device = next(model.parameters()).device
        original_hw = (image.shape[1], image.shape[2])   # H, W

        # ── prepare input ──────────────────────────────────────────────────
        x_pixel = self._comfy_to_bchw(image, resize=resize_to_224).to(device)  # [B,C,H,W] ∈[0,1]
        x_norm  = normalize(x_pixel)                                             # normalised

        # ── get original prediction ────────────────────────────────────────
        with torch.no_grad():
            orig_logits = model(x_norm)
        orig_idx = orig_logits.argmax(dim=1)   # [B]

        # ── decide label for the attack loss ──────────────────────────────
        targeted = target_class >= 0
        if targeted:
            attack_label = torch.tensor([target_class], dtype=torch.long, device=device).expand(x_norm.shape[0])
        else:
            attack_label = orig_idx          # untargeted: maximise loss on true class

        # ── run attack ────────────────────────────────────────────────────
        if attack_method == "FGSM":
            x_adv_norm = fgsm(model, x_norm, attack_label, epsilon, targeted=targeted)
        else:
            x_adv_norm = pgd(model, x_norm, attack_label, epsilon, alpha, iterations, targeted=targeted)

        # ── final prediction on adversarial image ─────────────────────────
        with torch.no_grad():
            adv_logits = model(x_adv_norm)
        adv_idx = adv_logits.argmax(dim=1)

        # ── look up class names ───────────────────────────────────────────
        weights = getattr(model, "_aa_weights", None)
        orig_name = get_class_name(weights, orig_idx[0].item()) if weights else str(orig_idx[0].item())
        adv_name  = get_class_name(weights, adv_idx[0].item())  if weights else str(adv_idx[0].item())

        # ── warn if attack failed (same class) ────────────────────────────
        if orig_idx[0].item() == adv_idx[0].item():
            print(
                f"[AdversarialAttack] Warning: attack did not change class "
                f"(still '{orig_name}'). Try larger epsilon or more iterations."
            )

        # ── convert back to ComfyUI format ────────────────────────────────
        x_adv_pixel = denormalize(x_adv_norm).clamp(0, 1)
        out_hw = original_hw if resize_to_224 else None
        adv_image = self._bchw_to_comfy(x_adv_pixel, original_hw=out_hw if resize_to_224 else None)
        adv_image = adv_image.to(image.device)

        print(f"[AdversarialAttack] Original: {orig_name} ({orig_idx[0].item()})")
        print(f"[AdversarialAttack] Adversarial: {adv_name} ({adv_idx[0].item()})")

        return (
            adv_image,
            orig_name,
            adv_name,
            orig_idx[0].item(),
            adv_idx[0].item(),
        )


# ══════════════════════════════════════════════════════════════════════════════
# Node 3 – ClassifyImage  (원본/적대 이미지 분류 결과를 나란히 보여주는 보조 노드)
# ══════════════════════════════════════════════════════════════════════════════

class ClassifyImage:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("RESNET_MODEL",),
                "image": ("IMAGE",),
                "top_k": ("INT", {"default": 5, "min": 1, "max": 20, "step": 1}),
                "resize_to_224": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES  = ("STRING",)
    RETURN_NAMES  = ("top_k_results",)
    OUTPUT_NODE   = True
    FUNCTION      = "classify"
    CATEGORY      = "AdversarialAttack"
    DESCRIPTION   = "이미지를 ResNet18로 분류하고 Top-K 결과를 문자열로 반환합니다."

    def classify(self, model, image: torch.Tensor, top_k: int, resize_to_224: bool):
        device = next(model.parameters()).device
        x = image.permute(0, 3, 1, 2).float().to(device)
        if resize_to_224:
            x = TF.resize(x, [224, 224], antialias=True)
        x_norm = normalize(x)

        with torch.no_grad():
            logits = model(x_norm)
        probs = torch.softmax(logits, dim=1)
        top_probs, top_idxs = probs[0].topk(top_k)

        weights = getattr(model, "_aa_weights", None)
        lines = []
        for rank, (prob, idx) in enumerate(zip(top_probs, top_idxs), start=1):
            name = get_class_name(weights, idx.item()) if weights else str(idx.item())
            lines.append(f"{rank}. {name} ({idx.item()}): {prob.item()*100:.2f}%")

        result = "\n".join(lines)
        print(f"[ClassifyImage]\n{result}")
        return (result,)
