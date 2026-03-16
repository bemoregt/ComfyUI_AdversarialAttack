# ComfyUI AdversarialAttack

A ComfyUI custom node set that generates adversarial examples using a pretrained ResNet18 classifier. Given an input image, the nodes produce a visually similar output image that is misclassified as a **different** ImageNet class.

## Nodes

### 1. Load ResNet18
Loads a ResNet18 model with optional ImageNet pretrained weights.

| Parameter | Type | Description |
|-----------|------|-------------|
| `pretrained` | Enum | `pretrained` — ImageNet weights; `random` — randomly initialized |

**Output:** `RESNET_MODEL`

---

### 2. Adversarial Attack (FGSM/PGD)
Applies an adversarial perturbation to the input image so that ResNet18 predicts a class different from the original.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | RESNET_MODEL | — | ResNet18 from *Load ResNet18* |
| `image` | IMAGE | — | Input image |
| `attack_method` | Enum | `FGSM` | `FGSM` or `PGD` |
| `epsilon` | Float | 0.03 | Maximum perturbation magnitude (L∞ norm, pixel space) |
| `iterations` | Int | 40 | Number of PGD steps (ignored for FGSM) |
| `alpha` | Float | 0.007 | Per-step size for PGD |
| `target_class` | Int | -1 | `-1` = untargeted; `0–999` = force prediction toward this ImageNet class |
| `resize_to_224` | Bool | True | Resize image to 224×224 before inference |

**Outputs:**

| Name | Type | Description |
|------|------|-------------|
| `adversarial_image` | IMAGE | Perturbed image (same spatial size as input) |
| `original_class_name` | STRING | Predicted class of the original image |
| `adversarial_class_name` | STRING | Predicted class of the adversarial image |
| `original_class_idx` | INT | ImageNet index of original prediction |
| `adversarial_class_idx` | INT | ImageNet index of adversarial prediction |

---

### 3. Classify Image (ResNet18)
Runs ResNet18 inference and returns the Top-K predictions. Useful for verifying results before and after an attack.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | RESNET_MODEL | — | ResNet18 from *Load ResNet18* |
| `image` | IMAGE | — | Image to classify |
| `top_k` | Int | 5 | Number of top predictions to return |
| `resize_to_224` | Bool | True | Resize image to 224×224 before inference |

**Output:** `STRING` — ranked list of class names and confidence scores.

---

## Attack Methods

### FGSM (Fast Gradient Sign Method)
Single-step attack (Goodfellow et al., 2014):

```
x_adv = x + ε · sign(∇ₓ L(f(x), y))
```

Fast but relatively weak. Good for quick experiments.

### PGD (Projected Gradient Descent)
Iterative attack (Madry et al., 2017):

```
x⁰   = x + Uniform(−ε, ε)
xᵗ⁺¹ = Proj_{B(x,ε)} [ xᵗ + α · sign(∇ₓ L(f(xᵗ), y)) ]
```

Stronger than FGSM. Increase `iterations` and tune `alpha` for harder attacks.

---

## Suggested Workflow

```
[Load Image] ──────────────────────────────────────────┐
                                                        ▼
[Load ResNet18] ──► [Adversarial Attack] ──► [Preview Image]
                            │
                            ├──► original_class_name
                            └──► adversarial_class_name
```

Add **Classify Image** nodes on both the original and adversarial images to compare predictions side by side.

---

## Installation

1. Copy (or symlink) this folder into your ComfyUI `custom_nodes/` directory:
   ```bash
   cp -r ComfyUI_AdversarialAttack /path/to/ComfyUI/custom_nodes/
   ```
2. Install dependencies:
   ```bash
   pip install torch torchvision
   ```
3. Restart ComfyUI. The three nodes will appear under the **AdversarialAttack** category.

---

## Requirements

| Package | Version |
|---------|---------|
| Python | ≥ 3.9 |
| PyTorch | ≥ 2.0 |
| torchvision | ≥ 0.15 |

---

## Tips

- **Attack failed (same class)?** Increase `epsilon` (e.g. `0.1`) or switch from FGSM to PGD with more iterations.
- **Targeted attack:** Set `target_class` to any ImageNet class index (0–999) to steer the prediction toward a specific class.
- **Perturbation visibility:** Larger `epsilon` values produce stronger attacks but more visible noise. Values around `0.01–0.05` are typically imperceptible.
- **GPU support:** The attack runs on whatever device the model is on. Move the model to GPU before connecting it to the attack node for faster PGD.

---

## References

- Goodfellow et al. (2014) — [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572)
- Madry et al. (2017) — [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083)
