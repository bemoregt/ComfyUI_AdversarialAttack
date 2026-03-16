from .nodes import LoadResNet18, AdversarialAttack, ClassifyImage

NODE_CLASS_MAPPINGS = {
    "LoadResNet18":      LoadResNet18,
    "AdversarialAttack": AdversarialAttack,
    "ClassifyImage":     ClassifyImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadResNet18":      "Load ResNet18",
    "AdversarialAttack": "Adversarial Attack (FGSM/PGD)",
    "ClassifyImage":     "Classify Image (ResNet18)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
