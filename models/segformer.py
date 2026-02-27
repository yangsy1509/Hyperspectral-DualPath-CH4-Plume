import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation


def build_segformer(
    in_channels=1,
    num_labels=1,
    config_overrides=None,
):
    config_kwargs = dict(num_channels=in_channels, num_labels=num_labels)
    if config_overrides is not None:
        config_kwargs.update(config_overrides)
    config = SegformerConfig(**config_kwargs)
    return SegformerForSemanticSegmentation(config)


def resize_logits(logits, target_hw):
    if logits.shape[-2:] != target_hw:
        logits = F.interpolate(logits, size=target_hw, mode="bilinear", align_corners=False)
    return logits