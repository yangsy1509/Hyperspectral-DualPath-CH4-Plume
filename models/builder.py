from models.cnn_aspp import CNN_ASPP
from models.inception_unet import InceptionUNetStandard
from models.segformer import build_segformer, resize_logits


def build_model(model_cfg):
    arch = model_cfg["arch"].lower()

    if arch == "cnn_aspp":
        return CNN_ASPP(
            in_channels=model_cfg.get("in_channels", 1),
            dropout_rate=model_cfg.get("dropout_rate", 0.5),
            atrous_rates=model_cfg.get("atrous_rates", [3, 6, 12, 18]),
            activation=model_cfg.get("activation", "relu"),
            negative_slope=model_cfg.get("negative_slope", 0.01),
        ), False

    if arch == "inception_unet":
        return InceptionUNetStandard(
            n_channels=model_cfg.get("in_channels", 1),
            n_classes=model_cfg.get("n_classes", 1),
        ), False

    if arch == "segformer":
        return build_segformer(
            in_channels=model_cfg.get("in_channels", 1),
            num_labels=model_cfg.get("num_labels", 1),
            config_overrides=model_cfg.get("segformer_config", None),
        ), True

    raise ValueError(f"Unsupported arch: {arch}")


def model_forward(model, arch, inputs, targets=None):
    arch = arch.lower()
    if arch == "segformer":
        logits = model(inputs).logits
        if targets is not None:
            logits = resize_logits(logits, targets.shape[-2:])
        return logits
    return model(inputs)