from methods.methods_retrain import FullRetrainMethod
from methods.methods_sisa import SISAMethod
from methods.methods_receraser import RecEraserMethod

from models.models_bpr import BPRWrapper
from models.models_lightgcn import LightGCNWrapper
from models.models_receraser_bpr import RecEraserBPRWrapper
from models.models_receraser_lightgcn import RecEraserLightGCNWrapper


def build_model(cfg, n_users, n_items):
    method = str(getattr(cfg, "method_type", getattr(cfg, "method", "retrain"))).lower()
    model_type = str(getattr(cfg, "model_type", "bpr")).lower()

    if method == "receraser":
        if model_type == "bpr":
            return RecEraserBPRWrapper
        if model_type == "lightgcn":
            return RecEraserLightGCNWrapper
        raise ValueError(f"RecEraser does not support model_type={model_type}")

    if model_type == "bpr":
        return BPRWrapper
    if model_type == "lightgcn":
        return LightGCNWrapper

    raise ValueError(f"Unsupported model_type={model_type}")


def build_method(cfg, loader, model_class):
    method = getattr(cfg, "method_type", getattr(cfg, "method", None))
    if method is None:
        raise ValueError("Config must have method_type or method")

    method = str(method).lower()

    if method == "retrain":
        return FullRetrainMethod(cfg, loader, model_class)
    if method == "sisa":
        return SISAMethod(cfg, loader, model_class)
    if method == "receraser":
        return RecEraserMethod(cfg, loader, model_class)

    raise ValueError(f"Unsupported method={method}")