from mmengine import OPTIM_WRAPPER_CONSTRUCTORS
from mmengine.optim import OptimWrapperDict, build_optim_wrapper


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class MultiOptimWrapperConstructor:
    def __init__(self, optim_wrapper_cfg, **kwargs):
        self.cfg = optim_wrapper_cfg

    def __call__(self, model):
        optim = {}
        for key, cfg in self.cfg.items():
            optim[key] = build_optim_wrapper(getattr(model, key), cfg)
        return OptimWrapperDict(**optim)
