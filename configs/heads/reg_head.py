import warnings

from configs.common import BaseModule
from configs.losses import HuberLoss
from configs.losses.mse_loss import MSELoss


def _register_if_possible(cls=None, name=None):
    """
    兼容性注册器：
    - 如果项目里存在某个注册表（如 HEADS / MODELS / 等），就把类注册进去
    - 如果不存在注册表，就什么都不做（返回原类），避免报错

    用法（常见两种）：
      @_register_if_possible()
      class XXX: ...

      @_register_if_possible(name="xxx")
      class XXX: ...
    或者直接：
      XXX = _register_if_possible(XXX)
    """
    def _do_register(_cls):
        # 你项目里是否有 registry 并不确定，所以全部 try/except，保证不炸
        registries = []

        # 常见命名：configs.registry 里有 HEADS / MODELS 等
        try:
            from configs.registry import HEADS  # type: ignore
            registries.append(HEADS)
        except Exception:
            pass

        try:
            from configs.registry import MODELS  # type: ignore
            registries.append(MODELS)
        except Exception:
            pass

        # 也可能在别的地方：configs.builder / configs.build 等（未知就宽松尝试）
        try:
            from configs.builder import HEADS as HEADS2  # type: ignore
            registries.append(HEADS2)
        except Exception:
            pass

        try:
            from configs.builder import MODELS as MODELS2  # type: ignore
            registries.append(MODELS2)
        except Exception:
            pass

        # 真正注册：registry 需要有 register_module 方法
        _name = name if name is not None else getattr(_cls, "__name__", None)
        for reg in registries:
            try:
                if hasattr(reg, "register_module"):
                    # 兼容不同 register_module 签名：有的支持 name=，有的不支持
                    try:
                        reg.register_module(name=_name)(_cls)
                    except TypeError:
                        reg.register_module()(_cls)
            except Exception:
                # 注册失败不影响运行（保证兼容）
                pass

        return _cls

    # 作为装饰器使用：@_register_if_possible(...)
    if cls is None:
        def _decorator(_cls):
            return _do_register(_cls)
        return _decorator

    # 直接调用：_register_if_possible(Class)
    return _do_register(cls)


class RegHead(BaseModule):
    """回归头部类。

    Args:
        loss (dict): 回归损失函数的配置。
        init_cfg (dict | None): 初始化配置字典。默认: None。
    """

    def __init__(self, loss=dict(type='HuberLoss', loss_weight=1.0), init_cfg=None):
        super(RegHead, self).__init__(init_cfg=init_cfg)
        assert isinstance(loss, dict), "loss 应该是一个字典"
        self.compute_loss = eval(loss.pop('type'))(**loss)

    def loss(self, pred, gt, **kwargs):
        losses = dict()
        loss = self.compute_loss(pred, gt, **kwargs)
        losses['loss'] = loss
        return losses

    def forward_train(self, pred, gt, **kwargs):
        losses = self.loss(pred, gt, **kwargs)
        return losses

    def simple_test(self, pred, post_process=False):
        if post_process:
            return self.post_process(pred)
        else:
            return pred

    def post_process(self, pred):
        return pred
