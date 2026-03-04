import torch.nn as nn
import copy
import warnings

class BaseModule(nn.Module):
    """
    ``BaseModule`` is a wrapper of ``torch.nn.Module`` with additional
    functionality of parameter initialization. Compared with
    ``torch.nn.Module``, ``BaseModule`` mainly adds three attributes.
    - ``init_cfg``: the config to control the initialization.
    - ``init_weights``: The function of parameter initialization and recording
      initialization information.
    - ``_params_init_info``: Used to track the parameter initialization
      information. This attribute only exists during executing the
      ``init_weights``.
    Args:
        init_cfg (dict, optional): Initialization config dict.
    """
    def __init__(self, init_cfg=None):
        """Initialize BaseModule, inherited from `torch.nn.Module`"""

        super(BaseModule, self).__init__()
        self._is_init = False
        self.init_cfg = copy.deepcopy(init_cfg)

    @property
    def is_init(self):
        return self._is_init

    def init_weights(self):
        """Initialize the weights."""

        from core.initialize import initialize

        if not self._is_init:
            if self.init_cfg:
                initialize(self, self.init_cfg)
                if isinstance(self.init_cfg, dict):
                    if self.init_cfg.get('type') == 'Pretrained':
                        self._is_init = True
                        return

            for m in self.children():
                if hasattr(m, 'init_weights'):
                    m.init_weights()

            self._is_init = True
        else:
            warnings.warn(f'init_weights of {self.__class__.__name__} has '
                          f'been called more than once.')

class Sequential(BaseModule, nn.Sequential):
    """Sequential module for regression tasks in openmmlab.

    Args:
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(self, *args, init_cfg=None):
        BaseModule.__init__(self, init_cfg)
        nn.Sequential.__init__(self, *args)
        self.init_weights()

    def init_weights(self):
        if not self.is_init:
            """Override the init_weights method to include regression-specific initialization."""
            super(Sequential, self).init_weights()
        # Optionally, you can add custom weight initialization here.

class ModuleList(BaseModule, nn.ModuleList):
    """ModuleList for regression tasks in openmmlab.

    Args:
        modules (iterable, optional): an iterable of modules to add.
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(self, modules=None, init_cfg=None):
        BaseModule.__init__(self, init_cfg)
        nn.ModuleList.__init__(self, modules)
        self.init_weights()

    def init_weights(self):
         if not self.is_init:
            super(ModuleList, self).init_weights()

class ModuleDict(BaseModule, nn.ModuleDict):
    """ModuleDict for regression tasks in openmmlab.

    Args:
        modules (dict, optional): a mapping (dictionary) of (string: module)
            or an iterable of key-value pairs of type (string, module).
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(self, modules=None, init_cfg=None):
        BaseModule.__init__(self, init_cfg)
        nn.ModuleDict.__init__(self, modules)
        self.init_weights()

    def init_weights(self):
        if not self.is_init:
            super(ModuleDict, self).init_weights()
