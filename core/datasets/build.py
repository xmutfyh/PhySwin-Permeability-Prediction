import inspect
from typing import Any, Dict, Optional

def build_from_cfg(cfg: Dict,
                   registry: 'Registry',
                   default_args: Optional[Dict] = None) -> Any:
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
    if 'type' not in cfg and (default_args is None or 'type' not in default_args):
        raise KeyError('`cfg` or `default_args` must contain the key "type"')

    if not isinstance(registry, Registry):
        raise TypeError('registry must be a Registry object')

    args = cfg.copy()
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    obj_type = args.pop('type')
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(f'{obj_type} is not in the {registry.name} registry')
    elif inspect.isclass(obj_type) or inspect.isfunction(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(f'type must be a str or valid type, but got {type(obj_type)}')

    try:
        return obj_cls(**args)
    except Exception as e:
        raise type(e)(f'{obj_cls.__name__}: {e}')

class Registry:
    def __init__(self, name: str, build_func=None, parent: Optional['Registry']=None, scope: Optional[str]=None):
        self._name = name
        self._module_dict = dict()
        self._children = dict()
        self._scope = scope or self.infer_scope()
        self.build_func = build_func or build_from_cfg
        self.parent = parent
        if parent:
            assert isinstance(parent, Registry)
            parent._add_children(self)

    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, key):
        return key in self._module_dict

    def __repr__(self):
        return f'{self.__class__.__name__}(name={self._name}, items={list(self._module_dict.keys())})'

    @property
    def name(self):
        return self._name

    @staticmethod
    def infer_scope():
        frame = inspect.currentframe()
        infer_scope_caller = frame.f_back.f_back if frame and frame.f_back else None
        try:
            filename = inspect.getmodule(infer_scope_caller).__name__
            return filename.split('.')[0] if filename else None
        except Exception:
            return None

    def _add_children(self, child: 'Registry'):
        self._children[child.name] = child

    def register_module(self, name: Optional[str]=None, force: bool=False, exist_ok: bool=True, module=None):
        """Register class/function to registry.

        Args:
            name: optional name, default to cls.__name__
            force: if True, overwrite existing entry
            exist_ok: if True, silently ignore if already registered
            module: direct class/function to register (when used as function)
        """
        if module is not None:
            module_name = name or module.__name__
            if module_name in self._module_dict:
                if force:
                    self._module_dict[module_name] = module
                elif exist_ok:
                    # silently keep the existing registration
                    return module
                else:
                    raise KeyError(f'{module_name} is already registered in {self._name}')
            else:
                self._module_dict[module_name] = module
            return module

        def _register(cls):
            cls_name = name or cls.__name__
            if cls_name in self._module_dict:
                if force:
                    self._module_dict[cls_name] = cls
                elif exist_ok:
                    # silently keep the existing registration
                    return cls
                else:
                    raise KeyError(f'{cls_name} is already registered in {self._name}')
            else:
                self._module_dict[cls_name] = cls
            return cls
        return _register

    def get(self, key: str):
        return self._module_dict.get(key, None)

    def build(self, cfg: Dict, default_args: Optional[Dict]=None):
        return self.build_func(cfg, self, default_args)

PIPELINES_REGRESSION = Registry('PIPELINES_REGRESSION')
