from abc import ABC, abstractmethod, ABCMeta
from typing import Callable, List, Optional, Dict, Any, Tuple


class NodeMeta(ABCMeta):
    special_methods = ['add', 'sub', 'mul', 'floordiv', 'truediv', 'mod', 'pow', 'lshift', 'rshift', 'and', 'xor', 'or',
                       'radd', 'rsub', 'rmul', 'rfloordiv', 'rtruediv', 'rmod', 'rpow', 'rlshift', 'rrshift', 'rand',
                       'rxor', 'ror',
                       'neg', 'pose', 'abs', 'invert', 'complex', 'int', 'long', 'float', 'oct', 'hex',
                       'lt', 'le', 'eq', 'ne', 'ge', 'gt',
                       'len', 'contains', 'getitem', 'call', 'index']

    def __new__(mcs, name, bases, namespace, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        for method in NodeMeta.special_methods:
            cls.set_special_method(f'__{method}__')
        cls.__getattr__ = lambda self, item: Function(lambda v: getattr(v, item))(self)
        return cls

    def set_special_method(cls, method: str):
        def apply_operator(v, *a, **kws):
            ret = getattr(v, method)(*a, **kws)
            if ret is NotImplemented:
                if method == '__gt__':
                    new_method = '__lt__'
                elif method == '__ge__':
                    new_method = '__le__'
                elif method == '__lt__':
                    new_method = '__gt__'
                elif method == '__le__':
                    new_method = '__ge__'
                else:
                    new_method = f'__r{method[2:]}'
                ret = getattr(a[0], new_method)(v)
            return ret

        def get_special(self, *args, **kwargs):
            return Function(apply_operator)(self, *args, **kwargs)

        setattr(cls, method, get_special)


class Node(ABC, metaclass=NodeMeta):
    __global_node_num = 0

    def __init__(self, name: str = None):
        if name is None:
            name = f'{self.__class__.__name__}_{self.__global_node_num}'
        self.__name = name
        self.__global_node_num += 1
        self.__is_active = True

    @property
    def name(self):
        return self.__name

    @property
    def is_active(self) -> bool:
        return self.__is_active

    @property
    def de(self) -> 'Node':
        self.__is_active = False
        return self

    @property
    def re(self) -> 'Node':
        self.__is_active = True
        return self

    @abstractmethod
    def _update_dict(self, _node_dict: Dict['Node', Any], _key: Optional[str], *args, **kwargs) \
            -> Tuple[tuple, Dict[str, Any]]:
        pass

    def eval(self, *args, **kwargs):
        orig_args_len = len(args)
        node_dict = {}
        args, kwargs = self._update_dict(node_dict, None, *args, **kwargs)
        if len(args) > 0 or len(kwargs) > 0:
            raise TypeError(f'{self} requires {orig_args_len - len(args)} number of positional arguments '
                            f' but {orig_args_len} were provided.'
                            f' Also the keywords {list(kwargs.keys())} are not valid.')
        return node_dict[self]

    @property
    def f(self) -> 'Function':
        return Function(self.eval)

    def __str__(self) -> str:
        return self.name

    def _get_inner_state(self):
        return self.__is_active

    def _set_inner_state(self, is_active: bool):
        self.__is_active = is_active


class Variable(Node):
    def __init__(self, func: Callable, name: Optional[str] = None, *nodes: 'Node', **kwnodes: 'Node'):
        super().__init__(name)
        self.__func = func
        self.__nodes = nodes
        self._kwnodes = kwnodes

    def _update_dict(self, _node_dict: Dict[Node, Any], _key: Optional[str], *args, **kwargs) \
            -> Tuple[tuple, Dict[str, Any]]:
        nodes_vals = []
        kwnodes_vals = {}
        for node in self.__nodes:
            if node not in _node_dict:
                args, kwargs = node._update_dict(_node_dict, None, *args, **kwargs)
            nodes_vals.append(_node_dict[node])
        for kw, node in self._kwnodes.items():
            if node not in _node_dict:
                args, kwargs = node._update_dict(_node_dict, kw, *args, **kwargs)
            kwnodes_vals[kw] = _node_dict[node]
        _node_dict[self] = self.__func(*nodes_vals, **kwnodes_vals)
        return args, kwargs

    @staticmethod
    def _recreate(func: Callable, name: Optional[str], nodes: List['Node'], kwnodes: Dict[str, 'Node'], state: Any):
        ret = Variable(func, name, *nodes, **kwnodes)
        ret._set_inner_state(state)
        return ret

    def __reduce__(self):
        return self._recreate, (self.__func, self.name, self.__nodes, self._kwnodes, self._get_inner_state())


class Constant(Variable):
    def __init__(self, const_val, name: str = None):
        super().__init__(lambda *args, **kwargs: const_val, name or f'Constant_{const_val}')


Const = Constant


class PlaceHolder(Node):
    __ph_names = {}

    def __new__(cls, name: str = None) -> 'PlaceHolder':
        if name in cls.__ph_names:
            return cls.__ph_names[name]
        else:
            return super().__new__(cls)

    def __init__(self, name: str = None):
        if name not in self.__ph_names:
            if name is not None:
                self.__ph_names[name] = self
            super().__init__(name)

    def _update_dict(self, _node_dict: Dict['Node', Any], _key: Optional[str], *args, **kwargs) \
            -> Tuple[tuple, Dict[str, Any]]:
        if self.name in kwargs:
            _node_dict[self] = kwargs.pop(self.name)
        elif _key in kwargs:
            _node_dict[self] = kwargs.pop(_key)
        else:
            _node_dict[self] = args[0]
            args = args[1:]
        return args, kwargs

    @property
    def forget(self) -> 'PlaceHolder':
        del self.__ph_names[self.name]
        return self

    @staticmethod
    def _recreate(name: Optional[str], state: Any):
        ret = PlaceHolder(name)
        ret._set_inner_state(state)
        return ret

    def __reduce__(self):
        return self._recreate, (self.name, self._get_inner_state())


PH = PlaceHolder


class Function:
    Member = Callable[..., Optional[Callable[[Variable], Variable]]]
    __members: List[Member] = []

    @staticmethod
    def register(member: Member) -> None:
        Function.__members.append(member)

    def __init__(self, func: Callable):
        self.__func = func

    def __call__(self, *args, **kwargs) -> Variable:
        res = Variable(self.__func,
                       f'{self.__func}(' + ','.join(map(str, args)) +
                       ','.join(map(lambda k: f'{k}={kwargs[k]}', kwargs)) + ')',
                       *[arg if isinstance(arg, Node) and arg.is_active else Constant(arg) for arg in args],
                       **{kw: arg if isinstance(arg, Node) and arg.is_active else Constant(arg)
                          for kw, arg in kwargs.items()})
        for member in Function.__members:
            cls = member(*args, **kwargs)
            if cls is not None:
                return cls(res)
        return res

    @staticmethod
    def _recreate(func: Callable):
        return Function(func)

    def __reduce__(self):
        return self._recreate, (self.__func,)


Func = Function


class IfClause(Function):
    def __new__(cls, *args, **kwargs):
        clause = super(IfClause, cls).__new__(cls)
        clause.__init__()
        return clause(*args, **kwargs)

    def __init__(self):
        super().__init__(lambda cond, true_res, false_res: true_res if cond else false_res)


If = IfClause


class ForClause(Function):
    def __new__(cls, *args, **kwargs):
        clause = super(ForClause, cls).__new__(cls)
        clause.__init__()
        return clause(*args, **kwargs)

    def __init__(self):
        def func(init, step, end, task):
            v = None
            for i in range(init, end, step):
                v = task(i, v)
            return v

        super().__init__(func)


For = ForClause
