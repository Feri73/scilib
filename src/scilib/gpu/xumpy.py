try:
    import cupy as _xp

    unravel_index = _xp.unravel_index
    _xp.unravel_index = lambda x, shape, **kwargs: unravel_index(x, shape, **kwargs)
except ImportError as e:
    print('Falling back on numpy:', e)
    import numpy as _xp

    _xp.asnumpy = lambda x: x
import sys

sys.modules[__name__] = _xp
