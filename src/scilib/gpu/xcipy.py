try:
    import cupyx.scipy as _xp
    from . import _cupyx_scipy_stats

    _xp.stats = _cupyx_scipy_stats
except ImportError as e:
    print('Falling back on scipy:', e)
    import scipy as _xp
import sys

sys.modules[__name__] = _xp
