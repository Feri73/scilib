from typing import Iterable, Dict, Any
import numpy.typing as npt
import matplotlib


def get_colors(elements: Iterable, cname='gist_rainbow') -> Dict[Any, npt.NDArray]:
    len = sum(1 for _ in elements)
    cmap = matplotlib.cm.get_cmap(cname)
    colors = {}
    for i, e in enumerate(elements):
        colors[e] = cmap(i / len)
    return colors
