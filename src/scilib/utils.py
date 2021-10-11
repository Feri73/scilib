import importlib.util
from types import ModuleType
from typing import Union, Any


def do_import(file: str, element: str = None, name='_tmp_') -> Union[ModuleType, Any]:
    spec = importlib.util.spec_from_file_location(name, file)
    file = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(file)
    if element is None:
        return file
    else:
        return getattr(file, element)
