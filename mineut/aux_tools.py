from types import ModuleType


def safe_value(key, value):
    if callable(value) or "__" in key or isinstance(value, ModuleType):
        return None
    else:
        return value


# This turns our detector modules into classes to make life easier
class Geom:
    def __init__(self, module):
        for name in dir(module):
            key, value = name, getattr(module, name)
            setattr(self, key, safe_value(key, value))


def turn_module_to_class(module):
    return Geom(module)
