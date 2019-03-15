from importlib import import_module
from importlib import resources

PLUGINS = dict()

def register_plugin(func):
    """Декоратор для регистрации плагинов"""
    name = func.__name__
    PLUGINS[name] = func
    return func

def __getattr__(name):
    """Возвращает плагин по его имени"""

    try:
        return PLUGINS[name]
    except KeyError:
        _import_plugins()
        if name in PLUGINS:
            return PLUGINS[name]
        else:
            raise AttributeError(
                f"module {__name__!r} has no attribute {name!r}"
            ) from None

def __dir__():
    """Список доступных плагинов"""
    _import_plugins()
    return list(PLUGINS.keys())

def _import_plugins():
    """Импортирует все ресурсы для регистрации плагинов"""
    for name in resources.contents(__name__):
        if name.endswith(".py"):
            import_module(f"{__name__}.{name[:-3]}")
