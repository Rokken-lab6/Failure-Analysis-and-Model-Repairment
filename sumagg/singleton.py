from functools import wraps

__instances = {}


def singleton(cls):
    @wraps(cls)
    def getInstance(*args, **kwargs):
        instance = __instances.get(cls, None)
        if not instance:
            instance = cls(*args, **kwargs)
            __instances[cls] = instance
        return instance

    return getInstance
