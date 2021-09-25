
from importlib import import_module
import functools

import yaml
import json

def get_class(class_path, package=None):
    module_name, class_name = class_path.rsplit(".", 1)
    c = getattr(import_module(module_name, package=package), class_name)
    return c


def read_yaml(path):
    with open(path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

def write_json(path, data):
    with open(path, "w") as f:
      json.dump(data, f)

def write_yaml(path, data):
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

# https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable
import numpy as np
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def write_json_np_ok(path, data):
    with open(path, "w") as f:
      json.dump(data, f, cls=NpEncoder)
