
from tools.utils import get_class

# Load the interface (NOTE: Was overided to keep the experiment cache valid in the publicly released version)
def get_interface(gcfg):
    interface_cfg = {
        "cfg_path": gcfg["override"]["model_cfg"],
        "model_path": gcfg["override"]["model_path"],
    }
    interface_name = gcfg["trigger_extract"]["interface"]
    interface = get_class(interface_name)(interface_cfg)

    return interface