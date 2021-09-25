
from tqdm import tqdm
import numpy as np
import zarr
import pandas as pd
import torch

from tools.utils import write_json

from model_repair.override_for_release import get_interface
from model_repair.cache42.cached_42 import cache_42

# @cache_42(ignore_args=["gcfg"], force_recompute=True)
@cache_42(ignore_args=["gcfg"])
def extract_checkpoint(cfg, checkpoint_path, cfg_trigger, gcfg, cache):
    # NOTE: cfg_trigger is needed for cache invalidation (until better cache lib)

    cache_path = cache.get_path()
    
    # Load the interface
    interface = get_interface(gcfg)

    # Get the dataset
    dataloader = interface.get_dataloader(shuffle=cfg["shuffle"])

    # Get the model
    model = interface.load_model(checkpoint_path)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    torch.set_grad_enabled(True)

    all_objects_grads = []
    all_objects_features_normal = []
    all_objects_features_avg = []
    all_objects_info = []
    feats_layers_info = None
    grads_layers_info = None

    model.eval()

    count_correct = 0 # For downsampling correct points
    count_object = 0
    for i_batch, sample in tqdm(enumerate(dataloader), total=len(dataloader)):

        if cfg["get_feats"]:
            FeaturesSaver().set_must_save(True)
        
        loss, preds = interface.forward_loss(sample, grad_pos=None, output_preds=True)

        # Get objects -----
        objects_info = interface.get_objects(preds, sample) # List of dicts

        # Downsample correct points if there are many
        if "downsample_correct_mod" in cfg:
            contains_one_correct = sum([e["correct"] for e in objects_info])
            if contains_one_correct:
                count_correct += 1
            if contains_one_correct and count_correct % cfg.downsample_correct_mod != 0:
                continue

        [e.update({"example_id": i_batch}) for e in objects_info]
        for e in objects_info:
            e.update({"object_id": count_object})
            count_object += 1

        all_objects_info.extend(objects_info)

        if cfg["get_feats"]:
            feats_normal = FeaturesSaver().saved_normal

            feats_avg = FeaturesSaver().saved_avg
            
            feats_layers_info = FeaturesSaver().feats_layers_info

            if interface.requires_grad_pos():
                # If grad pos, duplicate objects for each object (and ignore if no object)
                for object_info in objects_info:
                    all_objects_features_normal.append(feats_normal)
                    all_objects_features_avg.append(feats_avg)
            else:
                all_objects_features_normal.append(feats_normal)
                all_objects_features_avg.append(feats_avg)
            
            FeaturesSaver().set_must_save(False)

        if cfg["get_grads"]:
            for object_info in objects_info:

                if interface.requires_grad_pos():
                    grad_pos = (object_info["r"], object_info["t"])
                else:
                    grad_pos = None

                loss = interface.forward_loss(sample, grad_pos=grad_pos, output_preds=False)
                
                optimizer.zero_grad()
                loss.backward(retain_graph=True)

                all_grads, grads_layers_info = get_all_grads(model, max_dim=cfg["max_dim_grads"], get_info=True)
                all_objects_grads.append(all_grads)

        if i_batch + 1 >= cfg["n_samples"]:
            break

    save_data(all_objects_grads, all_objects_features_normal, all_objects_features_avg, all_objects_info, feats_layers_info, grads_layers_info, cache_path)

def save_data(all_objects_grads, all_objects_features_normal, all_objects_features_avg, all_objects_info, feats_layers_info, grads_layers_info, out_path):

    out_path.mkdir(exist_ok=True, parents=True)

    path = str(out_path / "extracted.zarr")
    z_file = zarr.open(path, mode='w')

    print("Saving per object gradients...")
    z_per_object_grads = z_file.create_group("per_object_grads")
    N_objects = len(all_objects_grads)
    if N_objects > 0:
        N_levels = len(all_objects_grads[0])
        for li in tqdm(range(N_levels)):

            # Stack all objects for level li
            yo = [all_objects_grads[n_o][li] for n_o in range(N_objects)]
            yo = np.stack(yo, 0)

            z_li = z_per_object_grads.zeros(str(li), shape=yo.shape)
            z_li[:] = yo[:]
    else:
        print("No grad information available")

    print("Saving images_feats_normal...")
    z_feats = z_file.create_group("images_feats_normal")
    N_img = len(all_objects_features_normal)

    if N_img > 0:
        N_levels = len(all_objects_features_normal[0])
        for li in tqdm(range(N_levels)):

            yo = [all_objects_features_normal[ni][li] for ni in range(N_img)]
            yo = np.stack(yo, 0)
            z_li = z_feats.zeros(str(li), shape=yo.shape)
            z_li[:] = yo[:]
    else:
        print("No feature information available")

    print("Saving images_feats_avg...")
    z_feats = z_file.create_group("images_feats_avg")
    N_img = len(all_objects_features_avg)
    if N_img > 0:
        N_levels = len(all_objects_features_avg[0])
        for li in tqdm(range(N_levels)):

            yo = [all_objects_features_avg[ni][li] for ni in range(N_img)]
            yo = np.stack(yo, 0)

            z_li = z_feats.zeros(str(li), shape=yo.shape)
            z_li[:] = yo[:]
    else:
        print("No feature information available")

    print("Saving per object info...")
    df = pd.DataFrame(all_objects_info)
    path = out_path / "per_object_info.feather"
    df.to_feather(str(path))

    print("Saving model_info...")
    model_info = {
        "feats_layers_info": feats_layers_info,
        "grads_layers_info": grads_layers_info,
    }
    write_json(str(out_path / "model_info.json"), model_info)

_state = {}
def get_all_grads(model, get_info=False, max_dim=None):

    if max_dim is not None and "grads_rand_project" not in _state:
        _state["grads_rand_project"] = PytorchRandomProjection(max_dim)

    all_grads = []
    if get_info:
        grads_layers_info = []
    for idx, m in enumerate(model.modules()):
        
        g_shape = None
        try:
            g_shape = tuple(m.weight.grad.shape)
            g = m.weight.grad
            g = g.reshape(1, -1)
        except:
            g_shape = None
            g = None
            pass

        if g is not None:
            D = g.shape[-1]
            if max_dim is not None and D > max_dim:
                g = _state["grads_rand_project"].project(g)
            
            if get_info:
                grads_layers_info.append((str(m), g_shape))

            g = g.flatten().cpu().numpy()
            all_grads.append(g)

    if get_info:
        return all_grads, grads_layers_info
    else:
        return all_grads


from model_repair.tools.singleton import singleton
@singleton
class FeaturesSaver:

    def __init__(self, max_dim=None):
        self.must_save = False
        self.reset()

        max_dim = 100
        if max_dim is not None and "R" not in _state:
            self.feat_rand_projector = PytorchRandomProjection(max_dim)

        self.max_dim = max_dim

    
    def set_must_save(self, must_save):
        self.must_save = must_save
        self.reset()

    def reset(self):
        self.saved_normal = []
        self.saved_avg = []
        self.feats_layers_info = []

    def save(self, name, features):
        if self.must_save: # TODO
            shape = tuple(features.shape)
            B, C, H, W = features.shape

            self.feats_layers_info.append((name, shape))

            # Save normal features
            features_normal = features.view(1, C*H*W)
            D = features_normal.shape[-1]
            if self.max_dim is not None and D > self.max_dim:
                features_normal = self.feat_rand_projector.project(features_normal)

            features_normal = features_normal.detach().flatten().cpu().numpy()
            self.saved_normal.append(features_normal)

            # Save spatially avg features
            features_avg = features.mean(-1).mean(-1)
            D = features_avg.shape[-1]
            if self.max_dim is not None and D > self.max_dim:
                features_avg = self.feat_rand_projector.project(features_avg)
            
            features_avg = features_avg.detach().flatten().cpu().numpy()
            self.saved_avg.append(features_avg)

class PytorchRandomProjection:

    def __init__(self, max_proj_dim=None):
        max_dim_random = 100000

        prng = np.random.RandomState()
        prng.seed(42)

        R = prng.randn(max_dim_random, max_proj_dim)
        self.R = torch.from_numpy(R).cuda().float() * np.sqrt(1/max_proj_dim)

        self.max_dim_random = max_dim_random
        self.max_proj_dim = max_proj_dim

    def project(self, data):
        A, D = data.shape

        assert A == 1 # No batch mode here (consider torch.bmm func)

        if D < self.max_proj_dim:  # No need to project
            projected = data
        elif D < self.max_dim_random:
            projected = torch.matmul(data, self.R[:D, :])
        else:
            
            start_batch = 0
            batch_size = (D//self.max_dim_random)
            end_batch = batch_size * self.max_dim_random
            data_slice_batch = data[:, start_batch:end_batch].view(batch_size, 1, -1)

            R_batched = self.R.view(1, self.max_dim_random, self.max_proj_dim).expand(batch_size, -1, -1)
            batched_projected = torch.bmm(data_slice_batch, R_batched)

            if D != end_batch:
                missing_start = end_batch
                missing_end = D
                missing_size = D - end_batch
                data_slice_missing = data[:, missing_start:missing_end]
                missing_projected = torch.matmul(data_slice_missing, self.R[:missing_size, :])
                projected = torch.cat([batched_projected.view(1, -1), missing_projected], -1)
            else:
                projected = batched_projected.view(1, -1)

        return projected