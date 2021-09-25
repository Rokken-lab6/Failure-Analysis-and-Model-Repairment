
from pathlib import Path
import warnings

import numpy as np
from box import Box
from torch.utils.data import DataLoader
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_auc_score

from model_repair.abstract_interface import AbstractInterface

from dataset.MedMNIST import MedMNIST, collate_fn
from model.model_vgg import Model
from loss.loss import Loss
from sumagg.sumagg import SumAgg

import tools.torch_utils

class MedMNISTInterface(AbstractInterface):

    def __init__(self, int_cfg):
        super().__init__(int_cfg)

        SumAgg(disable=True)

        with open(self.int_cfg["cfg_path"], 'r') as stream:
            self.cfg = Box.from_yaml(stream)

        self.loss_f = Loss(self.cfg)

    def get_checkpoints(self):
        # Output list of checkpoints
        base_path = Path(self.int_cfg["model_path"])
        checkpoints = [p for p in base_path.glob("*.pytorch") if "best" not in str(p)]
        checkpoints = sorted(checkpoints, key= lambda e: int(e.stem.lstrip("model")))
        return checkpoints

    def get_checkpoint_path(self, epoch):
        p = Path(self.int_cfg["model_path"])
        
        return p / f"model{epoch}.pytorch"

    def load_model(self, checkpoint):
        self.model = Model(self.cfg).cuda()

        filename = str(checkpoint)
        self.model.load_state_dict(torch.load(filename))
        
        return self.model

    def forward_loss(self, sample, grad_pos=None, output_preds=False):
        assert grad_pos == None, "No grad pos for MedMNIST classification"

        sample = self._format_sample(sample)
        sample = self._prepare_sample(sample)

        y = self.model(sample["img"])
        
        B, T, N_classes = y.logits.shape
        assert B == 1 # TODO
        assert T == 1

        gt = sample["class"]
        
        loss = self.loss_f(y, gt)

        if output_preds:
            caths_preds = self.loss_f.predict(y)[:,0] # Remove T dimension...

            # NOTE: Assume B=1 and T=1...
            conf = y.logits.softmax(-1)[0,0, 1].cpu().detach().numpy().item()
            caths_preds = [caths_preds, conf]
            return loss, caths_preds
        else:
            return loss

    def requires_grad_pos(self):
        return False

    def get_objects(self, preds, sample):
        preds, conf = preds
        pred_class = preds[0]
        gt_class = sample["class"][0][0]
        
        wrong_labels = None
        if wrong_labels is not None:
            warnings.warn("wrong_labels mode")
            assert gt_class <= 1 # NOTE: Assumes binary label
            if sample["idx"][0] % wrong_labels == 0:
                gt_class = 1 - gt_class
                label_is_wrong = True
            else:
                label_is_wrong = False
        else:
            label_is_wrong = False


        is_correct = pred_class == gt_class
        
        objects_info = [{
            "frame_idx": sample["idx"][0],

            "correct": is_correct,
            "pred": pred_class,
            "gt": gt_class,
            "label_group": sample["label_group"][0],
            "conf": conf,

            "label_is_wrong": label_is_wrong,
        }]

        return objects_info


    def get_dataloader(self, specific_frames=None, shuffle=True):
        
        dataset = self._get_dataset(specific_frames=specific_frames)

        dataloader = DataLoader(
            dataset,
            batch_size=1, # TODO
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        
        return dataloader

    def _get_dataset(self, specific_frames=None):
        dataset = MedMNIST(
            cfg=self.cfg,
            split="test",
            specific_frames=specific_frames,
        )
        self.dataset = dataset
        return dataset

    def get_example_key(self):
        # Key in the sample that identify every example case (e.g. an image, can contain multiple objects)
        return "frame_idx"

    def identify_cluster(self, new_objs, old_objs, frame_name):
        # cluster = -2: New object
        # otherwise same as matched object (easy for classification, less for object detection)

        assert len(new_objs) == 1 # TODO: Batching mode

        cluster = old_objs[old_objs["frame_idx"]==new_objs["frame_idx"].item()].iloc[0]["cluster"]
        new_objs["cluster"] = cluster

        return new_objs


    def compute_metrics(self, objs):

        if len(objs) == 0:
            auc = np.nan
        elif objs["gt"].min() != objs["gt"].max():
            auc = roc_auc_score(objs["gt"], objs["conf"])
        else:
            auc = np.nan

        return {
            "accuracy": float(objs["correct"].mean()),
            "loss": float(objs["loss"].mean()),
            "auc": auc,
        }

    def validation_metric(self):
        return "accuracy"
    
    def export_datapoint(self, obj, dist, order, cluster_dir_path):
        try:
            dataset = self.dataset
        except:
            dataset = self._get_dataset()

        img = dataset.img[obj["frame_idx"]]
        cluster_dir_path = Path(cluster_dir_path)
        cluster_dir_path.mkdir(parents=True, exist_ok=True)

        if img.shape[2] == 1:
            img = img[:,:,0]
        im = Image.fromarray(img)
        im.save(str(cluster_dir_path / f"{order}_i{obj['frame_idx']}_g{obj['gt']}_p{obj['pred']}_{dist}.png"))

    def plot_cluster_description(self, show_objs, show_dists, cluster_dir_path):

        cluster_dir_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            dataset = self.dataset
        except:
            dataset = self._get_dataset()

        N_objects = len(show_objs)
        N_objects = min(N_objects, 12) # max objects limits

        fig = plt.figure()
        
        n_cols = 4
        n_rows = N_objects // n_cols
        if N_objects % n_cols != 0:
            n_rows += 1

        spec = gridspec.GridSpec(ncols=n_cols, nrows=n_rows, figure=fig)
        
        spec.update(wspace = 0.0, hspace = 0.3)

        fig.suptitle(f"Cluster {show_objs[0]['cluster'] + 1}")

        for i in range(N_objects):
            obj = show_objs[i]
            dist = show_dists[i]
            img = dataset.img[obj["frame_idx"]] 

            ax = fig.add_subplot(spec[i])
            ax.imshow(img)

            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax.title.set_text(f"Gt:{obj['gt']},P{obj['pred']},L:{obj['label_group']}")

        plt.savefig(str(cluster_dir_path))
        plt.close()

    def get_cluster_stats_keys(self):
        return {
            "keys": ["pred", "gt", "label_is_wrong", "label_group"], # For catheters: add tp, fp, fn, tn, etc
            "cross_keys": [("pred", "gt"), ("gt", "label_group")],
        }


    # INTERNALS -----------------------------------------------------------------

    def _format_sample(self, sample):

        img = [torch.tensor(v) for v in sample["img"]]
        img = torch.stack(img, 0).float()
        B, H, W, C = img.shape
        T = 1
        img = img.permute(0, 3, 1, 2) # B, C, H, W
        img = img.view(B, T, C, H, W) # B, T, C, H, W

        class_ = [torch.tensor(v) for v in sample["class"]]
        class_ = torch.cat(class_, 0).float()
        class_ = class_.view(B, T)

        out = {
            "img": img,
            "class": class_,
        }

        if sample["metalabel"][0] is not None:
            metalabel = [torch.tensor(v) for v in sample["metalabel"]]
            metalabel = torch.stack(metalabel, 0).float()
            metalabel = metalabel.view(B, metalabel.shape[1])

            out["metalabel"] = metalabel


        if sample["label_group"][0] is not None:
            label_group = [torch.tensor(v) for v in sample["label_group"]]
            label_group = torch.stack(label_group, 0).float()
            label_group = label_group.view(B)

            out["label_group"] = label_group
        
        return out

    def _prepare_sample(self, sample):
        sample["img"] = sample["img"].cuda()
        return sample