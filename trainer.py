
import torch
import tools.torch_utils
from tools.utils import get_class
import numpy as np

from torch.utils.data import DataLoader

from base_trainer import BaseTrainer
from sumagg.sumagg import SumAgg
from summaries import MeanSummary
from model.model_vgg import Model
from dataset.MedMNIST import MedMNIST, collate_fn
from loss.loss import Loss

class Trainer(BaseTrainer):

    def __init__(self, cfg):
        super().__init__(cfg)

        self.loss_f = Loss(cfg)
        
    def declare_summaries(self):
        SumAgg().add_summaries([
            MeanSummary("loss_val"),
            MeanSummary("acc"),
            MeanSummary("auc"),
            MeanSummary("preds"),
        ])

    def load_model(self):
        model = Model(self.cfg).cuda()
        print(model)
        return model


    def create_optimizer(self, model_parameters):
        optimizer = get_class(self.cfg.optimizer.name)(
            model_parameters, **self.cfg.optimizer.params
        )
        return optimizer

    def create_scheduler(self, optimizer):

        if self.cfg.scheduler.use:
            scheduler = get_class(self.cfg.scheduler.name)(optimizer, **self.cfg.scheduler.params)
        else:
            scheduler = None

        return scheduler

    def _format_sample(self, sample):
        img = [torch.tensor(v) for v in sample["img"]]
        img = torch.stack(img, 0).float()#.cuda()
        B, H, W, C = img.shape
        T = 1
        img = img.permute(0, 3, 1, 2) # B, C, H, W
        img = img.view(B, T, C, H, W) # B, T, C, H, W

        class_ = [torch.tensor(v) for v in sample["class"]]
        class_ = torch.cat(class_, 0).float()#.cuda()
        class_ = class_.view(B, T)

        return {
            "img": img,
            "class": class_,
        }

    def _chunk_sample(self, sample, chunk_size):
        B, T, C, H, W = sample["img"].shape

        chunks = []
        for s_start in range(0, T, chunk_size):
            s_end = s_start + chunk_size
            chunk = {
                "img": sample["img"][:, s_start:s_end],
                "class": sample["class"][:, s_start:s_end],
            }
            chunks.append(chunk)
        
        return chunks

    def _prepare_sample(self, sample):
        sample["img"] = sample["img"].cuda()
        return sample

    def forward(self, sample, state):

        x = sample["img"]
        # B, T, C, H, W = x.shape

        y = self.model(x)

        return y
    
    def compute_loss(self, sample, y):
        
        B, T, n_classes = y.logits.shape

        gt = sample["class"]
        
        loss = self.loss_f(y, gt)
        
        preds = self.loss_f.predict(y)

        gt_np = sample["class"].cpu().numpy()
        
        acc = (gt_np == preds).sum() / (B*T)

        SumAgg().add("acc", acc)

        from sklearn.metrics import roc_auc_score
        
        assert T == 1
        
        if len(np.unique(gt_np[:, 0].sum())) < 2:
            auc = np.nan
        else:
            auc = roc_auc_score(gt_np[:, 0], y.logits.softmax(-1)[:,0, 1].detach().cpu().numpy())
        SumAgg().add("auc", auc)
        
        return loss

    def get_current_score(self):
        current_score = SumAgg().epoch_metrics["acc"]
        return current_score



    def get_dataloaders(self):

        dataloader_kwargs = dict(
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.n_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        # Train
        dataset = MedMNIST(
            cfg=self.cfg,
            split="train",
        )
        dataloader_train = DataLoader(
            dataset,
            **dataloader_kwargs
        )

        # Val
        dataset = MedMNIST(
            cfg=self.cfg,
            split="val"
        )
        dataloader_val = DataLoader(
            dataset,
            **dataloader_kwargs
        )

        # Test

        dataset = MedMNIST(
            cfg=self.cfg,
            split="test",
        )
        dataloader_test = DataLoader(
            dataset,
            **dataloader_kwargs
        )

        return dataloader_train, dataloader_val, dataloader_test