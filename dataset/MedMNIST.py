# -*- coding: utf8 -*-

from pathlib import Path

from torch.utils.data import Dataset
import numpy as np

class MedMNIST(Dataset):

    def __init__(self, cfg, split, specific_frames=None):

        npz_path = cfg.dataset_path
        npz_file = np.load(npz_path)

        if "increase_test_size" not in cfg or not cfg["increase_test_size"]:
            if split == 'train':
                self.img = npz_file['train_images']
                self.label = npz_file['train_labels']
            elif split == 'val':
                self.img = npz_file['val_images']
                self.label = npz_file['val_labels']
            elif split == 'test':
                self.img = npz_file['test_images']
                self.label = npz_file['test_labels']
        else:
            original_train_size = npz_file['train_images'].shape[0]
            original_val_size = npz_file['val_images'].shape[0]
            take_train = int(cfg["train_to_test"]*original_train_size)
            take_val = int(cfg["val_to_test"]*original_val_size)
            
            prng = np.random.RandomState()
            prng.seed(42)

            idx_into_train = np.arange(original_train_size)
            prng.shuffle(idx_into_train)
            idx_into_val = np.arange(original_val_size)
            prng.shuffle(idx_into_val)

            shuffled_train_img = npz_file['train_images'][idx_into_train]
            shuffled_val_img = npz_file['val_images'][idx_into_val]

            shuffled_train_label = npz_file['train_labels'][idx_into_train]
            shuffled_val_label = npz_file['val_labels'][idx_into_val]

            if split == 'train':
                new_train_img = shuffled_train_img[take_train:]
                new_train_label = shuffled_train_label[take_train:]

                self.img = new_train_img
                self.label = new_train_label
            elif split == 'val':

                new_val_img = shuffled_val_img[take_val:]
                new_val_label = shuffled_val_label[take_val:]
                
                self.img = new_val_img
                self.label = new_val_label
            elif split == 'test':
                new_test_img = np.concatenate((npz_file['test_images'], shuffled_train_img[:take_train], shuffled_val_img[:take_val]), 0)
                new_test_label = np.concatenate((npz_file['test_labels'], shuffled_train_label[:take_train], shuffled_val_label[:take_val]), 0)

                self.img = new_test_img
                self.label = new_test_label


        if len(self.img.shape) == 3:
            N, H, W = self.img.shape
            self.img = self.img.reshape(N, H, W, 1)

        self.metalabel = None
        self.label_group = None

        if "dataset_key" in cfg and cfg.dataset_key == "pathmnist":
            self.metalabel = self.label
            self.label_group = self.label[:, 0]

            if cfg.binary_mode:
                self.label = (self.label <= 6).astype(int) * 0.0 + (self.label > 6).astype(int) * 1.0
  
        if specific_frames is not None:
            self.real_idx = specific_frames
        else:
            self.real_idx = range(len(self.img))

    def __len__(self):
        return len(self.real_idx)

    def __getitem__(self, idx):
        
        real_id = self.real_idx[idx]

        img, target = self.img[real_id], self.label[real_id].astype(int)

        if self.metalabel is not None:
            metalabel = self.metalabel[real_id].astype(int)
        else:
            metalabel = None

        if self.label_group is not None:
            label_group = self.label_group[real_id].astype(int)
        else:
            label_group = None
            
        sample = {
            "idx": real_id,
            "frame_idx": real_id,
            "img": img,
            "class": target,
            "metalabel": metalabel,
            "label_group": label_group,
        }

        return sample

def collate_fn(batch):
    idx = [b['idx'] for b in batch]
    frame_idx = [b['frame_idx'] for b in batch]
    img = [b['img'] for b in batch]
    class_ = [b['class'] for b in batch]
    metalabel = [b['metalabel'] for b in batch]
    label_group = [b['label_group'] for b in batch]

    return {
        'idx': idx,
        'frame_idx': frame_idx,
        'img': img,
        'class': class_,
        'metalabel': metalabel,
        'label_group': label_group,
    }