import pickle

import torch
import numpy as np
import lmdb
from torch.utils.data import Dataset


class MMDataset(Dataset):
    _lmdb_env = None

    def __init__(self, db_path, dataset, augmentations):
        if not MMDataset._lmdb_env:
            MMDataset._lmdb_env = lmdb.open(
                db_path, readonly=True, max_readers=8, lock=False, readahead=True, meminit=False)
        self.augmentations = augmentations
        self.db = MMDataset._lmdb_env.open_db(dataset)

    def __getitem__(self, index):
        key = str(index).encode('ascii')
        with MMDataset._lmdb_env.begin(db=self.db, write=False) as txn:
            data = txn.get(key)
        sample = pickle.loads(data)
        sample['label'] = torch.tensor(sample['label'].astype(np.float32))
        sample['image'] = self._prepare_img_data(sample['image'])
        if self.augmentations:
            sample = self.augmentations(sample)
        return sample

    def __len__(self):
        with MMDataset._lmdb_env.begin(db=self.db, write=False) as txn:
            return txn.stat()['entries']

    @staticmethod
    def _prepare_img_data(img_data):
        # normalize values to range -1, 1
        return 2. * (torch.tensor(img_data.astype(np.float32)) / 255.) - 1.

    @staticmethod
    def close():
        if MMDataset._lmdb_env:
            MMDataset._lmdb_env.close()
