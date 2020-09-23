import pickle

import torch
import numpy as np
import lmdb
from torch.utils.data import Dataset


class MMDataset(Dataset):
    _lmdb_env = None

    def __init__(self, db_path, dataset, augmentations=None):
        if not MMDataset._lmdb_env:
            MMDataset._lmdb_env = lmdb.open(
                db_path, readonly=True, max_readers=8, lock=False, readahead=True, meminit=False, max_dbs=3)
        self.augmentations = augmentations
        self.db = MMDataset._lmdb_env.open_db(dataset.encode('ascii'))
        self.keys = self._load_keys()

    def __getitem__(self, index):
        key = self.keys[index]
        with MMDataset._lmdb_env.begin(db=self.db, write=False) as txn:
            data = txn.get(key)
        sample = pickle.loads(data)
        sample['label'] = torch.tensor(sample['label'].astype(np.float32))
        sample['image'] = self._prepare_img_data(sample['image'])
        if self.augmentations:
            sample = self.augmentations(sample)
        return sample

    def __len__(self):
        return len(self.keys)

    def _load_keys(self):
        with MMDataset._lmdb_env.begin(db=self.db, write=False) as txn:
            cursor = txn.cursor()
            cursor.first()
            keys = [key for key in cursor.iternext(keys=True, values=False)]
        return keys

    @staticmethod
    def _prepare_img_data(img_data):
        # normalize values to range -1, 1
        return 2. * (torch.tensor(img_data.astype(np.float32)) / 255.) - 1.

    @staticmethod
    def close():
        if MMDataset._lmdb_env:
            MMDataset._lmdb_env.close()
