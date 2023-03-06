import glob
import io
import logging
import unicodedata
from pathlib import Path, PurePath
from typing import Callable, Optional, Union, Tuple, List

import random
import numpy as np
import re
import torch
import lmdb
from PIL import Image
import cv2
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms

from data.utils import CharsetAdapter
from data.transforms import ImageToArray, ImageToPIL, CVColorJitter, CVDeterioration, CVGeometry

log = logging.getLogger(__name__)

def build_tree_dataset(root: Union[PurePath, str], *args, **kwargs):
    try:
        kwargs.pop('root')  # prevent 'root' from being passed via kwargs
    except KeyError:
        pass
    root = Path(root).absolute()
    log.info(f'dataset root:\t{root}')

    consistency_regularization = kwargs.get('consistency_regularization', None)
    exclude_folder = kwargs.get('exclude_folder', [])
    try :
        kwargs.pop('exclude_folder')
    except KeyError : # prevent 'exclude_folder' passed to Dataset
        pass 


    datasets = []
    for mdb in glob.glob(str(root / '**/data.mdb'), recursive=True):
        mdb = Path(mdb)
        ds_name = str(mdb.parent.relative_to(root))
        ds_root = str(mdb.parent.absolute())

        if str(mdb.parts[-2]) in exclude_folder : # child folder name in exclude_folder e.g) Vertical
            continue

        if consistency_regularization :
            dataset = ConsistencyRegulariationLmdbDataset(ds_root, *args, **kwargs)
        else : 
            dataset = LmdbDataset(ds_root, *args, **kwargs)
        log.info(f'\tlmdb:\t{ds_name}\tnum samples: {len(dataset)}')
        print(f'\tlmdb:\t{ds_name}\tnum samples: {len(dataset)}')
        datasets.append(dataset)
    return ConcatDataset(datasets)

class LmdbDataset(Dataset):
    """Dataset interface to an LMDB database.

    It supports both labelled and unlabelled datasets. For unlabelled datasets, the image index itself is returned
    as the label. Unicode characters are normalized by default. Case-sensitivity is inferred from the charset.
    Labels are transformed according to the charset.
    """

    def __init__(self, root: str, charset: str, max_label_len: int, min_image_dim: int = 0,
                 remove_whitespace: bool = True, normalize_unicode: bool = True,
                 unlabelled: bool = False, transform: Optional[Callable] = None,
                 limit_size: bool = False, size_of_limit: Optional[int] = None, img_size: Optional[Tuple] = (32,128), 
                 consistency_regularization = False, is_training: bool = False, twinreader_folders: Optional[List] = []):
        self._env = None
        self.root = root
        self.unlabelled = unlabelled
        self.transform = transform
        self.labels = []
        self.filtered_index_list = []
        self.index_list = []
        self.img_size = img_size
        self.limit_size = limit_size
        self.size_of_limit = size_of_limit
        self.min_image_dim = min_image_dim
        self.normalize_unicode = normalize_unicode
        self.remove_whitespace = remove_whitespace
        self.max_label_len = max_label_len
        self.is_training = is_training
        self.charset_adapter = CharsetAdapter(charset)
        
        if self.limit_size and self.root.split('/')[-1] in twinreader_folders:
            self.limit_size = False
        
        self.consistency_regularization = consistency_regularization
        
        if not self.is_training:
            self.num_samples = self._preprocess_labels(charset, remove_whitespace, normalize_unicode,
                                                   max_label_len, min_image_dim)
        else:
            self.num_samples = self.train_num_samples()
            
    def train_num_samples(self):
        with self._create_env() as env, env.begin() as txn:
            num_samples = int(txn.get('num-samples'.encode()))
            if self.unlabelled:
                return num_samples
            self.index_list = range(num_samples)
            if self.limit_size :
                self.index_list = np.random.permutation(self.index_list)[:self.size_of_limit]
            return len(self.index_list)

    def __del__(self):
        if self._env is not None:
            self._env.close()
            self._env = None

    def _create_env(self):
        return lmdb.open(self.root, max_readers=1, readonly=True, create=False,
                         readahead=False, meminit=False, lock=False)

    @property
    def env(self):
        if self._env is None:
            self._env = self._create_env()
        return self._env

    def _preprocess_labels(self, charset, remove_whitespace, normalize_unicode, max_label_len, min_image_dim):
        charset_adapter = CharsetAdapter(charset)
        with self._create_env() as env, env.begin() as txn:
            num_samples = int(txn.get('num-samples'.encode()))
            if self.unlabelled:
                return num_samples
            index_list = range(num_samples)
            if self.limit_size :
                index_list = np.random.permutation(index_list)[:self.limit_size]
                
            for index in index_list : #range(num_samples):
                index += 1  # lmdb starts with 1
                label_key = f'label-{index:09d}'.encode()
                label = txn.get(label_key).decode()
                #
                label = edit_label(label, charset_adapter.charset) # edit self.charset in CharsetAdapter
                if len(label) > max_label_len:
                    continue
                label = charset_adapter(label)
                # We filter out samples which don't contain any supported characters
                if not label:
                    continue
                # Filter images that are too small.
                if self.min_image_dim > 0:
                    img_key = f'image-{index:09d}'.encode()
                    buf = io.BytesIO(txn.get(img_key))
                    w, h = Image.open(buf).size
                    if w < self.min_image_dim or h < self.min_image_dim:
                        continue
                self.labels.append(label)
                self.filtered_index_list.append(index)
        return len(self.labels)

    def __len__(self):
        return self.num_samples
    
    def next_sample(self):
        next_index = random.randint(0, len(self) - 1)
        if self.limit_size:
            next_index = self.index_list[next_index]
        return self.get(next_index)
    
    def get(self, index):
        index += 1  # lmdb starts with 1
        with self._create_env() as env, env.begin() as txn:
            label_key = f'label-{index:09d}'.encode()
            label = txn.get(label_key).decode()
            #
            label = edit_label(label, self.charset_adapter.charset) # edit self.charset in CharsetAdapter
            if len(label) > self.max_label_len:
                return self.next_sample()
            label = self.charset_adapter(label)
            # We filter out samples which don't contain any supported characters
            
            if not label:
                return self.next_sample()

            # Filter images that are too small.
            img_key = f'image-{index:09d}'.encode()
            buf = io.BytesIO(txn.get(img_key))
            img = Image.open(buf).convert('RGB')
            if self.min_image_dim > 0:
                w, h = img.size
    #             w, h = Image.open(buf).size
                if w < self.min_image_dim or h < self.min_image_dim:
                    return self.next_sample()
            img=np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.dstack([img,img,img])
            img=Image.fromarray(img)        
            w,h=img.size
            # vertical data handling
            if h/w>2.5:
                img=img.rotate(90, expand=True)
            if self.transform is not None:
                img = self.transform(img)
            
        return img, label

    def __getitem__(self, index):
        if self.is_training:
            if self.limit_size:
                index = self.index_list[index]
            return self.get(index)
        
        if self.unlabelled:
            label = index
        else:
            label = self.labels[index]
            index = self.filtered_index_list[index]

        img_key = f'image-{index:09d}'.encode()
        with self.env.begin() as txn:
            imgbuf = txn.get(img_key)
        buf = io.BytesIO(imgbuf)
        img = Image.open(buf).convert('RGB')
        img=np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.dstack([img,img,img])
        img=Image.fromarray(img)        
        w,h=img.size
        # vertical data handling
        if h/w>2.5:
            img=img.rotate(90, expand=True)

        if self.transform is not None:
            img = self.transform(img)

        return img, label

def edit_label(label, char_list) :
################################## New editions
#     match=re.findall(r'\[UNK[0-9]*\]',label)
#     if match:
#         for mat in match: label=label.replace(mat,'[UNK]')
    out_of_char = r'[^{}]'.format(re.escape(''.join(char_list)))
    label = label.replace('###','[UNK]')
    label = re.sub(out_of_char, "[UNK]", label)

    return label


