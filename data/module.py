from pathlib import PurePath
from typing import Optional, Callable, Sequence, Tuple, List

# import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms as T

from .dataset import build_tree_dataset, LmdbDataset
    
class SceneTextDataModule():
    def __init__(self, root_dir: str, train_dir: str, val_dir : str, test_dir : str, 
                img_size: Sequence[int],
                max_label_length: int, 
                charset_train: str, 
                charset_test: str, 
                batch_size: int, 
                num_workers: int, 
                augment: bool, 
                remove_whitespace: bool = True, 
                normalize_unicode: bool = True,
                min_image_dim: int = 0, 
                rotation: int = 0, 
                collate_fn: Optional[Callable] = None, 
                limit_size : bool = False , 
                size_of_limit : int = None,
                consistency_regularization: Optional[bool] = False, 
                exclude_folder: Optional[List] = [],
                data_weights: Optional[List] = []):
        super().__init__()
        self.root_dir = root_dir
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.img_size = tuple(img_size)
        self.max_label_length = max_label_length
        self.charset_train = charset_train
        self.charset_test = charset_test
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augment = augment
        self.remove_whitespace = remove_whitespace
        self.normalize_unicode = normalize_unicode
        self.min_image_dim = min_image_dim
        self.rotation = rotation
        self.collate_fn = collate_fn
        
        self.limit_size = limit_size
        self.size_of_limit = size_of_limit
        self.consistency_regularization = consistency_regularization
        self.exclude_folder = exclude_folder
        self.data_weights = data_weights

        self._train_dataset = None
        self._val_dataset = None

    @staticmethod
    def get_transform(img_size: Tuple[int], augment: bool = False, rotation: int = 0, consistency_regularization: Optional[bool] = False):

        if consistency_regularization :
            from .augmentation_pipelines import get_augmentation_pipeline
            augmentation_severity = 2 # 2 suits to document image
            pipeline = get_augmentation_pipeline(augmentation_severity)
            # pipeline.append(iaa.Resize(img_size))
            return pipeline.augment_image

        else :
            transforms = []
            if augment:
                from .augment import rand_augment_transform
                transforms.append(rand_augment_transform())
            if rotation:
                transforms.append(lambda img: img.rotate(rotation, expand=True))
            transforms.extend([
                T.Resize(img_size, T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(0.5, 0.5)
            ])
            return T.Compose(transforms)        
        
    @property
    def train_dataset(self):
        if self._train_dataset is None:
            transform = self.get_transform(self.img_size, self.augment, consistency_regularization = self.consistency_regularization)
            root = PurePath(self.train_dir)
            self._train_dataset = build_tree_dataset(root, self.charset_train, self.max_label_length,
                                                     self.min_image_dim, self.remove_whitespace, self.normalize_unicode,
                                                     transform=transform, limit_size = self.limit_size, size_of_limit = self.size_of_limit,
                                                     consistency_regularization = self.consistency_regularization,
                                                     img_size = self.img_size, twinreader_folders = self.exclude_folder, is_training = True
                                                     )
        return self._train_dataset

    @property
    def val_dataset(self):
        if self._val_dataset is None:
            transform = self.get_transform(self.img_size)
            root = PurePath(self.val_dir)
            self._val_dataset = build_tree_dataset(root, self.charset_test, self.max_label_length, 
                                                   self.min_image_dim, self.remove_whitespace, self.normalize_unicode,
                                                   img_size = self.img_size,
                                                   transform=transform, limit_size = False, is_training = False)
        return self._val_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, persistent_workers=self.num_workers > 0,
                          pin_memory=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, persistent_workers=self.num_workers > 0,
                          pin_memory=True, collate_fn=self.collate_fn)

    def test_dataloaders(self, subset):
        transform = self.get_transform(self.img_size, rotation=self.rotation)
        root = PurePath(self.test_dir)
        datasets = {s: LmdbDataset(str(root / s), self.charset_test, self.max_label_length,
                                   self.min_image_dim, self.remove_whitespace, self.normalize_unicode,
                                   transform=transform, is_training=False) for s in subset}
        return {k: DataLoader(v, batch_size=self.batch_size, num_workers=self.num_workers,
                              pin_memory=True, collate_fn=self.collate_fn)
                for k, v in datasets.items()}
