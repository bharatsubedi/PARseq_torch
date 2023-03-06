
import math
import argparse
from pathlib import Path
from tqdm import tqdm
import time
import os
import sys
import shutil
from omegaconf import OmegaConf
import numpy as np
import string
from typing import List

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import OneCycleLR
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.tensorboard import SummaryWriter 

from data.module import SceneTextDataModule
from model.parseq import PARSeq
from torch.utils.data import DataLoader
from glob import glob

if 'LOCAL_RANK' in os.environ:
    # Environment variables set by torch.distributed.launch or torchrun
    LOCAL_RANK = int(os.environ['LOCAL_RANK'])
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    WORLD_RANK = int(os.environ['RANK'])
    print('LOCAL_RANK, WORLD_SIZE, WORLD_RANK: ', LOCAL_RANK, WORLD_SIZE, WORLD_RANK)
else:
    sys.exit("Can't find the evironment variables for local rank")

def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def sync_across_gpus(t):
    gather_t_tensor = [torch.ones_like(t) for _ in range(WORLD_SIZE)]
    torch.distributed.all_gather(gather_t_tensor, t)
    return torch.stack(gather_t_tensor)

def prepare(dataset, rank, world_size, batch_size=32, pin_memory=False, num_workers=0):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, \
                                 shuffle=True, drop_last=False)

    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, \
                            num_workers=num_workers, drop_last=False, shuffle=False, sampler=sampler)
    
    return dataloader

class Balanced_Batch(SceneTextDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataloader_list = []
        self.dataloader_iter_list = []
        self.batch_size_list = []
        self.root_dir = self.train_dir
        
    def prepare(self, rank, world_size, pin_memory=False, num_workers=0):
        sub_folders = glob(os.path.join(self.train_dir, '*'))

        for folder, folder_weight in self.data_weights:
            self.train_dir = os.path.join(self.root_dir, folder)
            self._train_dataset = None
            print("FOLDER/WEIGHT/DIRECTORY: ", folder, folder_weight, self.train_dir)
            assert os.path.exists(self.train_dir), f'directory {self.train_dir} does not exist. check the folders and weights argument' 
            folder_batch_size = max(round(self.batch_size * folder_weight), 1)
            folder_dataset = self.train_dataset
            sampler = DistributedSampler(folder_dataset, num_replicas=world_size, rank=rank, \
                                         shuffle=True, drop_last=False)
            folder_dataloader = DataLoader(folder_dataset, batch_size=folder_batch_size, pin_memory=pin_memory, 
                                    num_workers=num_workers, drop_last=False, shuffle=False, sampler=sampler)
#             print('folder data len: ', len(folder_dataset), len(folder_dataloader))
            self.batch_size_list.append(folder_batch_size)
            self.dataloader_list.append(folder_dataloader)
            self.dataloader_iter_list.append(iter(folder_dataloader))
            
        self.batch_size = sum(self.batch_size_list)

    def get_batch(self):
        balanced_batch_images = []
        balanced_batch_texts = []

        for i, data_loader_iter in enumerate(self.dataloader_iter_list):
            try:
                image, text = data_loader_iter.next()
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except StopIteration:
                self.dataloader_iter_list[i] = iter(self.dataloader_list[i])
                image, text = self.dataloader_iter_list[i].next()
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except ValueError:
                pass

        balanced_batch_images = torch.cat(balanced_batch_images, 0)

        return balanced_batch_images, balanced_batch_texts

def main(args):
    
    config = OmegaConf.load(args.config)
    
    # updating korean charset
    chars = ""
    with open(
        os.path.join(config.data_loader.character.dict_dir, "charset.txt"), "r"
    ) as f:
        curr_char = f.read()
    curr_char = curr_char.strip()
    chars += curr_char

    chars = "".join(sorted(set(chars)))
    config.model.charset_train = chars
    config.model.charset_test = config.model.charset_train

    # Special handling for PARseq
    if config.model.get('perm_mirrored', False):
        assert config.model.perm_num % 2 == 0, 'perm_num should be even if perm_mirrored = True'
    if ~os.path.exists(config.log_dir):
        os.makedirs(config.log_dir, exist_ok=True)
    
    # config save
    OmegaConf.save(config, os.path.join(config.log_dir, "config.yaml"))
    
    # Tensorboard logs
    exp_log = SummaryWriter(os.path.join(config.log_dir, 'tf_runs'))
    
    dist.init_process_group("nccl", rank=WORLD_RANK, world_size=WORLD_SIZE)

    # Balanced data setting
    datamodule = Balanced_Batch(**config.data)
    datamodule.prepare(WORLD_RANK, WORLD_SIZE, num_workers=config.data.num_workers)
    val_dataset = datamodule.val_dataset
    config.data.batch_size = datamodule.batch_size
    config.model.batch_size = config.data.batch_size
    print('Updated Batch_Size: ', config.data.batch_size)
    
    val_loader = prepare(val_dataset, WORLD_RANK, WORLD_SIZE, batch_size=config.data.batch_size, \
                           num_workers=config.data.num_workers)

    print('val-loader length: ', len(val_loader))
    
    if LOCAL_RANK == 0:
        print('Baseline Model Training ... \n')
    model = PARSeq(**config.model)
    save_name = os.path.join(config.log_dir, 'Baseline')

    ckpt_savepath = f'{save_name}_parseq_ckpt.pth'
    best_ckpt_savepath = f'{save_name}_best_parseq_ckpt.pth'
    
    if not config.get('resume', None) and config.get('pretrained', None):
        pretrained_ckpt = torch.load(config.pretrained_ckpt, map_location='cuda:{}'.format(LOCAL_RANK))
        model.load_state_dict(pretrained_ckpt['model'])
    
    if config.get('resume', None):  
        ckpt = torch.load(config.resume, map_location='cuda:{}'.format(LOCAL_RANK))
        model.load_state_dict(ckpt['model'])
        
    model = model.to(LOCAL_RANK)
    
    device = torch.device('cuda:{}'.format(LOCAL_RANK))
    model._device = device
    model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, find_unused_parameters=True)

    estimated_stepping_batches = config.trainer.num_iters
    
    # setting optimizers and schedular
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print('Trainable params num : ', sum(params_num))
    
    config.model.lr = config.model.lr * math.sqrt(WORLD_SIZE) * config.model.batch_size / 256
    optimizer = torch.optim.AdamW(filtered_parameters, lr=config.model.lr, \
                                weight_decay=config.model.weight_decay)
    
    sched = OneCycleLR(optimizer, config.model.lr, estimated_stepping_batches, \
                       pct_start=config.model.warmup_pct,
                           cycle_momentum=False)
    iter_start = 0
    if config.get('resume', None):
        optimizer.load_state_dict(ckpt['optimizer'])
        iter_start = ckpt['iter']
    
    validate_iter = config.trainer.validate_iter
    num_iters = config.trainer.num_iters
    best_acc = 0
    
    with tqdm(range(iter_start, num_iters)) as t_iter:
        for iterr in t_iter:
            if iterr % validate_iter == 0:
                start_timer = time.time()
            img, label = datamodule.get_batch()
            img = img.to(device)

            loss = model.module.training_step(img, label)                
            log_loss = (torch.sum(sync_across_gpus(loss))/WORLD_SIZE)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sched.step()
            
            t_iter.set_postfix(Loss=loss.item())
            exp_log.add_scalars('train/loss', {f'process_{LOCAL_RANK}':loss.item()}, iterr)

            if LOCAL_RANK == 0:
                exp_log.add_scalars('train/loss', {f'average':log_loss.item()}, iterr)
                exp_log.add_scalars('train/lr', {'lr_change':optimizer.param_groups[0]['lr']}, iterr)
        
            if iterr > 0 and iterr % validate_iter == 0:
                torch.cuda.synchronize()
                outputs = []
                for batch_idx, val_data in tqdm(enumerate(val_loader)):
                    with torch.no_grad():
                        img, label = val_data
                        img = img.to(device)
                        outputs.append(model.module.validation_step((img, label), batch_idx))

                acc, ned, loss = model.module._aggregate_results(outputs)
                exp_log.add_scalars('val/loss', {f'process_{LOCAL_RANK}':loss}, iterr)
                exp_log.add_scalars('val/acc', {f'process_{LOCAL_RANK}':acc}, iterr)
                exp_log.add_scalars('val/ned', {f'process_{LOCAL_RANK}':ned}, iterr)

                acc_avg = (torch.sum(sync_across_gpus(torch.tensor(acc).to(f'cuda:{LOCAL_RANK}')))/WORLD_SIZE)
                ned_avg = (torch.sum(sync_across_gpus(torch.tensor(ned).to(f'cuda:{LOCAL_RANK}')))/WORLD_SIZE)
                loss_avg = (torch.sum(sync_across_gpus(loss.to(f'cuda:{LOCAL_RANK}')))/WORLD_SIZE)

                end_timer = time.time()
                elapsed = end_timer - start_timer
                time_avg = (torch.sum(sync_across_gpus(torch.tensor(elapsed).to(f'cuda:{LOCAL_RANK}')))/WORLD_SIZE)

                exp_log.add_scalars('train/time', {f'time_per_{validate_iter}iter_{LOCAL_RANK}':elapsed}, iterr)

                if LOCAL_RANK == 0:
                    print("Validation Accuracy: ", acc_avg.item()*100)
                    print("Validation NED: ", ned_avg.item()*100)
                    print("Validation Loss: ", loss_avg.item())

                    exp_log.add_scalars('val/loss', {f'avg_loss':loss_avg.item()}, iterr)
                    exp_log.add_scalars('val/acc', {f'avg_acc':acc_avg.item()}, iterr)
                    exp_log.add_scalars('val/ned', {f'avg_ned':ned_avg.item()}, iterr)
                    exp_log.add_scalars('train/time', {f'avg_time_per_{validate_iter}iter':time_avg.item()}, iterr)

                    torch.save({'model':model.module.state_dict(), 'optimizer':optimizer.state_dict(), 'cfg':config}, ckpt_savepath)
                    if best_acc < acc_avg:
                        best_acc = acc_avg
                        shutil.copyfile(ckpt_savepath, best_ckpt_savepath)

    exp_log.close()
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='path of baseline or CR-based self-supervised config')
    #parser.add_argument('--korean_chars', type=str, default='./configs/korean_charset.txt', help='korean characters list text file')
    args = parser.parse_args()
    
    main(args)
