from glob import glob
import argparse
import string
import sys
import os
from dataclasses import dataclass
from typing import List
from omegaconf import DictConfig, open_dict, OmegaConf
from tqdm import tqdm
import torch
import string
from data.module import SceneTextDataModule
from model.parseq import PARSeq

@dataclass
class Result:
    dataset: str
    num_samples: int
    accuracy: float
    ned: float
    confidence: float
    label_length: float


def print_results_table(results: List[Result], file=None):
    w = max(map(len, map(getattr, results, ['dataset'] * len(results))))
    w = max(w, len('Dataset'), len('Combined'))
    print('| {:<{w}} | # samples | Accuracy | 1 - NED | Confidence | Label Length |'.format('Dataset', w=w), file=file)
    print('|:{:-<{w}}:|----------:|---------:|--------:|-----------:|-------------:|'.format('----', w=w), file=file)
    c = Result('Combined', 0, 0, 0, 0, 0)
    for res in results:
        c.num_samples += res.num_samples
        c.accuracy += res.num_samples * res.accuracy
        c.ned += res.num_samples * res.ned
        c.confidence += res.num_samples * res.confidence
        c.label_length += res.num_samples * res.label_length
        print(f'| {res.dataset:<{w}} | {res.num_samples:>9} | {res.accuracy:>8.2f} | {res.ned:>7.2f} '
              f'| {res.confidence:>10.2f} | {res.label_length:>12.2f} |', file=file)
    c.accuracy /= c.num_samples
    c.ned /= c.num_samples
    c.confidence /= c.num_samples
    c.label_length /= c.num_samples
    print('|-{:-<{w}}-|-----------|----------|---------|------------|--------------|'.format('----', w=w), file=file)
    print(f'| {c.dataset:<{w}} | {c.num_samples:>9} | {c.accuracy:>8.2f} | {c.ned:>7.2f} '
          f'| {c.confidence:>10.2f} | {c.label_length:>12.2f} |', file=file)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='path of baseline or CR-based self-supervised config')
    parser.add_argument('--checkpoint', default= '', help="Model checkpoint (or 'pretrained=<model_id>')")
    #parser.add_argument('--korean_chars', type=str, default='./configs/korean_charset.txt', help='korean characters list text file')
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--cased', action='store_true', default=False, help='Cased comparison')
    parser.add_argument('--punctuation', action='store_true', default=False, help='Check punctuation')
    parser.add_argument('--new', action='store_true', default=False, help='Evaluate on new benchmark datasets')
    parser.add_argument('--rotation', type=int, default=0, help='Angle of rotation (counter clockwise) in degrees.')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    if not args.checkpoint:
        print('\nDEFAULT CHECKPOINT PATH SET TO CONFIG LOG_DIR.\n')
        if config.data.consistency_regularization:
            args.checkpoint = os.path.join(config.log_dir, 'CR_best_parseq_ckpt')
        else:
            args.checkpoint = os.path.join(config.log_dir, 'Baseline_best_parseq_ckpt')
    print('CHECKPOINT PATH: ', args.checkpoint)

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
    
    args.data_root = 'dataset'
    
    model = PARSeq(**config.model)
    model.load_state_dict(torch.load(args.checkpoint)['model'])
    model._device = args.device
    model = model.eval().to(args.device)

    print('Model parameters: ', config.model, sep='\n')
    
    train_dir = os.path.join(args.data_root, 'train')
    val_dir = os.path.join(args.data_root, 'valid')
    test_dir = os.path.join(args.data_root, 'valid')
    
    datamodule = SceneTextDataModule(root_dir = args.data_root, 
                                      train_dir = train_dir, 
                                      val_dir = val_dir,
                                      test_dir = test_dir,
                                      img_size = config.data.img_size,
                                      max_label_length = config.data.max_label_length,
                                      charset_train = config.data.charset_test, # hp.charset_train,
                                      charset_test = config.data.charset_test, # hp.charset_test,
                                      batch_size = args.batch_size,
                                      num_workers = args.num_workers,
                                      remove_whitespace = False, 
                                      normalize_unicode = False,
                                      augment = False,
                                      rotation = args.rotation
                                      )


    test_folders = glob(os.path.join(test_dir, '*'))
    test_set= sorted(set([t.split('/')[-1] for t in test_folders]))
    
    results = {}
    max_width = max(map(len, test_set))
    for name, dataloader in datamodule.test_dataloaders(test_set).items():
        total = 0
        correct = 0
        ned = 0
        confidence = 0
        label_length = 0
        for imgs, labels in tqdm(iter(dataloader), desc=f'{name:>{max_width}}'): # f'{name:>{max_width}}'
            res = model.test_step((imgs.to(args.device), labels), -1)['output']
            total += res.num_samples
            correct += res.correct
            ned += res.ned
            confidence += res.confidence
            label_length += res.label_length
        accuracy = 100 * correct / total
        mean_ned = 100 * (1 - ned / total)
        mean_conf = 100 * confidence / total
        mean_label_length = label_length / total
        results[name] = Result(name, total, accuracy, mean_ned, mean_conf, mean_label_length)
    
    result_groups = {
        t : [t] for t in test_set
    }

    if args.new:
        result_groups.update({'New': SceneTextDataModule.TEST_NEW})
    save_folder = os.path.join('/'.join(args.checkpoint.split('/')[:-1]), 'test_logs')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    with open(os.path.join(save_folder, os.path.basename(args.checkpoint)[:-4] + '.log.txt'), 'w') as f:
        for out in [f, sys.stdout]:
            for group, subset in result_groups.items():
                print(f'{group} set:', file=out)
                print_results_table([results[s] for s in subset], out)  
                print('\n', file=out)


if __name__ == '__main__':
    main()
