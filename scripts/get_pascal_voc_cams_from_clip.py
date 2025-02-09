import argparse
import datetime
import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random
import sys
sys.path.append(".")
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets import voc
from utils.losses import get_aff_loss
from utils import evaluate
from utils.AverageMeter import AverageMeter
from utils.camutils import cams_to_affinity_label
from utils.optimizer import PolyWarmupAdamW
from WeCLIP_model.model_attn_aff_voc import WeCLIP

from clip.generate_cams_voc12 import perform, zeroshot_classifier, split_dataset, reshape_transform
import clip
from pytorch_grad_cam import GradCAM
from clip_text import class_names, new_class_names, BACKGROUND_CATEGORY
from torchvision.transforms import InterpolationMode
import warnings
warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_cams(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    train_list = np.loadtxt(args.split_file, dtype=str)
    train_list = [x + '.jpg' for x in train_list]

    if not os.path.exists(args.cam_out_dir):
        os.makedirs(args.cam_out_dir)

    model, _ = clip.load(args.model, device=device)
    bg_text_features = zeroshot_classifier(BACKGROUND_CATEGORY, ['a clean origami {}.'], model)
    fg_text_features = zeroshot_classifier(new_class_names, ['a clean origami {}.'], model)

    target_layers = [model.visual.transformer.resblocks[-1].ln_1]

    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)

    dataset_list = split_dataset(train_list, n_splits=args.num_workers)
    if args.num_workers == 1:
        perform(0, dataset_list, args, model, bg_text_features, fg_text_features, cam)
    else:
        multiprocessing.spawn(perform, nprocs=args.num_workers,
                              args=(dataset_list, args, model, bg_text_features, fg_text_features, cam))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate CAMs for Pascal VOC using CLIP')
    parser.add_argument('--img_root', type=str, default='/data1/zbf_data/dataset/VOCdevkit_pure/VOC2012/JPEGImages')
    parser.add_argument('--split_file', type=str, default='./voc12/train.txt')
    parser.add_argument('--cam_out_dir', type=str, default='./final/ablation/voc_baseline')
    parser.add_argument('--model', type=str, default='/data1/zbf_data/Project2023/CLIP-ES-main/checkpoints/ViT-B-16.pt')
    parser.add_argument('--num_workers', type=int, default=1)
    args = parser.parse_args()

    generate_cams(args)
    print("Start training")