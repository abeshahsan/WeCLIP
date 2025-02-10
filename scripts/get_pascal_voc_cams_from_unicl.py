import sys
sys.path.append(".")

import argparse
import datetime
import logging
import os
import shutil
from transformers import CLIPTokenizer

from UniCL.config import get_config
from UniCL.model.model import build_unicl_model
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random
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
from clip.clip_text import class_names, new_class_names, BACKGROUND_CATEGORY
from torchvision.transforms import InterpolationMode
import warnings
warnings.filterwarnings("ignore")
from torch import multiprocessing

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_tokenizer():
    
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'

    return tokenizer

tokenizer = build_tokenizer()

def zeroshot_classifier(classnames, templates, model, device=DEVICE):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates] #format with class
            # texts = clip.tokenize(texts).cuda() #tokenize
            tokens = tokenizer(
                texts, padding='max_length', truncation=True, max_length=77, return_tensors='pt'
            )               
            tokens = {key:val.to(device) for key,val in tokens.items()}

            class_embeddings = model.encode_text(tokens) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights.t()

def generate_cams(args):
    device = DEVICE

    train_list = np.loadtxt(args.split_file, dtype=str)
    train_list = [x + '.jpg' for x in train_list]

    if not os.path.exists(args.cam_out_dir):
        os.makedirs(args.cam_out_dir)
    else:
        for filename in os.listdir(args.cam_out_dir):
            file_path = os.path.join(args.cam_out_dir, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

    model = build_unicl_model(args.model, device=device)
    
    bg_text_features = zeroshot_classifier(BACKGROUND_CATEGORY, ['a clean origami {}.'], model, device)
    fg_text_features = zeroshot_classifier(new_class_names, ['a clean origami {}.'], model, device)

    target_layers = [model.image_encoder.layers[-1].blocks[-1].norm2]

    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)

    dataset_list = split_dataset(train_list, n_splits=args.num_workers)
    if args.num_workers == 1:
        perform(0, dataset_list, args, model, bg_text_features, fg_text_features, cam)
    else:
        multiprocessing.spawn(perform, nprocs=args.num_workers,
                              args=(0, dataset_list, args, model, bg_text_features, fg_text_features, cam))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate CAMs for Pascal VOC using UNICL')
    parser.add_argument('--img_root', type=str, default='/data1/zbf_data/dataset/VOCdevkit_pure/VOC2012/JPEGImages')
    parser.add_argument('--split_file', type=str, default='./voc12/train.txt')
    parser.add_argument('--cam_out_dir', type=str, default='./final/ablation/voc_baseline')
    parser.add_argument('--model', type=str, default='/data1/zbf_data/Project2023/CLIP-ES-main/checkpoints/ViT-B-16.pt')
    parser.add_argument('--num_workers', type=int, default=1)
    args = parser.parse_args()

    generate_cams(args)