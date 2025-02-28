# --------------------------------------------------------
# Focal Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Jianwei Yang (jianwyan@microsoft.com)
# Based on Swin Transformer written by Zhe Liu
# --------------------------------------------------------

import os
import torch
import torch.distributed as dist
from timm.models.layers import trunc_normal_

import logging

from model.model import UniCLModel

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def load_checkpoint(config, model, optimizer, lr_scheduler, logger):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    elif os.path.exists(config.MODEL.RESUME):
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    else:
        logger.info(f"==============> Cannot find {config.MODEL.RESUME}....................")
        return None
    
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    max_accuracy = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        if 'amp' in checkpoint and config.AMP_OPT_LEVEL != "O0" and checkpoint['config'].AMP_OPT_LEVEL != "O0":
            amp.load_state_dict(checkpoint['amp'])
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy


def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}
    if config.AMP_OPT_LEVEL != "O0":
        save_state['amp'] = amp.state_dict()

    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt
class_map = {
    0: 'aeroplane',
    1: 'bicycle',
    2: 'bird',
    3: 'boat',
    4: 'bottle',
    5: 'bus',
    6: 'car',
    7: 'cat',
    8: 'chair',
    9: 'cow',
    10: 'diningtable',
    11: 'dog',
    12: 'horse',
    13: 'motorbike',
    14: 'person',
    15: 'pottedplant',
    16: 'sheep',
    17: 'sofa',
    18: 'train',
    19: 'tvmonitor'
}

MY_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 
    'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

TEMPLATES = [
    'a clean origami {}.',
    # '{}.',
    # 'a photo of a {}.',
    # 'a bad photo of a {}.',
    # 'a photo of many {}.',
    # 'a sculpture of a {}.',
    # 'a photo of the hard to see {}.',
    # 'a low resolution photo of the {}.',
    # 'a rendering of a {}.',
    # 'graffiti of a {}.',
    # 'a bad photo of the {}.',
    # 'a cropped photo of the {}.',
    # 'a tattoo of a {}.',
    # 'the embroidered {}.',
    # 'a photo of a hard to see {}.',
    # 'a bright photo of a {}.',
    # 'a photo of a clean {}.',
    # 'a photo of a dirty {}.',
    # 'a dark photo of the {}.',
    # 'a drawing of a {}.',
    # 'a photo of my {}.',
    # 'the plastic {}.',
    # 'a photo of the cool {}.',
    # 'a close-up photo of a {}.',
    # 'a black and white photo of the {}.',
    # 'a painting of the {}.',
    # 'a painting of a {}.',
    # 'a pixelated photo of the {}.',
    # 'a sculpture of the {}.',
    # 'a bright photo of the {}.',
    # 'a cropped photo of a {}.',
    # 'a plastic {}.',
    # 'a photo of the dirty {}.',
    # 'a jpeg corrupted photo of a {}.',
    # 'a blurry photo of the {}.',
    # 'a photo of the {}.',
    # 'a good photo of the {}.',
    # 'a rendering of the {}.',
    # 'a {} in a video game.',
    # 'a photo of one {}.',
    # 'a doodle of a {}.',
    # 'a close-up photo of the {}.',
    # 'a photo of a {}.',
    # 'the origami {}.',
    # 'the {} in a video game.',
    # 'a sketch of a {}.',
    # 'a doodle of the {}.',
    # 'a origami {}.',
    # 'a low resolution photo of a {}.',
    # 'the toy {}.',
    # 'a rendition of the {}.',
    # 'a photo of the clean {}.',
    # 'a photo of a large {}.',
    # 'a rendition of a {}.',
    # 'a photo of a nice {}.',
    # 'a photo of a weird {}.',
    # 'a blurry photo of a {}.',
    # 'a cartoon {}.',
    # 'art of a {}.',
    # 'a sketch of the {}.',
    # 'a embroidered {}.',
    # 'a pixelated photo of a {}.',
    # 'itap of the {}.',
    # 'a jpeg corrupted photo of the {}.',
    # 'a good photo of a {}.',
    # 'a plushie {}.',
    # 'a photo of the nice {}.',
    # 'a photo of the small {}.',
    # 'a photo of the weird {}.',
    # 'the cartoon {}.',
    # 'art of the {}.',
    # 'a drawing of the {}.',
    # 'a photo of the large {}.',
    # 'a black and white photo of a {}.',
    # 'the plushie {}.',
    # 'a dark photo of a {}.',
    # 'itap of a {}.',
    # 'graffiti of the {}.',
    # 'a toy {}.',
    # 'itap of my {}.',
    # 'a photo of a cool {}.',
    # 'a photo of a small {}.',
    # 'a tattoo of the {}.',
]

def tokenize_text(classname, tokenizer, device = 'cuda'):
    tokens = tokenizer(
            [template.format(classname) for template in TEMPLATES],
            max_length=77,         # Set the maximum length to match the model's expectation
            padding="max_length",  # Pad the sequence to the maximum length
            truncation=True,       # Truncate if longer than max_length
            return_tensors="pt"    # Return PyTorch tensors
        ).to(device)
    return tokens


def get_text_embeddings(tokenizer, model: UniCLModel, device = 'cuda', norm = True):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in MY_CLASSES:
            texts = tokenize_text(classname, tokenizer, device)
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights.t()

def setup_logger(filename='test.log'):
    logFormatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s: %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fHandler = logging.FileHandler(filename, mode='w')
    fHandler.setFormatter(logFormatter)
    logger.addHandler(fHandler)
    cHandler = logging.StreamHandler()
    cHandler.setFormatter(logFormatter)
    logger.addHandler(cHandler)
    
setup_logger()