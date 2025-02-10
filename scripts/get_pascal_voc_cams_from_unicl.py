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

from clip.generate_cams_voc12 import split_dataset, reshape_transform
import clip
from pytorch_grad_cam import GradCAM
from clip.clip_text import class_names, new_class_names, BACKGROUND_CATEGORY
from torchvision.transforms import InterpolationMode
import warnings
warnings.filterwarnings("ignore")
from torch import multiprocessing

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def reshape_transform(tensor, height=28, width=28):
    tensor = tensor.permute(1, 0, 2)
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

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

def perform(process_id, dataset_list, args, model, bg_text_features, fg_text_features, cam):
    n_gpus = torch.cuda.device_count()
    device_id = "cuda:{}".format(process_id % n_gpus)
    databin = dataset_list[process_id]
    model = model.to(device_id)
    bg_text_features = bg_text_features.to(device_id)
    fg_text_features = fg_text_features.to(device_id)
    for im_idx, im in enumerate(tqdm(databin)):
        img_path = os.path.join(args.img_root, im)
        xmlfile = img_path.replace('/JPEGImages', '/Annotations')
        xmlfile = xmlfile.replace('.jpg', '.xml')
        with open(xmlfile) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)  # etree包 读取xml文件
        data = parse_xml_to_dict(xml)["annotation"]

        ori_width = int(data['size']['width'])
        ori_height = int(data['size']['height'])

        label_list = []
        label_id_list = []
        for obj in data["object"]:
            obj["name"] = new_class_names[class_names.index(obj["name"])]
            if obj["name"] not in label_list:
                label_list.append(obj["name"])
                label_id_list.append(new_class_names.index(obj["name"]))

        if len(label_list) == 0:
            print("{} not have valid object".format(im))
            return

        ms_imgs = img_ms_and_flip(img_path, ori_height, ori_width, scales=[1.0])
        ms_imgs = [ms_imgs[0]]
        cam_all_scales = []
        highres_cam_all_scales = []
        refined_cam_all_scales = []
        for image in ms_imgs:
            image = image.unsqueeze(0)
            h, w = image.shape[-2], image.shape[-1]
            image = image.to(device_id)
            image_features, attn_weight_list = model.encode_image(image)

            cam_to_save = []
            highres_cam_to_save = []
            refined_cam_to_save = []
            keys = []

            bg_features_temp = bg_text_features.to(device_id)  # [bg_id_for_each_image[im_idx]].to(device_id)
            fg_features_temp = fg_text_features[label_id_list].to(device_id)
            text_features_temp = torch.cat([fg_features_temp, bg_features_temp], dim=0)
            input_tensor = [image_features, text_features_temp.to(device_id), h, w]

            for idx, label in enumerate(label_list):
                keys.append(new_class_names.index(label))
                targets = [ClipOutputTarget(label_list.index(label))]

                #torch.cuda.empty_cache()
                grayscale_cam, logits_per_image, attn_weight_last = cam(input_tensor=input_tensor,
                                                                                   targets=targets,
                                                                                   target_size=None)  # (ori_width, ori_height))

                grayscale_cam = grayscale_cam[0, :]

                grayscale_cam_highres = cv2.resize(grayscale_cam, (ori_width, ori_height))
                highres_cam_to_save.append(torch.tensor(grayscale_cam_highres))

                if idx == 0:
                    attn_weight_list.append(attn_weight_last)
                    attn_weight = [aw[:, 1:, 1:] for aw in attn_weight_list]  # (b, hxw, hxw)
                    attn_weight = torch.stack(attn_weight, dim=0)[-8:]
                    attn_weight = torch.mean(attn_weight, dim=0)
                    attn_weight = attn_weight[0].cpu().detach()
                attn_weight = attn_weight.float()

                box, cnt = scoremap2bbox(scoremap=grayscale_cam, threshold=0.4, multi_contour_eval=True)
                aff_mask = torch.zeros((grayscale_cam.shape[0],grayscale_cam.shape[1]))
                for i_ in range(cnt):
                    x0_, y0_, x1_, y1_ = box[i_]
                    aff_mask[y0_:y1_, x0_:x1_] = 1

                aff_mask = aff_mask.view(1,grayscale_cam.shape[0] * grayscale_cam.shape[1])
                aff_mat = attn_weight

                trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)
                trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)

                for _ in range(2):
                    trans_mat = trans_mat / torch.sum(trans_mat, dim=0, keepdim=True)
                    trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)
                trans_mat = (trans_mat + trans_mat.transpose(1, 0)) / 2

                for _ in range(1):
                    trans_mat = torch.matmul(trans_mat, trans_mat)

                trans_mat = trans_mat * aff_mask

                cam_to_refine = torch.FloatTensor(grayscale_cam)
                cam_to_refine = cam_to_refine.view(-1,1)

                # (n,n) * (n,1)->(n,1)
                cam_refined = torch.matmul(trans_mat, cam_to_refine).reshape(h //16, w // 16)
                cam_refined = cam_refined.cpu().numpy().astype(np.float32)
                cam_refined_highres = scale_cam_image([cam_refined], (ori_width, ori_height))[0]
                refined_cam_to_save.append(torch.tensor(cam_refined_highres))

            keys = torch.tensor(keys)
            #cam_all_scales.append(torch.stack(cam_to_save,dim=0))
            highres_cam_all_scales.append(torch.stack(highres_cam_to_save,dim=0))
            refined_cam_all_scales.append(torch.stack(refined_cam_to_save,dim=0))


        #cam_all_scales = cam_all_scales[0]
        highres_cam_all_scales = highres_cam_all_scales[0]
        refined_cam_all_scales = refined_cam_all_scales[0]

        np.save(os.path.join(args.cam_out_dir, im.replace('jpg', 'npy')),
                {"keys": keys.numpy(),
                # "strided_cam": cam_per_scales.cpu().numpy(),
                #"highres": highres_cam_all_scales.cpu().numpy().astype(np.float16),
                "attn_highres": refined_cam_all_scales.cpu().numpy().astype(np.float16),
                })
    return 0

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