import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
from PIL import Image
import torchvision
import matplotlib.pyplot as plt
from config import get_config
from datasets import voc
from model.model import UniCLModel, build_unicl_model
from model.text_encoder.build import build_tokenizer
import cv2
import numpy as np
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from utils import MY_CLASSES, get_text_embeddings

parser = argparse.ArgumentParser()

parser.add_argument('--batch-size', type=int, default=4, help="batch size for single GPU")




parser.add_argument('--output', default='output', type=str, metavar='PATH', help='root of output folder')
parser.add_argument('--cfg', type=str, default='configs/unicl_swin_tiny.yaml', help='config file path')
parser.add_argument('--unicl_model', type=str, default='checkpoint/yfcc14m.pth', help='unicl model path')
parser.add_argument('--data-path', type=str, help='path to dataset')
parser.add_argument('--name-list', type=str, default=None, help='path to name list file')

# torch.manual_seed(0)

def load_cls_dataset(cfg, args):
    val_dataset = voc.VOC12ClsDataset(
        root_dir=cfg.DATASET.DATA_DIR,
        name_list_dir=cfg.DATASET.NAME_LIST_DIR,
        split=cfg.DATASET.SPLIT,
        stage='val',
        ignore_index=cfg.DATASET.IGNORE_INDEX,
        num_classes=cfg.DATASET.NUM_CLASSES,
        resize_shape=cfg.DATASET.IMG_SIZE,
    )
    return val_dataset

def create_val_loader(val_dataset, cfg, args):
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.DATASET.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.DATASET.NUM_WORKERS,
        pin_memory=False,
        drop_last=False
    )
    return 

def add_intermideate_fts_hook(model:UniCLModel):
    for layer in model.image_encoder.layers:
        for block in layer.blocks:
            block.register_forward_hook(feature_forward_hook)
            block.attn.attn_drop.register_forward_hook(attn_forward_hook)

def remove_intermideate_fts_hook(model:UniCLModel):
    for layer in model.image_encoder.layers:
        for block in layer.blocks:
            block._forward_hooks.clear()

def add_gradcam_hook(model:UniCLModel):
    target_layer = model.image_encoder.layers[-1].blocks[-1]
    target_layer.register_forward_hook(gradcam_forward_hook)
    target_layer.register_backward_hook(gradcam_backward_hook)

def remove_gradcam_hook(model:UniCLModel):
    target_layer = model.image_encoder.layers[-1].blocks[-1]
    target_layer._forward_hooks.clear()
    target_layer._backward_hooks.clear()

# Global variables to store activations and gradients for GradCAM
gradcam_activations = None
gradcam_gradients = None
feature_activations = []
attn_activations = []

def process_image(model:UniCLModel, image_path, label_path,  text_embeddings, logit_scale, cfg, args):
    image = Image.open(image_path).convert('RGB')
    
    label = np.array(Image.open(label_path), dtype=np.uint8)
    label = np.unique(label - 1)
    label_list = label[label < 254].tolist()
    
    # Preprocess the image
    input_tensor = torchvision.transforms.Compose([
        torchvision.transforms.Resize((cfg.DATASET.IMG_SIZE[0], cfg.DATASET.IMG_SIZE[1])),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD
        )
    ])(image)

    input_tensor = input_tensor.unsqueeze(0).cuda()

    
    add_intermideate_fts_hook(model)
    model.encode_image(input_tensor, norm=False)
    remove_intermideate_fts_hook(model)

    add_gradcam_hook(model)

    feature_activation = feature_activations.pop()
    logits_per_image = model.forward_last_layer(feature_activations[-1], text_embeddings)

    # logits_per_image = logits_per_image.softmax(dim=-1)
    torch.set_printoptions(profile="full")
    print(f'Logits per image: {logits_per_image}')
    torch.set_printoptions(profile="default")

    label_list = [torch.argmax(logits_per_image, dim=-1).item()]

    for label in label_list:
        score = logits_per_image[0, label]
        # print(f'Predicted class: {MY_CLASSES[pred]} with score {score.item()}')
        model.zero_grad()
        score.backward(retain_graph=True)

        B, L, C = gradcam_activations.shape

        H = W = int(L ** 0.5)  # Assumes a square feature map
        activations_reshaped = gradcam_activations.view(B, H, W, C).permute(0, 3, 1, 2)
        gradients_reshaped = gradcam_gradients.view(B, H, W, C).permute(0, 3, 1, 2)
        weights = gradients_reshaped.mean(dim=(2, 3), keepdim=True)
        gradcam_map = torch.sum(weights * activations_reshaped, dim=1, keepdim=True)
        gradcam_map = F.relu(gradcam_map)
        gradcam_map_min = gradcam_map.min()
        gradcam_map_max = gradcam_map.max()
        gradcam_map = (gradcam_map - gradcam_map_min) / (gradcam_map_max - gradcam_map_min + 1e-8)
        
        gradcam_map_np = gradcam_map.cpu().detach().numpy()[0, 0].astype(np.float32)
        heatmap = cv2.resize(gradcam_map_np, (image.width, image.height))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR), 0.5, heatmap, 0.5, 0)
        
        if args.output is None:
            output_dir = 'output'
        else:
            output_dir = args.output
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

        image_name = os.path.basename(image_path).split('.')[0]
        
        cv2.imwrite(os.path.join(output_dir, f'{image_name}_{label}.png'), overlay)
        print(f'GradCAM for image {image_path} saved to {output_dir}/{image_name}_{label}.png')

        torch.cuda.empty_cache()

    for block_idx, feature_activation in enumerate(feature_activations):
        print(f'Block {block_idx} feature activation shape: {feature_activation.shape}')
    for block_idx, attn_activation in enumerate(attn_activations):
        print(f'Block {block_idx} attention activation shape: {attn_activation.shape}')
    
    remove_gradcam_hook(model)

def feature_forward_hook(module, input, output):
    feature_activations.append(output)

def attn_forward_hook(module, input, output):
    attn_activations.append(output)

def gradcam_forward_hook(module, input, output):
    global gradcam_activations
    gradcam_activations = output

def gradcam_backward_hook(module, grad_in, grad_out):
    global gradcam_gradients
    gradcam_gradients = grad_out[0]

########################################################################################
# Test UniCL Classification with GradCAM
########################################################################################

def test_unicl_classification(cfg, args):
    model = build_unicl_model(cfg, args)
    model = model.cuda()
    model.eval()
    
    # Build the tokenizer for text input
    conf_lang_encoder = cfg['MODEL']['TEXT_ENCODER']
    tokenizer = build_tokenizer(conf_lang_encoder)
    
    val_dataset = load_cls_dataset(cfg, args)
    val_loader = create_val_loader(val_dataset, cfg, args)
    
    # Precompute text embeddings (these are not used for GradCAM, so no gradient needed)
    with torch.no_grad():
        text_embeddings = get_text_embeddings(tokenizer, model, norm=False)

    logit_scale = model.logit_scale.exp()
    
    # Switch to evaluation mode (but allow gradients for the image branch)


    with open(args.name_list, 'r') as f:
        image_name_list = f.readlines()

    image_path_list = [os.path.join(f'{args.data_path}/JPEGImages', name.strip() + '.jpg') for name in image_name_list]
    class_label_list = [os.path.join(f'{args.data_path}/SegmentationClassAug', name.strip() + '.png') for name in image_name_list]



    # print(f'Class label for image {class_label_list[idx]}: {label}')

    for image_path, label_path in zip(image_path_list, class_label_list):
        process_image(model, image_path, label_path, text_embeddings, logit_scale, cfg, args)

if __name__ == '__main__':
    args = parser.parse_args()
    cfg = get_config(args)
    test_unicl_classification(cfg, args)
