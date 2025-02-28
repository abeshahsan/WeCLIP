
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
from PIL import Image
import torchvision
from config import get_config
from datasets import voc
from model.model import build_unicl_model
from model.text_encoder.build import build_tokenizer  # Add this import
import cv2
import numpy as np
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD



from utils import MY_CLASSES, get_text_embeddings



parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default='configs/unicl_swin_tiny.yaml', help='config file path')
parser.add_argument('--unicl_model', type=str, default='checkpoint/yfcc14m.pth', help='unicl model path')
parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs.", default=None, nargs='+',)
parser.add_argument('--batch-size', type=int, default=4, help="batch size for single GPU")
parser.add_argument('--dataset', type=str, default='imagenet', help='dataset name')       
parser.add_argument('--data-path', type=str, help='path to dataset')
parser.add_argument('--name-list-path', type=str, help='path to name list')
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, full: cache all data, part: sharding the dataset into pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true', help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O0', choices=['O0', 'O1', 'O2'], help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--output', default='output', type=str, metavar='PATH', help='root of output folder')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')
parser.add_argument('--debug', action='store_true', help='Perform debug only')
parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel')

torch.manual_seed(0)

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
    return val_loader

# Global variables to store activations and gradients for GradCAM
gradcam_activations = None
gradcam_gradients = None
feature_activations = []
attn_activations = []

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
        text_embeddings = model.get_imnet_embeddings(MY_CLASSES)
    logit_scale = model.logit_scale.exp()
    
    # Switch to evaluation mode (but allow gradients for the image branch)
    

    image_name = 'bike.jpg'
    image = Image.open(image_name).convert('RGB')
    
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
    # cls_label = cls_label.cuda()

    target_layer = model.image_encoder.layers[-1].blocks[-1]
    forward_handle = target_layer.register_forward_hook(gradcam_forward_hook)
    backward_handle = target_layer.register_backward_hook(gradcam_backward_hook)
    
    for layer in model.image_encoder.layers:
        for block in layer.blocks:
            block.register_forward_hook(feature_forward_hook)
            block.attn.attn_drop.register_forward_hook(attn_forward_hook)
    
    # Forward pass (do not use torch.no_grad here so that gradients can be computed)
    image_features = model.encode_image(input_tensor, norm=False)

    # image_features_norm = image_features / image_features.norm(dim=1, keepdim=True)
    # text_embeddings_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
    # cosine similarity as logits
    logit_scale = model.logit_scale.exp()
    # logits_per_image = logit_scale * image_features @ text_embeddings.t()
    logits_per_image = logit_scale * image_features @ text_embeddings.t()

    # shape = [global_batch_size, global_batch_size]
    # logits_per_image = logits_per_image.softmax(dim=-1)

    print(logits_per_image)
    
    # print(logits_per_image)

    pred = torch.argmax(logits_per_image)
    score = logits_per_image[0, 13]
    print(f'Predicted class: {MY_CLASSES[pred]} with score {score.item()}')
    # score = logits_per_image.sum()
    model.zero_grad()
    score.backward()
    
    # At this point gradcam_activations and gradcam_gradients are set by our hooks.
    # We assume gradcam_activations has shape (B, L, C) where L = H * W.
    B, L, C = gradcam_activations.shape

    # print(gradcam_activations)
    # print(gradcam_gradients)

    H = W = int(L ** 0.5)  # Assumes a square feature map
    # Reshape activations and gradients to (B, C, H, W)
    activations_reshaped = gradcam_activations.view(B, H, W, C).permute(0, 3, 1, 2)
    gradients_reshaped = gradcam_gradients.view(B, H, W, C).permute(0, 3, 1, 2)
    # Compute weights: global average pooling on the gradients
    weights = gradients_reshaped.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
    # Compute GradCAM map: weighted combination of the activations
    gradcam_map = torch.sum(weights * activations_reshaped, dim=1, keepdim=True)  # (B, 1, H, W)
    gradcam_map = F.relu(gradcam_map)
    # Normalize the heatmap between 0 and 1
    gradcam_map_min = gradcam_map.min()
    gradcam_map_max = gradcam_map.max()
    gradcam_map = (gradcam_map - gradcam_map_min) / (gradcam_map_max - gradcam_map_min + 1e-8)
    
    # Convert GradCAM map to a numpy array
    gradcam_map_np = gradcam_map.cpu().detach().numpy()[0, 0].astype(np.float32)
    
    # Resize the GradCAM heatmap to match the original image dimensions
    heatmap = cv2.resize(gradcam_map_np, (image.width, image.height))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Overlay the heatmap on the original image (convert image to BGR for OpenCV)
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR), 0.5, heatmap, 0.5, 0)
    
    # Save the GradCAM overlay
    cv2.imwrite(f'output/gradcam_{image_name}', overlay)

    # print('Printing Intermediate Features')
    # for i, act in enumerate(feature_activations):
    #     print(f'Layer {i}: {act.shape}')
    
    # print('Printing Attention Weights')
    # for i, act in enumerate(attn_activations):
    #     print(f'Layer {i}: {act.shape}')
    
    # Remove the hooks to avoid interference with future iterations
    forward_handle.remove()
    backward_handle.remove()
    for layer in model.image_encoder.layers:
        for block in layer.blocks:
            block._forward_hooks.clear()


if __name__ == '__main__':
    args = parser.parse_args()
    cfg = get_config(args)
    test_unicl_classification(cfg, args)
