# Global variables to store activations and gradients for GradCAM
import cv2
import numpy as np
from UniCL.model.model import UniCLModel
import torch.nn.functional as F
import os
import re

gradcam_activations = None
gradcam_gradients = None
feature_activations = []
attn_activations = []

def feature_forward_hook(module, input, output):
    feature_activations.append(output)

def attn_forward_hook(module, input, output):
    attn_activations.append(output)


def freeze_model(model, unfreeze_layer_names):
    for name, param in model.named_parameters():
        if all([layer_name not in name for layer_name in unfreeze_layer_names]):
            param.requires_grad = False
        else:
            param.requires_grad = True
            
def add_intermideate_fts_hook(model:UniCLModel):
    feature_activations.clear()
    attn_activations.clear()
    for layer in model.image_encoder.layers:
        for block in layer.blocks:
            block.register_forward_hook(feature_forward_hook)
            block.attn.attn_drop.register_forward_hook(attn_forward_hook)

def remove_intermideate_fts_hook(model:UniCLModel):
    for layer in model.image_encoder.layers:
        for block in layer.blocks:
            block._forward_hooks.clear()
            block.attn.attn_drop._forward_hooks.clear()


def attn_post_processing(model, batch_size, attn_weight_list, attn_weight_last):

    new_attn_weight_list = []

    for attn_weight in attn_weight_list:
        B = batch_size
        attn_weight = attn_weight.mean(dim = 1)# for heads
        grid_size = int((attn_weight.shape[0]//B) ** 0.5)

        window_size = model.layers[0].blocks[0].window_size

        attn_weight = attn_weight.view(B, grid_size, grid_size, window_size**2, window_size**2)

        attn_weight = attn_weight.permute(0, 1, 3, 2, 4).contiguous()  
        attn_weight = attn_weight.view(B, grid_size * (window_size ** 2), grid_size * (window_size ** 2))

        new_attn_weight_list.append(attn_weight)


    attn_weight_last = attn_weight_last.mean(dim = 1)
    attn_weight_last = F.interpolate(attn_weight_last.unsqueeze(1), size=(98, 98), mode='bilinear', align_corners=False).squeeze(1)

    return new_attn_weight_list, attn_weight_last

selected_image_names = ['2007_000032', '2007_002105', '2007_002227', '2007_002234', '2007_002273']

def save_some_cams(cam, annotation_path, cam_idx):
    annotation_path = annotation_path.replace('\\', '/')
    image_name = re.sub(r'.*/VOC2012/SegmentationClassAug/(.*).png', r'\1', annotation_path)
    jpeg_image_path = re.sub(r'(.*/VOC2012/).*', r'\1JPEGImages/' + image_name + '.jpg', annotation_path)
    original_image = cv2.imread(jpeg_image_path, cv2.IMREAD_COLOR)

    # with open(f'./imgs.txt', 'a') as f:
    #     f.write(image_name + '\n')

    if image_name in selected_image_names:
        # Ensure cam is in the correct format

        # with open(f'./imgs.txt', 'a') as f:
        #     f.write(str(cam))

        cam = (cam * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(cv2.resize(cam, (original_image.shape[1], original_image.shape[0])), cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(original_image, 0.5, heatmap, 0.5, 0)

        cv2.imwrite(f'./initial_cams/{image_name}_{cam_idx}.jpg', superimposed_img)

