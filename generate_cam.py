import sys
import os
import cv2
import numpy as np
import argparse
import torch
import torch.nn.functional as F
import clip
from PIL import Image
from clip.clip_text import new_class_names, BACKGROUND_CATEGORY
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import warnings

warnings.filterwarnings("ignore")

def zeroshot_classifier(classnames, templates, model, device='cuda'):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates]  # format with class
            texts = clip.tokenize(texts).to(device)  # tokenize
            class_embeddings = model.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights.t()

parser = argparse.ArgumentParser(description='Generate CAMs for Pascal VOC using CLIP')
parser.add_argument('--model', type=str, default='checkpoint/ViT-B-16.pt', help='Path to CLIP model')
parser.add_argument('--data-path', type=str, help='path to dataset')
parser.add_argument('--name-list', type=str, help='path to name list file')
parser.add_argument('--output', default='output', type=str, metavar='PATH', help='root of output folder')
args = parser.parse_args()


gradcam_activations = None
gradcam_gradients = None

def gradcam_forward_hook(module, input, output):
    global gradcam_activations
    gradcam_activations = output

def gradcam_backward_hook(module, grad_in, grad_out):
    global gradcam_gradients
    gradcam_gradients = grad_out[0]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model, _ = clip.load(args.model, device=DEVICE)
model.eval()
model.to(DEVICE)

bg_text_features = zeroshot_classifier(BACKGROUND_CATEGORY, ['a clean origami {}.'], model, DEVICE)
fg_text_features = zeroshot_classifier(new_class_names, ['a clean origami {}.'], model, DEVICE)

target_layer = model.visual.transformer.resblocks[-1].ln_1
forward_handle = target_layer.register_forward_hook(gradcam_forward_hook)
backward_handle = target_layer.register_backward_hook(gradcam_backward_hook)

transform = Compose([
    Resize((224, 224), interpolation=Image.BICUBIC),
    ToTensor(),
    Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711)),
])

def process_image(model, image_path, label_path, fg_text_features, bg_text_features, cfg, args):
    global gradcam_activations, gradcam_gradients

    image = Image.open(image_path).convert('RGB')
    label = np.array(Image.open(label_path), dtype=np.uint8)
    label = np.unique(label - 1)
    label_list = label[label < 254].tolist()

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    image_features, attn_list = model.encode_image(input_tensor, input_tensor.shape[-2], input_tensor.shape[-1])
    final_image_features = model.forward_last_layer(image_features, torch.cat([fg_text_features], dim=0))[0]

    label_list = [torch.argmax(final_image_features, dim=-1).item()]

    for label in label_list:
        score = final_image_features[0, label]
        model.zero_grad()
        score.backward(retain_graph=True)

        if gradcam_activations is None or gradcam_gradients is None:
            raise RuntimeError("GradCAM hooks did not capture activations or gradients")

        gradcam_activations = gradcam_activations.permute(1, 0, 2)[:, 1:, :]  # L, B, C -> B, L, C
        gradcam_gradients = gradcam_gradients.permute(1, 0, 2)[:, 1:, :]  # L, B, C -> B, L, C

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

if __name__ == '__main__':
    with open(args.name_list, 'r') as f:
        image_name_list = f.readlines()

    image_path_list = [os.path.join(f'{args.data_path}/JPEGImages', name.strip() + '.jpg') for name in image_name_list]
    class_label_list = [os.path.join(f'{args.data_path}/SegmentationClassAug', name.strip() + '.png') for name in image_name_list]

    for image_path, label_path in zip(image_path_list, class_label_list):
        process_image(model, image_path, label_path, fg_text_features, bg_text_features, None, args)
    
    
    forward_handle.remove()
    backward_handle.remove()
    for layer in model.visual.transformer.resblocks:
        layer._forward_hooks.clear()