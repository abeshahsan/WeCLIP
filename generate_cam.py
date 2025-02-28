import sys

import cv2
import numpy as np
sys.path.append(".")


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
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).to(device) #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights.t()


parser = argparse.ArgumentParser(description='Generate CAMs for Pascal VOC using CLIP')
parser.add_argument('--model', type=str, default='checkpoint/ViT-B-16.pt', help='Path to CLIP model')
args = parser.parse_args()


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

image_name = 'person.png'
# image_name = 'cat.jpg'
# image_name = 'aeroplane.jpg'
# image_name = 'bike.jpg'
image_name = 'emma-sunglass.jpg'

image = Image.open(image_name).convert('RGB')

input_tensor = transform(image).unsqueeze(0).to(DEVICE)

image_features, attn_list = model.encode_image(input_tensor, input_tensor.shape[-2], input_tensor.shape[-1])

final_image_features = model.forward_last_layer(image_features, torch.cat([fg_text_features], dim=0))[0]

score = final_image_features[0, 14]

# score = logits_per_image.sum()
model.zero_grad()
score.backward()

gradcam_activations = gradcam_activations.permute(1, 0, 2)[:, 1:, :] #L, B, C -> B, L, C
gradcam_gradients = gradcam_gradients.permute(1, 0, 2)[:, 1:, :] #L, B, C -> B, L, C

B, L, C = gradcam_activations.shape


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

# 
# for i, act in enumerate(feature_activations):
#     

# 
# for i, act in enumerate(attn_activations):
#     

# Remove the hooks to avoid interference with future iterations
forward_handle.remove()
backward_handle.remove()