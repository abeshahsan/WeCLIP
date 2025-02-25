

import sys
sys.path.append(".")


import argparse
import torch
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
parser.add_argument('--model', type=str, default='WeCLIP_model/pretrained/ViT-B-16.pt', help='Path to CLIP model')
args = parser.parse_args()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   

model, _ = clip.load(args.model, device=DEVICE)
model.eval()
model.to(DEVICE)

bg_text_features = zeroshot_classifier(BACKGROUND_CATEGORY, ['a clean origami {}.'], model, DEVICE)
fg_text_features = zeroshot_classifier(new_class_names, ['a clean origami {}.'], model, DEVICE)


target_layers = [model.visual.transformer.resblocks[-1].ln_1]

transform = Compose([
        Resize((224, 224), interpolation=Image.BICUBIC),
        ToTensor(),
        Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073), 
            std=(0.26862954, 0.26130258, 0.27577711)),
    ])

image = Image.open('cat.jpg')
image = transform(image).unsqueeze(0).to(DEVICE)

image_features = model.encode_image(image, image.shape[0], image.shape[1])

print(image_features)