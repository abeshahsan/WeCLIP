import pathlib
import tempfile
from collections import OrderedDict
from typing import Tuple, Union
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from timm.models.layers import DropPath, trunc_normal_

from .image_encoder import build_image_encoder
from .text_encoder import build_text_encoder
from .text_encoder import build_tokenizer
from data.imagenet import IMAGENET_CLASSES, IMAGENET_DEFAULT_TEMPLATES

logger = logging.getLogger(__name__)


class UniCLModel(nn.Module):
    def __init__(self, config: dict,):
        super().__init__()

        self.conf_lang_encoder = config['MODEL']['TEXT_ENCODER']
        self.tokenizer = build_tokenizer(self.conf_lang_encoder)

        self.text_encoder = build_text_encoder(self.conf_lang_encoder, self.tokenizer, config['VERBOSE'])

        dim_projection = config['MODEL']['DIM_PROJECTION']
        if hasattr(self.text_encoder, 'dim_out'):
            dim_out = self.text_encoder.dim_out
        else:
            with torch.no_grad():
                dim_out = self.text_encoder(
                    torch.zeros(1,1).type(torch.LongTensor)
                )['last_hidden_state'].size(2)

        self.text_projection = nn.Parameter(torch.empty(dim_out, dim_projection))

        self.conf_image_encoder = config['MODEL']['IMAGE_ENCODER']
        self.image_encoder = build_image_encoder(self.conf_image_encoder)

        self.image_projection = nn.Parameter(
            torch.empty(self.image_encoder.dim_out, dim_projection)
        )

        self.logit_scale = nn.Parameter(torch.ones([]))

        trunc_normal_(self.text_projection, std=.02)
        trunc_normal_(self.image_projection, std=.02)

    def _convert_old_weights(self, model_dict):
        model_dict_updated = {}
        for k, v in model_dict.items():
            if k.startswith('visual.'):
                model_dict_updated['image_encoder.'+k[7:]] = v
            elif k.startswith('text.'):
                model_dict_updated['lang_encoder.'+k[5:]] = v
            elif k == 'vision_projection':
                model_dict_updated['image_projection'] = v
            elif k == 'text_projection':
                model_dict_updated['text_projection'] = v
            else:
                model_dict_updated[k] = v

        return model_dict_updated

    def from_pretrained(self, pretrained='', pretrained_layers=['*'], verbose=True):
        if not os.path.isfile(pretrained):
            logger.warning(f'=> Pretrained model ({pretrained}) is not a file, skip init weight')
            return

        # pretrained_dict = torch.load(pretrained, map_location='cpu')['model']

        # logger.info(f'=> Loading pretrained model {pretrained}')
        # pretrained_dict = self._convert_old_weights(pretrained_dict)
        # model_dict = self.state_dict()

        # with open('parameters/model_parameters.txt', 'w') as f:
        #     for key in model_dict.keys():
        #         f.write(f"{key}\n")
        
        # with open('parameters/in1k_yfcc14m.txt', 'w') as f:
        #     for key in pretrained_dict.keys():
        #         f.write(f"{key}\n")
        
        pretrained_dict = torch.load(pretrained, map_location='cpu')['model']
        logger.info(f'=> Loading pretrained model {pretrained}')
        pretrained_dict = self._convert_old_weights(pretrained_dict)
        model_dict = self.state_dict()
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items()
            if k in model_dict.keys()
        }
        need_init_state_dict = {}
        image_encoder_state_dict = {}
        for k, v in pretrained_dict.items():
            need_init = (
                k.split('.')[0] in pretrained_layers
                or pretrained_layers[0] == '*'
            )

            if need_init:
                if k.startswith('image_encoder.'):
                    image_encoder_state_dict[k] = v
                else:
                    if verbose:
                        logger.info(f'=> init {k} from {pretrained}')

                need_init_state_dict[k] = v
        self.load_state_dict(need_init_state_dict, strict=False)
        self.image_encoder.load_state_dict(image_encoder_state_dict, strict=False)



    @torch.jit.ignore
    def no_weight_decay(self):
        no_weight_decay = {'logit_scale'}
        if hasattr(self.text_encoder, 'no_weight_decay'):
            for k in self.text_encoder.no_weight_decay():
                no_weight_decay.add('lang_encoder.'+k)

        if hasattr(self.image_encoder, 'no_weight_decay'):
            for k in self.image_encoder.no_weight_decay():
                no_weight_decay.add('image_encoder.'+k)

        return no_weight_decay

    @property
    def dtype(self):
        return self.logit_scale.dtype

    def get_imnet_embeddings(self, classes):
        templates = IMAGENET_DEFAULT_TEMPLATES
        clss_embeddings = []
        for clss in classes:
            txts = [template.format(clss) for template in templates]
            
            tokens = self.tokenizer(
                txts, padding='max_length', truncation=True, max_length=77, return_tensors='pt'
            )                
            tokens = {key:val.cuda() for key,val in tokens.items()}

            clss_embedding = self.encode_text(tokens)
            clss_embedding = clss_embedding.mean(dim=0)
            clss_embedding /= clss_embedding.norm()
            clss_embeddings.append(clss_embedding)
        imnet_text_embeddings = torch.stack(clss_embeddings, dim=0)
        return imnet_text_embeddings

    def encode_image(self, image, norm=True):
        x = self.image_encoder.forward_features(image)
        x = x @ self.image_projection

        if norm:
            x = x / x.norm(dim=-1, keepdim=True)

        return 
    
    def forward_last_layer(self, image_features, text_features):
        features_image = self.image_encoder.layers[-1].blocks[-1](image_features)

        if self.image_encoder.layers[-1].downsample is not None:
            features_image = self.image_encoder.layers[-1].downsample(features_image)
        
        x = self.image_encoder.norm(features_image)  # B L C
        x = self.image_encoder.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        x = x @ self.image_projection

        features_image = x

        print(f'Features image: {features_image.size()}')
        print(f'Features text: {text_features.size()}')
        
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * features_image @ text_features.t()
        
        return logits_per_image

    def encode_text(self, text, norm=True):
        x = self.text_encoder(**text)
        x = x['last_hidden_state']

        if self.conf_lang_encoder['TOKENIZER'] == 'clip':
            x = x[torch.arange(x.size(0)), text['input_ids'].argmax(dim=-1)]
        else:
            x = x[:, 0]

        x = x @ self.text_projection

        if norm:
            x = x / x.norm(dim=-1, keepdim=True)

        return x

    def forward(self, image, text):
        features_image = self.encode_image(image)
        features_text = self.encode_text(text)

        # cosine similarity as logits
        T = self.logit_scale.exp()

        return features_image, features_text, T


def build_unicl_model(config, args=None, verbose=False, **kwargs):
    model = UniCLModel(config)

    model.from_pretrained(
        pretrained=args.unicl_model,
        pretrained_layers=['*'],
        verbose=verbose
    )

    return model
