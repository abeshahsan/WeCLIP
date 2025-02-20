from genericpath import isfile
import logging
import os

import torch
from torch import nn

from timm.models.layers import trunc_normal_

from .image_encoder import build_image_encoder
from .text_encoder import build_text_encoder
from .text_encoder import build_tokenizer
import torch.nn.functional as F

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

        self.original_last_fts = None
        self.original_last_attn_weight = None

        self.logit_scale = nn.Parameter(torch.ones([]))

        trunc_normal_(self.text_projection, std=.02)
        trunc_normal_(self.image_projection, std=.02)

        """
            torch.Size([4, 784, 192]) torch.Size([4, 3136, 96])
            torch.Size([4, 784, 192]) torch.Size([4, 3136, 96])
            torch.Size([4, 196, 384]) torch.Size([4, 784, 192])
            torch.Size([4, 196, 384]) torch.Size([4, 784, 192])
            torch.Size([4, 49, 768]) torch.Size([4, 196, 384])
            torch.Size([4, 49, 768]) torch.Size([4, 196, 384])
            torch.Size([4, 49, 768]) torch.Size([4, 196, 384])
            torch.Size([4, 49, 768]) torch.Size([4, 196, 384])
            torch.Size([4, 49, 768]) torch.Size([4, 196, 384])
            torch.Size([4, 49, 768]) torch.Size([4, 196, 384])
            torch.Size([4, 49, 768]) torch.Size([4, 49, 768])
        """
        
        self.target_hw = (14, 14) 
        self.target_channels = 768
        


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
            

        pretrained_dict = torch.load(pretrained, map_location='cpu')
        logger.info(f'=> Loading pretrained model {pretrained}')
        pretrained_dict = self._convert_old_weights(pretrained_dict)
        model_dict = self.state_dict()
        pretrained_dict = {
            k: v for k, v in pretrained_dict['model'].items()
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
        self.image_encoder.load_state_dict(image_encoder_state_dict, strict=False)
        self.load_state_dict(need_init_state_dict, strict=False)

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
    
    def get_original_last_fts(self):
        return self.original_last_fts.permute(1, 0, 2), self.original_last_attn_weight
    
    def encode_image(self, image, norm=True):
        b = image.shape[0]
        x, attn = self.image_encoder.forward_features(image, image.shape[0], image.shape[1], require_all_fts=True)

        projected_fts_all = []
        projected_attn_weight_list = []

        self.original_last_fts = x[-1].clone()
        self.original_last_attn_weight = attn[-1].clone()

        for i, fts in enumerate(x):
            projected_fts_all.append(interpolate_and_project(fts, (14, 14), 768))
            projected_attn_weight_list.append(interpolate_and_project(attn[i], (14, 14), 196))
        
        del x, attn

        for i in range(len(projected_fts_all)):
            # x[i] = x[i] @ self.image_projection
            if norm:
                # x[i] = x[i] / x[i].norm(dim=-1, keepdim=True)
                projected_fts_all[i] = projected_fts_all[i] / projected_fts_all[i].norm(dim=-1, keepdim=True)


        return projected_fts_all, projected_attn_weight_list

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
    


    def forward_last_layer(self, image_features, text_features):
        image_features = image_features.permute(1, 0, 2)
        # image_features = interpolate_and_project(image_features, (7, 7), 768)
        logits_per_image, attn_weight = self.image_encoder.forward_last_layer(image_features, text_features)

        attn_weight = interpolate_and_project(attn_weight, (14, 14), 196)

        return logits_per_image, attn_weight


def build_unicl_model(config, **kwargs):
    model = UniCLModel(config)
    if config['MODEL']['PRETRAINED'] != '':
        pretrained_path = config['MODEL']['PRETRAINED']
 
        model.from_pretrained(pretrained= pretrained_path,  verbose=config['BACKBONE']['VERBOSE'])

    return model

def interpolate_and_project(x, target_hw, target_channels):
    """
    Resizes and optionally projects the tensor to match the target size and channels.
    x: Input tensor of shape (b, hw, c)
    """
    b, hw, c = x.shape
    h = w = int(hw ** 0.5)  # Assuming square spatial dimensions
    assert h * w == hw, "Input spatial dimensions must form a square"

    # Reshape to (b, c, h, w)
    x = x.permute(0, 2, 1).reshape(b, c, h, w)

    # Resize to target spatial dimensions
    x = F.interpolate(x, size=target_hw, mode='bilinear', align_corners=False)

    # Project channels if needed
    if c != target_channels:
        projection_layer = nn.Conv2d(c, target_channels, kernel_size=1).cuda()
        x = projection_layer(x)

    x = x.reshape(b, -1, target_channels)

    return x