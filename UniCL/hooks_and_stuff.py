# Global variables to store activations and gradients for GradCAM
from UniCL.model.model import UniCLModel
import torch.nn.functional as F


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

def freeze_model(model, unfreeze_layer_names):
    for name, param in model.named_parameters():
        if all([layer_name not in name for layer_name in unfreeze_layer_names]):
            param.requires_grad = False
        else:
            param.requires_grad = True
            
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

def attn_post_processing(model, attn_weight_list, attn_weight_last):

    new_attn_weight_list = []

    for attn_weight in attn_weight_list:
        B = attn_weight.shape[0]
        attn_weight = attn_weight.mean(dim = 1)# for heads
        grid_size = int((attn_weight.shape[0]//B) ** 0.5)

        window_size = model.layers[0].blocks[0].window_size

        attn_weight = attn_weight.view(B, grid_size, grid_size, window_size**2, window_size**2)

        attn_weight = attn_weight.permute(0, 1, 3, 2, 4).contiguous()  
        attn_weight = attn_weight.view(B, grid_size * (window_size ** 2), grid_size * (window_size ** 2))

        new_attn_weight_list.append(attn_weight)


    attn_weight_last = attn_weight_last.mean(dim = 1)
    attn_weight_last = F.interpolate(attn_weight_last.unsqueeze(1), size=(98, 98), mode='bilinear', align_corners=False).squeeze(1)

    return new_attn_weight_list

    