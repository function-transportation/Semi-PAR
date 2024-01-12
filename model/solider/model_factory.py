from models.solider.registry import BACKBONE
from models.solider.registry import CLASSIFIER
from models.solider.registry import LOSSES
from models.solider.backbone.swin_transformer import swin_base_patch4_window7_224, swin_tiny_patch4_window7_224, swin_small_patch4_window7_224


model_dict = {
    'swin_t': 768,
    'swin_s': 768,
    'swin_b': 1024,

}
def build_backbone(key, multi_scale=False, device='cpu'):
    print('build backbone device', device)
    if key=='swin_s':
        model = swin_small_patch4_window7_224(device=device)
    elif key=='swin_t':
        model = swin_tiny_patch4_window7_224(device=device)
    elif key=='swin_b':
        model = swin_base_patch4_window7_224(device=device)
    #model = BACKBONE[key]()
    output_d = model_dict[key]

    return model, output_d


def build_classifier(key):

    return CLASSIFIER[key]


def build_loss(key):

    return LOSSES[key]

