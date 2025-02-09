import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

# ALM model import
import model as alm_models

# SOLIDER model import
from models.solider.model_factory import build_backbone, build_classifier
from models.solider.base_block import FeatClassifier

# Utility functions
def fix_model_state_dict(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.module.'):
            name = name.replace('module.module.', '')
        elif name.startswith('module.'):
            name = name.replace('module.', '')
        new_state_dict[name] = v
    return new_state_dict


def default_loader(path):
    if os.path.exists(path):
        return Image.open(path).convert('RGB')
    return None

# Image preprocessing
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_test = transforms.Compose([
    transforms.Resize(size=(256, 128)),
    transforms.ToTensor(),
    normalize
])

# Model configurations
configs = {
    'alm_peta': {
        'model': alm_models.__dict__['inception_iccv'](pretrained=True, num_classes=35),
        'path': './pretrained/peta/peta_epoch_31.pth.tar',
        'gender_idx': 16
    },
    'alm_pa100k': {
        'model': alm_models.__dict__['inception_iccv'](pretrained=True, num_classes=26),
        'path': './pretrained/pa100k/pa100k_epoch_8.pth.tar',
        'gender_idx': 0
    },
    'alm_rap': {
        'model': alm_models.__dict__['inception_iccv'](pretrained=True, num_classes=51),
        'path': './pretrained/rap/rap_epoch_9.pth.tar',
        'gender_idx': 0
    },
    'solider_pa100k': {
        'model': None,  # Will be initialized later
        'path': './pretrained/pa100k/47.pth.tar',
        'gender_idx': 22
    },
    'solider_custom': {
        'model': None,  # Will be initialized later
        'path': './pretrained/pa100k/expo_age_40.pth.tar',
        'gender_idx': 22  # Adjust this based on your custom model
    }
}

# Initialize SOLIDER models
def init_solider_model(num_classes):
    with open('./model/solider/config/configs/peta_zs.yaml', 'r') as f:
        import yaml
        cfg = yaml.safe_load(f)
    backbone, c_output = build_backbone(cfg["BACKBONE"]["TYPE"], device='cuda')
    classifier = build_classifier(cfg["CLASSIFIER"]["NAME"])(
        nattr=num_classes,
        c_in=c_output,
        bn=cfg["CLASSIFIER"]["BN"],
        pool=cfg["CLASSIFIER"]["POOLING"],
        scale=cfg["CLASSIFIER"]["SCALE"]
    )
    return FeatClassifier(backbone, classifier)

configs['solider_pa100k']['model'] = init_solider_model(26)
configs['solider_custom']['model'] = init_solider_model(26)  # Adjust if your custom model has a different number of classes

# Load models
for config in configs.values():
    checkpoint = torch.load(config['path'], map_location='cuda')
    if 'solider' in config['path']:
        # print(fix_model_state_dict(checkpoint['state_dict']))

        # print(fix_model_state_dict(checkpoint['state_dict']))
        # exit()
        config['model'].load_state_dict(fix_model_state_dict(checkpoint['state_dict']))
    else:
        config['model'].load_state_dict(fix_model_state_dict(checkpoint['state_dict']))
    config['model'] = config['model'].cuda()
    config['model'].eval()

# Batch prediction functions
def predict_gender_alm(model, images, gender_idx, config_name):
    with torch.no_grad():
        outputs = model(images)
        if isinstance(outputs, (tuple, list)):
            outputs = torch.max(torch.max(torch.max(outputs[0], outputs[1]), outputs[2]), outputs[3])
        outputs = torch.sigmoid(outputs.data).cpu().numpy()
        outputs = np.where(outputs > 0.5, 1, 0)
    if 'peta' not in config_name:
        return ["Female" if output[gender_idx] == 1 else "Male" for output in outputs]
    else:
        return ["Male" if output[gender_idx] == 1 else "Female" for output in outputs]

def predict_gender_solider(model, images, gender_idx, config_name):
    with torch.no_grad():
        outputs = model(images)[0][0]
        outputs = torch.sigmoid(outputs.data).cpu().numpy()
        outputs = np.where(outputs > 0.5, 1, 0)
    if 'peta' not in config_name:
        return ["Female" if output[gender_idx] == 1 else "Male" for output in outputs]
    else:
        return ["Male" if output[gender_idx] == 1 else "Female" for output in outputs]


def process_image_list(image_list_file, output_file, batch_size=50):
    with open(image_list_file, 'r') as f:
        image_paths = [line.strip().split(',')[0] for line in f]
    print('image_paths', image_paths[:5])

    with open(output_file, 'w') as f:
        f.write("Image Path,ALM PETA,ALM PA100K,ALM RAP,SOLIDER PA100K,SOLIDER Custom\n")
        
        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            valid_indices = []
            
            for idx, path in enumerate(batch_paths):
                img = default_loader(os.path.join('/home/ubuntu/workspace_yamanishi/Camera-App-ML/src/DATA/person_image', 
                                                path))
                if img is not None:
                    batch_images.append(transform_test(img).unsqueeze(0))
                    valid_indices.append(idx)
            
            if not batch_images:
                # If all images in the batch are invalid, write -1 for all
                for path in batch_paths:
                    f.write(f"{path},-1,-1,-1,-1,-1\n")
                continue
            
            batch_tensor = torch.cat(batch_images).cuda()

            results = []
            for config_name, config in configs.items():
                if 'alm' in config_name:
                    genders = predict_gender_alm(config['model'], batch_tensor, config['gender_idx'], config_name)
                else:
                    genders = predict_gender_solider(config['model'], batch_tensor, config['gender_idx'], config_name)
                results.append(genders)

            for j, path in enumerate(batch_paths):
                if j in valid_indices:
                    idx = valid_indices.index(j)
                    f.write(f"{path},{results[0][idx]},{results[1][idx]},{results[2][idx]},{results[3][idx]},{results[4][idx]}\n")
                else:
                    f.write(f"{path},-1,-1,-1,-1,-1\n")

            # Clear GPU memory
            del batch_tensor
            torch.cuda.empty_cache()

# Run the prediction
image_list_file = './data/label/all_label_310k.txt'
output_file = './data/label/all_label_310k_gender.txt'
process_image_list(image_list_file, output_file)

print(f"Predictions completed. Results saved to {output_file}")