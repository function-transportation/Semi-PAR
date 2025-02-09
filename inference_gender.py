import os
import time
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
from models.solider.model_factory import build_backbone, build_classifier
from models.solider.base_block import FeatClassifier
from torch.backends import cudnn
from torch.utils.data import DataLoader, SequentialSampler
import yaml
from tqdm import tqdm
from collections import OrderedDict

def default_loader(path):
    return Image.open(path).convert('RGB')

def read_image_list(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [line.strip().split(',') for line in lines]

def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name.replace('module.module.', '') # remove 'module.' of dataparallel
        new_state_dict[name] = v
    return new_state_dict

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform
        self.image_dir = '/home/ubuntu/workspace_yamanishi/Camera-App-ML/src/DATA/person_image'

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_paths[idx])
        if not os.path.exists(img_path):
            return None
        img = default_loader(img_path)
        if self.transform:
            img = self.transform(img)
        return img

def main():
    # Set up the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load configuration
    with open('./model/solider/config/configs/peta_zs.yaml', 'r') as filehandle:
        cfg = yaml.load(filehandle, Loader=yaml.Loader)

    # Set up the model
    backbone, c_output = build_backbone(cfg["BACKBONE"]["TYPE"], device=device)
    classifier = build_classifier(cfg["CLASSIFIER"]["NAME"])(
        nattr=1,  # For carry bag detection
        c_in=c_output,
        bn=cfg["CLASSIFIER"]["BN"],
        pool=cfg["CLASSIFIER"]["POOLING"],
        scale=cfg["CLASSIFIER"]["SCALE"]
    )
    model = FeatClassifier(backbone, classifier)

    # Load the trained weights

    model_path = './exp/gender/fixmatch/2024_08_21_06:08:46/21.pth.tar'
    checkpoint = torch.load(model_path, map_location=device)
    model = torch.nn.DataParallel(model).to(device)
    model.load_state_dict(fix_model_state_dict(checkpoint['state_dict']))
    model.eval()

    # Set up data transformation
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_test = transforms.Compose([
        transforms.Resize(size=(256, 128)),
        transforms.ToTensor(),
        normalize
    ])

    # Read image list
    image_list_file = "./data/label/poc2_gender_carry_age_train.txt"
    image_list = read_image_list(image_list_file)
    image_paths = [item[0] for item in image_list]
    original_labels = [item[1] for item in image_list]

    # Create dataset and dataloader
    dataset = SimpleDataset(image_paths, transform_test)
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        num_workers=4,
        shuffle=False,
        collate_fn=lambda x: [item for item in x if item is not None]
    )

    # Perform inference
    results = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inferencing"):
            if not batch:  # 空のバッチをスキップ
                continue
            batch = torch.stack(batch).to(device)
            output = model(batch)[0][0]
            output = torch.sigmoid(output.data).cpu().numpy()
            labels = np.where(output > 0.5, 1, 0)
            confidences = output  # 生の確率値を信頼度として使用
            results.extend(np.column_stack((labels, confidences)).tolist())

    # 結果の保存
    #/home/ubuntu/workspace_yamanishi/ALM-pedestrian-attribute/data/label/poc2_gender_carry_age_train.txt
    output_file = "./data/label/gender_psudo_poc2.txt"
    with open(output_file, 'w') as f:
        f.write("image_path,label1,onfidence1\n")  # ヘッダーを更新
        for path, orig_label in zip(image_paths, original_labels):
            if os.path.exists(os.path.join(dataset.image_dir, path)):
                if results:
                    inferred_result = results.pop(0)
                    label1, conf1 = inferred_result
                else:
                    label1,conf1 = -1, -1
            else:
                label1, conf1 = -1, -1
            f.write(f"{path},{int(label1)},{conf1:.4f}\n")

    print(f"Inference completed. Results saved to {output_file}")

if __name__ == '__main__':
    main()