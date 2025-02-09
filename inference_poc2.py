import os
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
from models.solider.model_factory import build_backbone, build_classifier
from models.solider.base_block import FeatClassifier
from torch.utils.data import DataLoader
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
        if name.startswith('module.module.'):
            name = name.replace('module.module.', '')  # remove 'module.' of dataparallel
        elif name.startswith('module.'):
            name = name.replace('module.', '')  # remove 'module.' of dataparallel
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('./model/solider/config/configs/peta_zs.yaml', 'r') as filehandle:
        cfg = yaml.load(filehandle, Loader=yaml.Loader)

    backbone, c_output = build_backbone(cfg["BACKBONE"]["TYPE"], device=device)
    classifier = build_classifier(cfg["CLASSIFIER"]["NAME"])(
        nattr=6,  # 3クラス分類
        c_in=c_output,
        bn=cfg["CLASSIFIER"]["BN"],
        pool=cfg["CLASSIFIER"]["POOLING"],
        scale=cfg["CLASSIFIER"]["SCALE"]
    )
    model = FeatClassifier(backbone, classifier)

    model_path = './exp/poc2/fixmatch/poc2_290_all_clean/8.pth.tar'
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(fix_model_state_dict(checkpoint['state_dict']))
    model = torch.nn.DataParallel(model).to(device)
    model.eval()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_test = transforms.Compose([
        transforms.Resize(size=(256, 128)),
        transforms.ToTensor(),
        normalize
    ])

    image_list_file = "./data/label/poc2_gender_carry_age_test.txt"
    image_list = read_image_list(image_list_file)
    image_paths = [item[0] for item in image_list]
    original_labels = [item[1] for item in image_list]

    dataset = SimpleDataset(image_paths, transform_test)
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        num_workers=4,
        shuffle=False,
        collate_fn=lambda x: [item for item in x if item is not None]
    )

    results = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inferencing"):
            if not batch:
                continue
            batch = torch.stack(batch).to(device)
            output = model(batch)[0][0]
            
            # 最初の3クラスはバイナリ予測
            binary_probs = torch.sigmoid(output[:, :3]).cpu().numpy()
            binary_preds = (binary_probs > 0.5).astype(int)
            
            # 後の3クラスは分類予測
            class_probs = torch.softmax(output[:, 3:], dim=1).cpu().numpy()
            class_preds = np.eye(3)[np.argmax(class_probs, axis=1)]
            
            # 結果を結合
            combined_preds = np.hstack([binary_preds, class_preds])
            combined_probs = np.hstack([binary_probs, np.max(class_probs, axis=1, keepdims=True)])
            
            results.extend(zip(combined_preds, combined_probs))

    output_file = './data/label/poc2_test_pseudo.txt'
    with open(output_file, 'w') as f:
        f.write("image_path,label1,label2,label3,label4,label5,label6,confidence1,confidence2,confidence3,confidence4,confidence5,confidence6\n")
        for path, orig_label in zip(image_paths, original_labels):
            if os.path.exists(os.path.join(dataset.image_dir, path)):
                if results:
                    preds, probs = results.pop(0)
                    f.write(f"{path},{','.join(map(str, preds))},{','.join([f'{p:.4f}' for p in probs])}\n")
                else:
                    f.write(f"{path},0,0,0,0,0,0,-1,-1,-1,-1,-1,-1\n")
            else:
                f.write(f"{path},0,0,0,0,0,0,-1,-1,-1,-1,-1,-1\n")

    print(f"Inference completed. Results saved to {output_file}")

if __name__ == '__main__':
    main()