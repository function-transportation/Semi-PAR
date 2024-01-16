import argparse
import os
import logging
import shutil
import time
import sys
import numpy as np
import math
from  tqdm import tqdm
import yaml

import torch
import torchvision.transforms as transforms
from models.solider.model_factory import build_backbone,build_classifier
from models.solider.base_block import FeatClassifier
from torch.utils.data import DataLoader
from utils.datasets import MultiLabelDatasetSSL
import pickle

def get_feature(model, dataset) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    '''
    get feature of labeled image and unlabeled image
    output:
        all_feature: [attr_num x N x feature_dim]
        pseudo_label: [N x attr_num]
        confidence: [N x attr_num]
    '''
    model.eval()
    #labeled_loader_tmp = DataLoader(labeled_dataset,batch_size=args.batch_size*3,num_workers=args.num_workers,shuffle=False,drop_last=False)
    unlabeled_loader_tmp = DataLoader(dataset, batch_size=500, num_workers=args.num_workers, shuffle=False, drop_last=False)
    pseudo_label, confidence = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(unlabeled_loader_tmp):
            #pred_3b, pred_4d, pred_5b, main_pred, pred_feature_3b, pred_feature_4d, pred_feature_5b, main_feat = model(imgs[0], return_feature=True)
            #N, d = main_feat.shape
            #main_feat = main_feat.unsqueeze(0).expand((attr_num, N, d))
            #all_features.append(torch.cat([pred_feature_3b.cpu(), pred_feature_4d.cpu(), pred_feature_5b.cpu(), main_feat.cpu()], axis=2))
            output = model(imgs)[0][0]
            #print(output.shape)
            pred = torch.sigmoid(output)
            confidence.append(pred.cpu())
            pseudo_label.append(torch.ge(pred, 0.5).cpu().to(int))
            
    return torch.cat(pseudo_label, axis=0).numpy(), torch.cat(confidence, axis=0).numpy()

parser = argparse.ArgumentParser(description='Pedestrian Attribute Framework')
parser.add_argument('--experiment', default='peta', type=str, required=False, help='(default=%(default)s)')
parser.add_argument('--label_file', type=str, default='./data/solider.txt')
parser.add_argument('--root', type=str, default='.')
parser.add_argument('--save_name', type=str, default='.')
parser.add_argument('--max_size', type=int, default=None)
parser.add_argument('--num-workers', type=int, default=4,
                    help='number of workers')

args = parser.parse_args()
attr_nums = {'peta':35, 'pa100k':26}

with open('./model/solider/config/configs/peta_zs.yaml', 'r') as filehandle:
    cfg = yaml.load(filehandle,Loader=yaml.Loader)
            
backbone, c_output = build_backbone(cfg["BACKBONE"]["TYPE"], device='cuda')
print('build backbone complete')
classifier = build_classifier(cfg["CLASSIFIER"]["NAME"])(
    nattr=attr_nums[args.experiment],
    c_in=c_output,
    bn=cfg["CLASSIFIER"]["BN"],
    pool=cfg["CLASSIFIER"]["POOLING"],
    scale =cfg["CLASSIFIER"]["SCALE"]
)
    
model = FeatClassifier(backbone, classifier)

if args.experiment=='pa100k':
    model.load_state_dict(torch.load('../CameraX/pretrained/pa100k/ckpt_max_solider3.pth')['state_dicts'])
    print('load pretrained')
model = torch.nn.DataParallel(model)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_val = transforms.Compose([
    transforms.Resize(size=(256, 128)),
    transforms.ToTensor(),
    normalize
])

dataset = MultiLabelDatasetSSL(
                    root=args.root,
                    label_file=args.label_file,
                    transform=transform_val,
                    label=False,
                    max_size=args.max_size,
                    dataset=args.experiment
                )

pseudo_label, confidence = get_feature(model,dataset)

with open(f'./data/pseudo_label_{args.save_name}_{args.experiment}.pkl', 'wb') as f:
    pickle.dump(pseudo_label, f)