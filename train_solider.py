import argparse
import os
import logging
import shutil
import time
import sys
import numpy as np
import math
from  tqdm import tqdm

import torch
import torch.nn.parallel
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import torch.optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from datetime import datetime

from utils.datasets import Get_fixmatch_Dataset, Get_SF_Dataset, SimpleDataset
import pulp
from collections import OrderedDict
import pickle
import yaml
import wandb
from torch_ema import ExponentialMovingAverage
from models.solider.model_factory import build_backbone,build_classifier
from models.solider.base_block import FeatClassifier
from evaluate import real_data_evaluate, test, test_poc2, get_pseudo_label_and_conf
from loss import Weighted_BCELoss, MixedLoss
from sampler import BiasBalancer

parser = argparse.ArgumentParser(description='Pedestrian Attribute Framework')
parser.add_argument('--experiment', default='peta', type=str, required=False, help='(default=%(default)s)')
parser.add_argument('--approach', default='inception_iccv', type=str, required=False, help='(default=%(default)s)')
parser.add_argument('--epochs', default=60, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--batch_size', default=16, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, required=False, help='(default=%(default)f)')
parser.add_argument('--optimizer', default='adam', type=str, required=False, help='(default=%(default)s)')
parser.add_argument('--momentum', default=0.9, type=float, required=False, help='(default=%(default)f)')
parser.add_argument('--weight_decay', default=0.0005, type=float, required=False, help='(default=%(default)f)')
parser.add_argument('--start-epoch', default=0, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--print_freq', default=100, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--save_freq', default=10, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--resume', default='', type=str, required=False, help='(default=%(default)s)')
parser.add_argument('--decay_epoch', default=(20,40), type=eval, required=False, help='(default=%(default)d)')
parser.add_argument('--prefix', default='', type=str, required=False, help='(default=%(default)s)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', required=False, help='evaluate model on validation set')
parser.add_argument('--gpu-id', default='0', type=int,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--num-workers', type=int, default=4,
                    help='number of workers')
parser.add_argument('--num-labeled', type=int, default=4000,
                    help='number of labeled data')
parser.add_argument("--expand-labels", action="store_true",
                    help="expand labels to fit eval steps")
parser.add_argument('--arch', default='wideresnet', type=str,
                    choices=['wideresnet', 'resnext'],
                    help='dataset name')
parser.add_argument('--total-steps', default=2**20, type=int,
                    help='number of total steps to run')
parser.add_argument('--eval-step', default=1024, type=int,
                    help='number of eval steps to run')
parser.add_argument('--warmup', default=0, type=float,
                    help='warmup epochs (unlabeled data based)')
parser.add_argument('--wdecay', default=5e-4, type=float,
                    help='weight decay')
parser.add_argument('--nesterov', action='store_true', default=True,
                    help='use nesterov momentum')
parser.add_argument('--use-ema', action='store_true', default=True,
                    help='use EMA model')
parser.add_argument('--ema-decay', default=0.999, type=float,
                    help='EMA decay rate')
parser.add_argument('--mu', default=7, type=int,
                    help='coefficient of unlabeled batch size')
parser.add_argument('--lambda-u', default=1, type=float,
                    help='coefficient of unlabeled loss')
parser.add_argument('--T', default=1, type=float,
                    help='pseudo label temperature')
parser.add_argument('--threshold', default=0.95, type=float,
                    help='pseudo label threshold')
parser.add_argument('--out', default='result',
                    help='directory to output the result')
parser.add_argument('--seed', default=None, type=int,
                    help="random seed")
parser.add_argument("--amp", action="store_true",
                    help="use 16-bit (mixed) precision through NVIDIA apex AMP")
parser.add_argument("--opt_level", type=str, default="O1",
                    help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                    "See details at https://nvidia.github./apex/amp.html")
parser.add_argument("--local_rank", type=int, default=-1,
                    help="For distributed training: local_rank")
parser.add_argument('--no-progress', action='store_true',
                    help="don't use progress bar")
parser.add_argument('--method_name', type=str, default='fixmatch')
parser.add_argument('--train_label_file', type=str, default='./data/solider.txt')
parser.add_argument('--eval_label_file', type=str, default='./data/solider.txt')
parser.add_argument('--root', type=str, default='.')
parser.add_argument('--exp_id', type=str, default=None)
parser.add_argument('--lamb_localization', type=float, default=0)
parser.add_argument('--use_mask', action='store_true')
parser.add_argument('--max_size', type=int, default=None)
parser.add_argument('--lamb_reg', type=float, default=0)
parser.add_argument('--curriculum', action='store_true')
parser.add_argument('--ema', action='store_true')
parser.add_argument('--pseudo_thresh', type=float, default=0.9)
parser.add_argument('--curriculum_thresh', action='store_true')
parser.add_argument('--localization', action='store_true')
parser.add_argument('--train_fc_only', action='store_true')
parser.add_argument('--label_size', type=int, default=-1)
parser.add_argument('--age_loss', action='store_true')
parser.add_argument('--save_all', action='store_true')
# Seed
np.random.seed(1)
torch.manual_seed(1)
if torch.cuda.is_available(): torch.cuda.manual_seed(1)
else: print('[CUDA unavailable]'); sys.exit()
best_accu = 0
best_acc = 0
logger = logging.getLogger(__name__)

#####################################################################################################
def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)

def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

def calc_pseudo_thresh(epoch,):
    return max(0.99-0.003*epoch, 0.8)

def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

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

def localization_loss(x_maxs, x_mins, y_maxs, y_mins):
    '''
    input:
        x_maxs, x_mins, y_maxs, y_mins: [35xbatch]
    output:
        loss (scalar)
        ['Age16-30','Age31-45','Age46-60','AgeAbove61','Backpack',
        'CarryingOther','Casual lower','Casual upper','Formal lower','Formal upper',
        'Hat','Jacket','Jeans','Leather Shoes','Logo',
        'Long hair','Male','Messenger Bag','Muffler','No accessory',
        'No carrying','Plaid','PlasticBags','Sandals','Shoes',
        'Shorts','Short Sleeve','Skirt','Sneaker','Stripes',
        'Sunglasses','Trousers','Tshirt','UpperOther','V-Neck']
    '''
    anywhere_ind = [0, 1, 2, 3, 5, 16, 19, 20, 22] #age
    head_ind = [10, 15, 30]
    upper_ind = [4, 7, 9, 11, 14, 17, 18, 21, 26, 29,32, 33, 34]
    lower_ind = [6, 8, 12,25, 27, 28,31]
    foot_ind = [13,23,24, ]
    y_centers = (y_maxs+y_mins)//2
    head_loss = torch.sum(y_centers[head_ind, :]>256*0.3)/y_centers[head_ind, :].numel()
    upper_loss = torch.sum(y_centers[upper_ind, :]>256*0.6)/y_centers[upper_ind, :].numel()
    lower_loss = torch.sum(y_centers[lower_ind, :]<256*0.4)/y_centers[lower_ind, :].numel()
    foot_loss = torch.sum(y_centers[foot_ind, :]<256*0.7)/y_centers[foot_ind, :].numel()
    return lower_loss+foot_loss

def regularization_loss(output):
    '''
    input: [batchxattr]
    '''
    age_ind = [0, 1, 2, 3]
    cf_lower_ind = [6,8]
    cf_upper_ind = [7,9]
    lower_ind = [12, 32]
    shoes_ind = [13, 23, 24, 28]
    upper_ind = [32,34, 11]
    reg_loss_tot = 0
    for inds in [age_ind, cf_lower_ind, cf_upper_ind, lower_ind, shoes_ind, upper_ind]:
        reg_loss = torch.mean(torch.sum(output[:, inds], dim=1)**2-torch.sum(output[:, inds]**2, dim=1), dim=0)
        reg_loss_tot += reg_loss
        
    return reg_loss_tot

def regularization_loss_age(output):
    '''
    input: [batchxattr]
    '''
    age_ind = [1, 2, 3, 4]
    reg_loss_tot = 0
    for inds in [age_ind, ]:
        reg_loss = torch.mean(torch.sum(output[:, inds], dim=1)**2-torch.sum(output[:, inds]**2, dim=1), dim=0)
        reg_loss_tot += reg_loss
        
    return reg_loss_tot

def get_loss(logits, batch_size, criterion, targets_x, epoch, ):
    logits0 = de_interleave(logits[0], 2*args.mu+1)
    logits_x0 = logits0[:batch_size]
    logits_u_w0, logits_u_s0 = logits0[batch_size:].chunk(2)
    
    logits1 = de_interleave(logits[1], 2*args.mu+1)
    logits_x1 = logits1[:batch_size]
    logits_u_w1, logits_u_s1 = logits1[batch_size:].chunk(2)
    
    logits2 = de_interleave(logits[2], 2*args.mu+1)
    logits_x2 = logits2[:batch_size]
    logits_u_w2, logits_u_s2 = logits2[batch_size:].chunk(2)
    
    logits3 = de_interleave(logits[3], 2*args.mu+1)
    logits_x3 = logits3[:batch_size]
    logits_u_w3, logits_u_s3 = logits3[batch_size:].chunk(2)
    del logits
    
    #弱い拡張による画像とラベルのクロスエントロピー
    output = (logits_x0, logits_x1, logits_x2, logits_x3)
    if type(output) == type(()) or type(output) == type([]):
        loss_list = []
        # deep supervision
        for k in range(len(output)):
            out = output[k]
            loss_list.append(criterion.forward(torch.sigmoid(out), targets_x, epoch))
        Lx = sum(loss_list)
        # maximum voting
        output_x = torch.max(torch.max(torch.max(output[0],output[1]),output[2]),output[3])
    else:
        Lx = criterion.forward(torch.sigmoid(output), targets_x, epoch)
    return Lx

def load_partial_weight(model, cfg, attr_nums):
    backbone, c_output = build_backbone(cfg["BACKBONE"]["TYPE"], device=args.device)
    print('build backbone complete')
    classifier = build_classifier(cfg["CLASSIFIER"]["NAME"])(
        nattr=attr_nums['pa100k'],
        c_in=c_output,
        bn=cfg["CLASSIFIER"]["BN"],
        pool=cfg["CLASSIFIER"]["POOLING"],
        scale =cfg["CLASSIFIER"]["SCALE"]
    )
        
    model_pa100k = FeatClassifier(backbone, classifier)
    model_pa100k.load_state_dict(fix_model_state_dict(torch.load('./pretrained/pa100k/expo_age_40.pth.tar')['state_dict']))
    #index = [22, 21, 20, 19]
    index = [22, 21, 20, 20, 19]
    for name, param in model_pa100k.named_parameters():
        if 'classifier.logits.0' in name:
            if 'weight' in name:
                new_weight = param.data[index, :]
                #model.state_dict()[name][torch.tensor([0,1,4]), :] = new_weight
                model.state_dict()[name][:5, :] = new_weight
            elif 'bias' in name:
                new_bias = param.data[index]
                model.state_dict()[name][:5] = new_bias
        else:
            model.state_dict()[name].copy_(param.data)
    return model

def load_gender_and_age_weight(model, cfg, attr_nums, gender=True, age=True):
    backbone_pa100k, c_output = build_backbone(cfg["BACKBONE"]["TYPE"], device=args.device)
    print('build backbone complete')
    classifier_pa100k = build_classifier(cfg["CLASSIFIER"]["NAME"])(
        nattr=attr_nums['pa100k'],
        c_in=c_output,
        bn=cfg["CLASSIFIER"]["BN"],
        pool=cfg["CLASSIFIER"]["POOLING"],
        scale =cfg["CLASSIFIER"]["SCALE"]
    )

    model_pa100k = FeatClassifier(backbone_pa100k, classifier_pa100k)
    model_pa100k.load_state_dict(fix_model_state_dict(torch.load('./pretrained/pa100k/expo_age_40.pth.tar')['state_dict']))
    #model_pa100k.load_state_dict(fix_model_state_dict(torch.load('./pretrained/pa100k/47.pth.tar')['state_dict']))
    backbone_age, c_output = build_backbone(cfg["BACKBONE"]["TYPE"], device=args.device)
    print('build backbone complete')
    if age:
        classifier_age = build_classifier(cfg["CLASSIFIER"]["NAME"])(
            nattr=3,
            c_in=c_output,
            bn=cfg["CLASSIFIER"]["BN"],
            pool=cfg["CLASSIFIER"]["POOLING"],
            scale =cfg["CLASSIFIER"]["SCALE"]
        )
        model_age = FeatClassifier(backbone_age, classifier_age)
        model_age.load_state_dict(fix_model_state_dict(torch.load('./exp/age/fixmatch/train9200/18.pth.tar')['state_dict']))
    #index = [22, 21, 20, 19]
    #index = [22, 21, 20, 20, 19]
    if gender:
        index = [22]
        for name, param in model_pa100k.named_parameters():
            if 'classifier.logits.0' in name:
                if 'weight' in name:
                    new_weight = param.data[index, :]
                    model.state_dict()[name][:1, :] = new_weight
                elif 'bias' in name:
                    new_bias = param.data[index]
                    #model.state_dict()[name][:5] = new_bias
                    model.state_dict()[name][:1] = new_bias
            else:
                model.state_dict()[name].copy_(param.data)
            
    if age:
        index = [0, 1, 2]
        for name, param in model_age.named_parameters():
            if 'classifier.logits.0' in name:
                if 'weight' in name:
                    new_weight = param.data[index, :]
                    model.state_dict()[name][1:4, :] = new_weight
                elif 'bias' in name:
                    new_bias = param.data[index]
                    #model.state_dict()[name][:5] = new_bias
                    model.state_dict()[name][1:4] = new_bias
            else:
                pass
                #model.state_dict()[name].copy_(param.data)
        print('model', model)
    return model

def load_carry_weight(model, cfg, attr_nums):
    # Build PA-100K model and load pre-trained weights
    backbone_pa100k, c_output = build_backbone(cfg["BACKBONE"]["TYPE"], device=args.device)
    classifier_pa100k = build_classifier(cfg["CLASSIFIER"]["NAME"])(
        nattr=attr_nums['pa100k'],
        c_in=c_output,
        bn=cfg["CLASSIFIER"]["BN"],
        pool=cfg["CLASSIFIER"]["POOLING"],
        scale=cfg["CLASSIFIER"]["SCALE"]
    )
    model_pa100k = FeatClassifier(backbone_pa100k, classifier_pa100k)
    model_pa100k.load_state_dict(fix_model_state_dict(torch.load('./pretrained/pa100k/expo_age_40.pth.tar')['state_dict']))
    #model_pa100k.load_state_dict(fix_model_state_dict(torch.load('./pretrained/pa100k/47.pth.tar')['state_dict']))
    #model_pa100k.load_state_dict(fix_model_state_dict(torch.load('./pretrained/pa100k/ckpt_pretrain_solider.pth')['state_dicts']))


    # Copy weights from PA-100K model to carry model, except for classifier
    for name, param in model_pa100k.named_parameters():
        if 'classifier' not in name:
            model.state_dict()[name].copy_(param.data)

    print('Loaded PA-100K weights and initialized carry classifier')
    return model
        

def load_poc2_weight(model, cfg, attr_nums):
    # Build PA-100K model and load pre-trained weights
    backbone, c_output = build_backbone(cfg["BACKBONE"]["TYPE"], device=args.device)
    classifier_pa100k = build_classifier(cfg["CLASSIFIER"]["NAME"])(
        nattr=attr_nums['pa100k'],
        c_in=c_output,
        bn=cfg["CLASSIFIER"]["BN"],
        pool=cfg["CLASSIFIER"]["POOLING"],
        scale=cfg["CLASSIFIER"]["SCALE"]
    )
    model_pa100k = FeatClassifier(backbone, classifier_pa100k)
    model_pa100k.load_state_dict(fix_model_state_dict(torch.load('./pretrained/pa100k/expo_age_40.pth.tar')['state_dict']))

    classifier_carry = build_classifier(cfg["CLASSIFIER"]["NAME"])(
        nattr=attr_nums['carry'],
        c_in=c_output,
        bn=cfg["CLASSIFIER"]["BN"],
        pool=cfg["CLASSIFIER"]["POOLING"],
        scale=cfg["CLASSIFIER"]["SCALE"]
    )

    classifier_age = build_classifier(cfg["CLASSIFIER"]["NAME"])(
        nattr=attr_nums['age'],
        c_in=c_output,
        bn=cfg["CLASSIFIER"]["BN"],
        pool=cfg["CLASSIFIER"]["POOLING"],
        scale=cfg["CLASSIFIER"]["SCALE"]
    )
    model_carry = FeatClassifier(backbone, classifier_carry)
    model_carry.load_state_dict(fix_model_state_dict(torch.load('./exp/carry/fixmatch/expo_age_40_carry/8.pth.tar')['state_dict']))

    model_age = FeatClassifier(backbone, classifier_age)
    model_age.load_state_dict(fix_model_state_dict(torch.load('./exp/age/fixmatch/train9200/18.pth.tar')['state_dict']))
    # Copy weights from PA-100K model to carry model, except for classifier
    for name, param in model_pa100k.named_parameters():
        if 'classifier' not in name:
            model.state_dict()[name].copy_(param.data)

    index = [22]
    for name, param in model_pa100k.named_parameters():
        if 'classifier.logits.0' in name:
            if 'weight' in name:
                new_weight = param.data[index, :]
                model.state_dict()[name][:1, :] = new_weight
            elif 'bias' in name:
                new_bias = param.data[index]
                #model.state_dict()[name][:5] = new_bias
                model.state_dict()[name][:1] = new_bias

    index = [0, 1]
    for name, param in model_carry.named_parameters():
        if 'classifier.logits.0' in name:
            if 'weight' in name:
                new_weight = param.data[index, :]
                model.state_dict()[name][1:3, :] = new_weight
            elif 'bias' in name:
                new_bias = param.data[index]
                #model.state_dict()[name][:5] = new_bias
                model.state_dict()[name][1:3] = new_bias

    index = [0, 1, 2]
    for name, param in model_age.named_parameters():
        if 'classifier.logits.0' in name:
            if 'weight' in name:
                new_weight = param.data[index, :]
                model.state_dict()[name][3:, :] = new_weight
            elif 'bias' in name:
                new_bias = param.data[index]
                #model.state_dict()[name][:5] = new_bias
                model.state_dict()[name][3:] = new_bias


    print('model', model)
    print('Loaded PA-100K weights and initialized poc2 classifier')
    return model
    

def get_target_ratio(ratio):
    if ratio>0.9:
        return ratio-0.1
    elif ratio>0.5:
        return ratio-0.15 #(ratio//0.2+1) * 0.2
    elif ratio>0.1:
        return ratio+0.15 #ratio//0.2*0.2
    else:
        return ratio+0.1
        
        
def main():
    global args, best_accu
    args = parser.parse_args()
    if args.max_size is None:
        wandb.init(name=args.exp_id)
    print('=' * 100)
    print('Arguments = ')
    for arg in vars(args):
        print('\t' + arg + ':', getattr(args, arg))
    print('=' * 100)
    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1
        
    args.device = device
    # Data loading code
    attr_nums = {'peta':35, 'pa100k':26, 'gender_age': 4, 'carry': 2, 'gender': 1, 'age': 3, 'poc2': 6}
    attr_num = attr_nums[args.experiment]
    # (既存のセットアップコードは省略)

    # データセットの準備
    labeled_dataset, test_dataset, description = Get_SF_Dataset(
        dataset=args.experiment,
        train_label_txt=args.train_label_file,
        test_label_txt=args.eval_label_file,
        root=args.root,
        max_size=args.max_size,
        args=args
    )
    
    # データローダーの準備
    train_sampler = RandomSampler(labeled_dataset)
    train_loader = DataLoader(
        labeled_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # モデルの準備
    with open('./model/solider/config/configs/peta_zs.yaml', 'r') as filehandle:
        cfg = yaml.load(filehandle,Loader=yaml.Loader)
    backbone, c_output = build_backbone(cfg["BACKBONE"]["TYPE"], device=args.device)
    classifier = build_classifier(cfg["CLASSIFIER"]["NAME"])(
        nattr=attr_nums[args.experiment],
        c_in=c_output,
        bn=cfg["CLASSIFIER"]["BN"],
        pool=cfg["CLASSIFIER"]["POOLING"],
        scale=cfg["CLASSIFIER"]["SCALE"]
    )
    model = FeatClassifier(backbone, classifier)

    if args.experiment=='pa100k':
        model.load_state_dict(fix_model_state_dict(torch.load('./pretrained/pa100k/expo_age_40.pth.tar')['state_dict']))
        print('load pretrained')
        
    elif args.experiment == "gender_age":
        #model = load_partial_weight(model, cfg, attr_nums)
        #model.load_state_dict(fix_model_state_dict(torch.load('./exp/gender_age/fixmatch/2024_06_03_13:36:00/60.pth.tar')['state_dict']))
        model = load_gender_and_age_weight(model, cfg, attr_nums, gender=True, age=True)
    elif args.experiment == 'gender':
        #model.load_state_dict(fix_model_state_dict(torch.load('./pretrained/pa100k/expo_age_40.pth.tar')['state_dict']))
        model = load_gender_and_age_weight(model, cfg, attr_nums, gender=True, age=False)
        print('load gender')
    elif args.experiment == 'carry':
        model = load_carry_weight(model, cfg, attr_nums)

    elif args.experiment == 'poc2':
        model.load_state_dict(fix_model_state_dict(torch.load('./exp/poc2/fixmatch/joint/2.pth.tar')['state_dict']))
        # model = load_poc2_weight(model, cfg, attr_nums)
        print('load poc2 weight')
        # exit()

    model = torch.nn.DataParallel(model).cuda()
    if args.train_fc_only:
        for k,param in model.named_parameters():
            if 'classifier' not in k:
                param.requires_grad = False
        print('train fc only')


    # 損失関数とオプティマイザの準備
    if args.experiment == 'poc2':
        criterion = MixedLoss(args.experiment)
    else:
        criterion = Weighted_BCELoss(args.experiment)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=0.9, weight_decay=args.weight_decay)
    best_ma = 0
    exp_id = args.exp_id if args.exp_id is not None else datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
    # トレーニングループ
    directory = "./exp/" + args.experiment + '/' + args.method_name+ '/' + exp_id + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(directory+'result.txt', 'a') as f:
        f.write(args.train_label_file)
        f.write('\n')
        f.write(args.eval_label_file)
    test_func = test_poc2 if args.experiment == 'poc2' else test
    
    ma = test_func(test_loader, model, attr_num, description, best_ma, exp_id, args)
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.decay_epoch)
        if args.train_fc_only:
            for k,param in model.named_parameters():
                if 'classifier' not in k:
                    param.requires_grad = False
            #print('train fc only')
        # トレーニング
        train(train_loader, model, criterion, optimizer, epoch)
        
        # テスト
        ma = test_func(test_loader, model, attr_num, description, best_ma, exp_id, args)
        
        # ベストモデルの保存
        is_best = ma > best_ma
        best_ma = max(ma, best_ma)
        if is_best or args.save_all:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_ma': best_ma,
            }, epoch+1, args.prefix, args, exp_id)

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    model.train()
    end = time.time()
    
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        input = input.cuda()
        target = target.cuda()
        
        output = model(input)[0][0]
        #print('output', output, 'target', target)
        if args.experiment == 'poc2':
            loss = criterion.forward(output, target, epoch)
        else:
            loss = criterion.forward(torch.sigmoid(output), target, epoch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if args.experiment == 'poc2':
            # Accuracy calculation for poc2
            bce_output, ce_output = output[:, :3], output[:, 3:]
            bce_target, ce_target = target[:, :3], target[:, 3:]
            
            bce_pred = (torch.sigmoid(bce_output) > 0.5).float()
            ce_pred = ce_output.argmax(dim=1)
            
            bce_correct = (bce_pred == bce_target).float()
            ce_correct = (ce_pred == ce_target.argmax(dim=1)).float()
            
            bce_mask = (bce_target != -1).float()
            ce_mask = (ce_target.sum(dim=1) != -3).float()  # Assuming -1 for all three age classes means ignore
            
            accuracy = (bce_correct.sum() + ce_correct.sum()) / (bce_mask.sum() + ce_mask.sum() + 1e-6)
        else:
            # Original accuracy calculation
            output = torch.sigmoid(output) > 0.5
            accuracy = ((output == target) & (target != -1)).float().mean()

        losses.update(loss.item(), input.size(0))
        top1.update(accuracy.item(), input.size(0))
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accu {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))
            
def get_image_dirs(dir):
    '''
    対象のディレクトリ内の画像ファイルとpidを返す
    '''
    image_files = []
    image_dirs = []
    for curDir, dirs, files in os.walk(dir):
        if len(dirs)==0:
            image_files+=[os.path.join(curDir, f) for f in files]
            image_dirs+=[int(curDir.split('/')[-1]) for _ in range(len(files))]
    _, image_dirs = np.unique(np.array(image_dirs), return_inverse=True)
    return image_files, image_dirs


def save_checkpoint(state, epoch, prefix, args, exp_id ,filename='.pth.tar'):
    """Saves checkpoint to disk"""
    if not os.path.exists('./exp'):
        os.makedirs('./exp/')
    directory = "./exp/" + args.experiment + '/' + args.method_name+ '/' + exp_id + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    if prefix == '':
        filename = directory + str(epoch) + filename
    else:
        filename = directory + prefix + '_' + str(epoch) + filename
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch, decay_epoch):
    lr = args.lr
    for epc in decay_epoch:
        if epoch >= epc:
            lr = lr * 0.1
        else:
            break
    print()
    print('Learning Rate:', lr)
    print()
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target):
    batch_size = target.size(0)
    attr_num = target.size(1)

    output = torch.sigmoid(output).cpu().numpy()
    output = np.where(output > 0.5, 1, 0)
    pred = torch.from_numpy(output).long()
    target = target.cpu().long()
    correct = pred.eq(target)
    correct = correct.numpy()

    res = []
    for k in range(attr_num):
        label_size = len(target[target!=-1])
        res.append(1.0*sum(correct[:,k]) / label_size)
        
    return sum(res) / attr_num


if __name__ == '__main__':
    main()
