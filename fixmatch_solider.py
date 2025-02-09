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

from utils.datasets import Get_fixmatch_Dataset, Get_Dataset, SimpleDataset
import pulp
from collections import OrderedDict
import pickle
import yaml
import wandb
from torch_ema import ExponentialMovingAverage
from models.solider.model_factory import build_backbone,build_classifier
from models.solider.base_block import FeatClassifier
from evaluate import real_data_evaluate, test, get_pseudo_label_and_conf
from loss import Weighted_BCELoss
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
parser.add_argument('--unlabel_label_file', type=str, default='./data/solider.txt')
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
parser.add_argument('--label_size', type=int, default=-1)
parser.add_argument('--unlabel_size', type=int, default=-1)
parser.add_argument('--age_loss', action='store_true')
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
        if name.startswith('module.'):
            name = name.replace('module.module.', '')  # remove 'module.' of dataparallel
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
    
    backbone_age, c_output = build_backbone(cfg["BACKBONE"]["TYPE"], device=args.device)
    print('build backbone complete')
    classifier_age = build_classifier(cfg["CLASSIFIER"]["NAME"])(
        nattr=3,
        c_in=c_output,
        bn=cfg["CLASSIFIER"]["BN"],
        pool=cfg["CLASSIFIER"]["POOLING"],
        scale =cfg["CLASSIFIER"]["SCALE"]
    )
    model_age = FeatClassifier(backbone_age, classifier_age)
    model_age.load_state_dict(fix_model_state_dict(torch.load('./exp/age/fixmatch/2024_06_10_02:35:03/5.pth.tar')['state_dict']))
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
                model.state_dict()[name].copy_(param.data)
        print('model', model)
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
    attr_nums = {'peta':35, 'pa100k':26, 'gender_age': 4, 'carry': 2, 'gender': 2}
    attr_num = attr_nums[args.experiment]
    
    peta_label_ratio = np.array([0.497, 0.329, 0.102, 0.062, 0.197, 0.199, 0.861, 0.853, 0.137,
       0.134, 0.102, 0.069, 0.306, 0.296, 0.04 , 0.238, 0.548, 0.296,
       0.084, 0.749, 0.276, 0.027, 0.076, 0.02 , 0.363, 0.035, 0.142,
       0.045, 0.216, 0.017, 0.029, 0.515, 0.084, 0.456, 0.012])
    
    labeled_dataset, unlabeled_dataset, test_dataset, description\
        = Get_fixmatch_Dataset(dataset=args.experiment,
                                train_label_txt=args.train_label_file,
                                train_unlabel_txt=args.unlabel_label_file,
                                test_label_txt=args.eval_label_file,
                                root=args.root,
                                max_size=args.max_size,
                                args=args)
    print(len(labeled_dataset), len(unlabeled_dataset), len(test_dataset))
    new_label_ratio = np.array(labeled_dataset.attributes).mean(axis=0)
    print('Old', peta_label_ratio)
    print('New', new_label_ratio)
    train_sampler = RandomSampler if True else DistributedSampler

    labeled_epoch, unlabeled_epoch = 0, 0

    with open('./model/solider/config/configs/peta_zs.yaml', 'r') as filehandle:
        cfg = yaml.load(filehandle,Loader=yaml.Loader)
                
    print('cfg', cfg)
    backbone, c_output = build_backbone(cfg["BACKBONE"]["TYPE"], device=args.device)
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
        model.load_state_dict(fix_model_state_dict(torch.load('./pretrained/pa100k/expo_age_40.pth.tar')['state_dict']))
        print('load pretrained')
        
    elif args.experiment == "gender_age":
        #model = load_partial_weight(model, cfg, attr_nums)
        #model.load_state_dict(fix_model_state_dict(torch.load('./exp/gender_age/fixmatch/2024_06_03_13:36:00/60.pth.tar')['state_dict']))
        model = load_gender_and_age_weight(model, cfg, attr_nums, gender=True, age=True)
    elif args.experiment == 'gender':
        model = load_gender_weight(model, cfg, attr_nums, gender=True, age=True)
        print('load pretrained')
    model = torch.nn.DataParallel(model)

    #print('model', model)
    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    print('')

    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    model = torch.nn.DataParallel(model).cuda()

    # define loss function
    criterion = Weighted_BCELoss(args.experiment)

    #no_decay = ['bias', 'bn']
    fc = ['finalfc']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in fc)], 'weight_decay': args.wdecay, 'lr': args.lr*0.1},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in fc)], 'weight_decay': 0.0, 'lr': args.lr}
    ]
    optimizer = torch.optim.SGD(grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)

    args.epochs = args.epochs
    print('total epoch', args.epochs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.total_steps)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=32,
        num_workers=args.num_workers)
    if args.evaluate:
        test(test_loader, model, attr_num, description)
        return
    #print('start test')
    best_ma = 0
    exp_id = args.exp_id if args.exp_id is not None else datetime.now().strftime('%Y_%m_%d_%H:%M:%S')
    test(test_loader, model, attr_num, description, best_ma, exp_id, args)
    for i,epoch in enumerate(range(args.start_epoch, args.start_epoch+args.epochs)):
        adjust_learning_rate(optimizer, i, args.decay_epoch)
        
        if args.experiment not in ['carry']:
            real_eval_attribute_ratio = real_data_evaluate(eval_dirs=['../../DATA/person_image_00006',
                                                                    '../../DATA/person_image_00033',
                                                                    './data/DATA/person_image_select/C3_0819PM_HD',
                                                                    './data/DATA/person_image_select/A101_0819AM_HD'],
                                                        model=model, attr_num=attr_num, 
                                                        dataset=args.experiment,
                                                        max_image_num=5000)
        #print('real_aval_attr', real_eval_attribute_ratio)
        target_attribute_ratio = {'Female': 0.53, 'AgeLess18': 0.04, 'AgeOver60': 0.04}
        target_attribute_ratio_tmp = {}
        target_attribute_ratio_unlabel_tmp = {}
        for attr in ['Female', 'AgeLess18', 'AgeOver60']:
            if attr not in description:continue
            attr_ind = description.index(attr)
            target_attribute_ratio_tmp[attr_ind] = target_attribute_ratio[attr] * 2 - real_eval_attribute_ratio[attr_ind]
            if attr=='Female':
                target_attribute_ratio_unlabel_tmp[attr_ind] = target_attribute_ratio[attr] * 2 - real_eval_attribute_ratio[attr_ind]
        #labeled_dataset.select_image(target_attribute_ratio_tmp)
        #unlabeled_dataset.select_image(target_attribute_ratio_unlabel_tmp)
        
        if args.curriculum:
            pseudo_label, confidence = get_pseudo_label_and_conf(model, args, labeled_dataset, unlabeled_dataset)
            unlabeled_dataset.pseudo_label = pseudo_label
            unlabeled_dataset.confidence = confidence
            label_sampler = BiasBalancer(labeled_dataset, epoch, args.batch_size, real_eval_attribute_ratio, unlabel=False)
            unlabel_sampler = BiasBalancer(unlabeled_dataset, epoch, args.batch_size*args.mu, real_eval_attribute_ratio, unlabel=True)
        else:
            label_sampler = train_sampler(labeled_dataset)
            unlabel_sampler = train_sampler(unlabeled_dataset)
        labeled_trainloader = DataLoader(
            labeled_dataset,
            sampler=label_sampler,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=True)
        print(labeled_trainloader.dataset)

        unlabeled_trainloader = DataLoader(
            unlabeled_dataset,
            sampler=unlabel_sampler,
            batch_size=args.batch_size*args.mu,
            num_workers=args.num_workers,
            drop_last=True)

        # train for one epoch
        if args.ema:
            ema = ExponentialMovingAverage(model.parameters(), decay=0.9)
        else:
            ema = None
        train(labeled_trainloader, unlabeled_trainloader, test_loader,
              model, optimizer, scheduler, epoch, criterion, ema=ema, age_loss=args.age_loss)

        ma=test(test_loader, model, attr_num, description, best_ma, exp_id, args, ema=ema)

        # remember best Accu and save checkpoint
        is_best = ma > best_ma
        best_ma = max(ma, best_ma)

        #if is_best:
        print('exp_id', exp_id)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_ma': best_ma,
        }, epoch+1, args.prefix, args, exp_id)
        

def train(labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer,  scheduler, epoch, criterion, ema=None, age_loss=False):
    global best_acc, labeled_epoch, unlabeled_epoch
    test_accs = []
    end = time.time()
    
    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_trainloader.sampler.set_epoch(labeled_epoch)
        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
        
    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    mask_probs = AverageMeter()
    top1 = AverageMeter()
    model.train()
    if not args.no_progress:
        p_bar = tqdm(range(args.eval_step),)
    if args.age_loss:
        age_cross_entropy = nn.CrossEntropyLoss()
    if args.curriculum_thresh:
        pseudo_thresh = calc_pseudo_thresh(epoch-args.start_epoch)
        #print('pseudo_thresh', pseudo_thresh)
    else:
        pseudo_thresh = args.pseudo_thresh
    for batch_idx in tqdm(range(args.eval_step)):
        try:
            inputs_x, targets_x = next(labeled_iter)
        except:
            if args.world_size > 1:
                labeled_epoch += 1
                labeled_trainloader.sampler.set_epoch(labeled_epoch)
            labeled_iter = iter(labeled_trainloader)

            inputs_x, targets_x = next(labeled_iter)

        try:
            (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)
        except:
            if args.world_size > 1:
                unlabeled_epoch += 1
                unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
            unlabeled_iter = iter(unlabeled_trainloader)
            (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)

        data_time.update(time.time() - end)
        batch_size = inputs_x.shape[0]
        inputs = interleave(
            torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2*args.mu+1).to(args.device)
    
        targets_x = targets_x.to(args.device)
        #print('target_x', (targets_x[:, 1:]==1).nonzero())
        #logits,  = model(inputs, )
        output = model(inputs)[0][0]
        output = de_interleave(output, 2*args.mu+1)
        output_x = output[:batch_size]
        #print('output_x', output_x)
        if args.age_loss:
            age_ind = (targets_x[:, 1:]==1).nonzero()
            batch_age_valid_ind = age_ind[:, 0]
            batch_age_ind = age_ind[:, 1]
            age_logit = output[:, 1:][batch_age_valid_ind]
            loss_age = age_cross_entropy(age_logit, batch_age_ind)
            reg_loss_age = regularization_loss_age(torch.sigmoid(output_x))

            # print(age_logit)
            # print(batch_age_valid_ind, batch_age_ind)
        output_u_w, output_u_s = output[batch_size:].chunk(2)
        #print('targets_x', targets_x)
        #弱い拡張による画像とラベルのクロスエントロピー
        label_mask = targets_x!=-1
        Lx = criterion.forward(torch.sigmoid(output_x), targets_x, epoch, label_mask=label_mask)
            
        #Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

        #pseudo_label = torch.softmax(logits_u_w.detach()/args.T, dim=-1)
        #max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        #mask = max_probs.ge(args.threshold).float()
        #強い拡張に得た画像と弱い拡張により得たpseudo-labelによるクロスエントロピー
        logit = output_u_w
        out = output_u_s
        mask = ((logit>=pseudo_thresh) | (logit<(1-pseudo_thresh))).to(int)
        # if args.age_loss:
        #     # ageYoung, ageOldはPA100KのAge18-60で初期化しているため
        #     if epoch<3:
        #         mask[:, [2, 3]] = 0
        #print('mask', mask)
        Lu = criterion.forward(torch.sigmoid(out), logit.ge(0.5).float(), epoch, mask=mask)
        if epoch < 20:
            loss = Lx
        else:
            loss = Lx + args.lambda_u * Lu
        #if args.age_loss:
        #    loss += 0.3 * loss_age# + 0.1 * reg_loss_age
        if args.experiment=='peta':
            reg_loss = regularization_loss(torch.sigmoid(output_x))
            loss+=args.lamb_reg * reg_loss

        loss.backward()
        bs = targets_x.size(0)
        accu = accuracy(output_x.data, targets_x)
        losses.update(loss.data, bs)
        top1.update(accu, bs)

        losses.update(loss.item())
        losses_x.update(Lx.item())
        losses_u.update(Lu.item())
        optimizer.step()
        if ema is not None:
            ema.update()
        scheduler.step()

        model.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accu {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, batch_idx, args.eval_step, batch_time=batch_time,
                      loss=losses, top1=top1))
            
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
