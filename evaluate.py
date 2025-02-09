import numpy as np
import argparse
import os
import logging
import shutil
import time
import sys
import numpy as np
import math
from  tqdm import tqdm
from utils.datasets import Get_fixmatch_Dataset, Get_Dataset, SimpleDataset
from torch.utils.data import DataLoader
import torch
import wandb
from utils.utils import print_color_and_alert
import pandas as pd

def test(val_loader, model, attr_num, description, best_ma, exp_id, args, ema=None):
    model.eval()

    pos_cnt = []
    pos_tol = []
    neg_cnt = []
    neg_tol = []

    accu = 0.0
    prec = 0.0
    recall = 0.0
    tol = 0

    for it in range(attr_num):
        pos_cnt.append(0)
        pos_tol.append(0)
        neg_cnt.append(0)
        neg_tol.append(0)
    outputs = []
    for i, _ in tqdm(enumerate(val_loader)):
        input, target = _
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        if ema is None:
            output = model(input)[0][0]
        else:
            with ema.average_parameters():
                output = model(input)[0][0]
        bs = target.size(0)

        batch_size = target.size(0)
        tol = tol + batch_size
        output = torch.sigmoid(output.data).cpu().numpy()
        output = np.where(output > 0.5, 1, 0)
        outputs.append(output)
        target = target.cpu().numpy()

        for it in range(attr_num):
            for jt in range(batch_size):
                if target[jt][it] == 1:
                    pos_tol[it] = pos_tol[it] + 1
                    if output[jt][it] == 1:
                        pos_cnt[it] = pos_cnt[it] + 1
                if target[jt][it] == 0:
                    neg_tol[it] = neg_tol[it] + 1
                    if output[jt][it] == 0:
                        neg_cnt[it] = neg_cnt[it] + 1

        #if attr_num == 1:
        #   continue
        for jt in range(batch_size):
            tp = 0
            fn = 0
            fp = 0
            for it in range(attr_num):
                if output[jt][it] == 1 and target[jt][it] == 1:
                    tp = tp + 1
                elif output[jt][it] == 0 and target[jt][it] == 1:
                    fn = fn + 1
                elif output[jt][it] == 1 and target[jt][it] == 0:
                    fp = fp + 1
            if tp + fn + fp != 0:
                accu = accu +  1.0 * tp / (tp + fn + fp)
            if tp + fp != 0:
                prec = prec + 1.0 * tp / (tp + fp)
            if tp + fn != 0:
                recall = recall + 1.0 * tp / (tp + fn)

    print('=' * 100)
    print('\t     Attr              \tp_true/n_true\tp_tol/n_tol\tp_pred/n_pred\tcur_mA')
    mA = 0.0
    ma_dict = {}
    for it in range(attr_num):
        cur_mA = ((1.0*pos_cnt[it]/(pos_tol[it]+1e-6)) + (1.0*neg_cnt[it]/(neg_tol[it]+1e-6))) / 2.0
        mA = mA + cur_mA
        ma_dict[description[it]] = cur_mA
        print('\t#{:2}: {:18}\t{:4}\{:4}\t{:4}\{:4}\t{:4}\{:4}\t{:.5f}'.format(it,description[it],pos_cnt[it],neg_cnt[it],pos_tol[it],neg_tol[it],(pos_cnt[it]+neg_tol[it]-neg_cnt[it]),(neg_cnt[it]+pos_tol[it]-pos_cnt[it]),cur_mA))
    mA = mA / attr_num
    print('\t' + 'mA:        '+str(mA))

    outputs = np.concatenate(outputs)
    # print(outputs.shape)
    initial_labels, image_paths = [], []
    with open(args.eval_label_file, 'r') as f:
        attributes = f.readlines()
        for attr in attributes:
            att = attr.replace('¥n', '').split(',')[1:]
            att = [int(float(a)) for a in att]
            image_path = attr.replace('¥n', '').split(',')[0]
            initial_labels.append(att)
            image_paths.append(image_path)
    new_lines = []
    for i in range(outputs.shape[0]):
        tmp = []
        pseudo_label = outputs[i]
        tmp.append(image_paths[i])
        for label in pseudo_label:
            tmp.append(str(label))
        tmp[-1]+='\n'
        new_lines.append(','.join(tmp))
    with open('./data/label/eval_result.txt', 'w') as f:
        for line in new_lines:
            f.write(line)
        #print(new_lines)
    if True:#attr_num != 1:
        accu = accu / tol
        prec = prec / tol
        recall = recall / tol
        f1 = 2.0 * prec * recall / (prec + recall+1e-6)
        print('\t' + 'Accuracy:  '+str(accu))
        print('\t' + 'Precision: '+str(prec))
        print('\t' + 'Recall:    '+str(recall))
        print('\t' + 'F1_Score:  '+str(f1))
    print('=' * 100)
    ma_dict['mA'] = mA
    ma_dict['accuracy'] = accu
    ma_dict['precision'] = prec
    ma_dict['recall'] = recall
    #ma_dict['f1'] = f1
    if args.max_size is None:
        wandb.log(ma_dict)
    directory = "./exp/" + args.experiment + '/' + args.method_name+ '/' + exp_id + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(directory+'result.txt', 'a') as f:
        f.write('=' * 100+'\n')
        f.write('\t     Attr              \tp_true/n_true\tp_tol/n_tol\tp_pred/n_pred\tcur_mA'+'\n')
        for it in range(attr_num):
            cur_mA = ((1.0*pos_cnt[it]/(pos_tol[it]+1e-6)) + (1.0*neg_cnt[it]/(neg_tol[it]+1e-6))) / 2.0
            f.write('\t#{:2}: {:18}\t{:4}\{:4}\t{:4}\{:4}\t{:4}\{:4}\t{:.5f}'.format(it,description[it],pos_cnt[it],neg_cnt[it],pos_tol[it],neg_tol[it],(pos_cnt[it]+neg_tol[it]-neg_cnt[it]),(neg_cnt[it]+pos_tol[it]-pos_cnt[it]),cur_mA)+'\n')
        f.write('\t' + 'mA:        '+str(mA)+'\n')

        if attr_num != 1:
            f1 = 2.0 * prec * recall / (prec + recall+1e-5)
            f.write('\t' + 'Accuracy:  '+str(accu)+'\n')
            f.write('\t' + 'Precision: '+str(prec)+'\n')
            f.write('\t' + 'Recall:    '+str(recall)+'\n')
            #f.write('\t' + 'F1_Score:  '+str(f1)+'\n')
        f.write('=' * 100+'\n')
        f.write(str(vars(args)))
        
    return mA

def test_poc2(val_loader, model, attr_num, description, best_ma, exp_id, args, ema=None):
    model.eval()
    print(attr_num, description)
    pos_cnt = [0] * attr_num
    pos_tol = [0] * attr_num
    neg_cnt = [0] * attr_num
    neg_tol = [0] * attr_num

    accu = 0.0
    prec = 0.0
    recall = 0.0
    tol = 0

    outputs = []
    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(val_loader)):
            target = target.cuda(non_blocking=True)
            input = input.cuda(non_blocking=True)
            if ema is None:
                output = model(input)[0][0]
            else:
                with ema.average_parameters():
                    output = model(input)[0][0]

            batch_size = target.size(0)
            tol += batch_size
            output_prob = torch.sigmoid(output.data).cpu().numpy()
            target = target.cpu().numpy()

            # Gender and Carry (first 3 attributes)
            output_gc = np.where(output_prob[:, :3] > 0.5, 1, 0)
            # Age (last 3 attributes)
            output_age = output_prob[:, 3:].argmax(axis=1)
            output_age_onehot = np.eye(3)[output_age]

            output_combined = np.hstack((output_gc, output_age_onehot))
            outputs.append(output_combined)

            for jt in range(batch_size):
                for it in range(attr_num):
                    if target[jt][it] == -1:
                        continue
                    if it < 3:  # Gender and Carry
                        if target[jt][it] == 1:
                            pos_tol[it] += 1
                            if output_gc[jt][it] == 1:
                                pos_cnt[it] += 1
                        elif target[jt][it] == 0:
                            neg_tol[it] += 1
                            if output_gc[jt][it] == 0:
                                neg_cnt[it] += 1
                    else:  # Age
                        if target[jt][it] == 1:
                            pos_tol[it] += 1
                            if output_age_onehot[jt][it-3] == 1:
                                pos_cnt[it] += 1
                        else:
                            neg_tol[it] += 1
                            if output_age_onehot[jt][it-3] == 0:
                                neg_cnt[it] += 1

                tp = fp = fn = 0
                for it in range(attr_num):
                    if target[jt][it] == -1:
                        continue
                    if it < 3:  # Gender and Carry
                        if output_gc[jt][it] == 1 and target[jt][it] == 1:
                            tp += 1
                        elif output_gc[jt][it] == 0 and target[jt][it] == 1:
                            fn += 1
                        elif output_gc[jt][it] == 1 and target[jt][it] == 0:
                            fp += 1
                    else:  # Age
                        if output_age_onehot[jt][it-3] == 1 and target[jt][it] == 1:
                            tp += 1
                        elif output_age_onehot[jt][it-3] == 0 and target[jt][it] == 1:
                            fn += 1
                        elif output_age_onehot[jt][it-3] == 1 and target[jt][it] == 0:
                            fp += 1
                
                if tp + fn + fp != 0:
                    accu += 1.0 * tp / (tp + fn + fp)
                if tp + fp != 0:
                    prec += 1.0 * tp / (tp + fp)
                if tp + fn != 0:
                    recall += 1.0 * tp / (tp + fn)

    print('=' * 100)
    print('\t     Attr              \tp_true/n_true\tp_tol/n_tol\tp_pred/n_pred\tcur_mA')
    mA = 0.0
    ma_dict = {}
    valid_attr_count = 0
    for it in range(attr_num):
        if pos_tol[it] + neg_tol[it] > 0:
            cur_mA = ((1.0*pos_cnt[it]/(pos_tol[it]+1e-6)) + (1.0*neg_cnt[it]/(neg_tol[it]+1e-6))) / 2.0
            mA += cur_mA
            ma_dict[description[it]] = cur_mA
            valid_attr_count += 1
            print('\t#{:2}: {:18}\t{:4}\{:4}\t{:4}\{:4}\t{:4}\{:4}\t{:.5f}'.format(it,description[it],pos_cnt[it],neg_cnt[it],pos_tol[it],neg_tol[it],(pos_cnt[it]+neg_tol[it]-neg_cnt[it]),(neg_cnt[it]+pos_tol[it]-pos_cnt[it]),cur_mA))
    mA /= valid_attr_count
    print('\t' + 'mA:        '+str(mA))

    outputs = np.concatenate(outputs)
    print(outputs.shape)
    initial_labels, image_paths = [], []
    with open(args.eval_label_file, 'r') as f:
        attributes = f.readlines()
        for attr in attributes:
            parts = attr.strip().split(',')
            image_path = parts[0]
            att = [int(a) for a in parts[1:]]
            initial_labels.append(att)
            image_paths.append(image_path)

    new_lines = []
    for i in range(outputs.shape[0]):
        tmp = [image_paths[i]]
        pseudo_label = outputs[i]
        for j, label in enumerate(pseudo_label):
            tmp.append(str(int(label)))
        tmp[-1] += '\n'
        new_lines.append(','.join(tmp))
    with open('./data/label/eval_result.txt', 'w') as f:
        for line in new_lines:
            f.write(line)

    accu = accu / tol
    prec = prec / tol
    recall = recall / tol
    f1 = 2.0 * prec * recall / (prec + recall + 1e-6)
    print('\t' + 'Accuracy:  '+str(accu))
    print('\t' + 'Precision: '+str(prec))
    print('\t' + 'Recall:    '+str(recall))
    print('\t' + 'F1_Score:  '+str(f1))
    print('=' * 100)
    ma_dict['mA'] = mA
    ma_dict['accuracy'] = accu
    ma_dict['precision'] = prec
    ma_dict['recall'] = recall
    if args.max_size is None:
        wandb.log(ma_dict)
    directory = "./exp/" + args.experiment + '/' + args.method_name+ '/' + exp_id + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(directory+'result.txt', 'a') as f:
        f.write('=' * 100+'\n')
        f.write('\t     Attr              \tp_true/n_true\tp_tol/n_tol\tp_pred/n_pred\tcur_mA'+'\n')
        for it in range(attr_num):
            if pos_tol[it] + neg_tol[it] > 0:
                cur_mA = ((1.0*pos_cnt[it]/(pos_tol[it]+1e-6)) + (1.0*neg_cnt[it]/(neg_tol[it]+1e-6))) / 2.0
                f.write('\t#{:2}: {:18}\t{:4}\{:4}\t{:4}\{:4}\t{:4}\{:4}\t{:.5f}'.format(it,description[it],pos_cnt[it],neg_cnt[it],pos_tol[it],neg_tol[it],(pos_cnt[it]+neg_tol[it]-neg_cnt[it]),(neg_cnt[it]+pos_tol[it]-pos_cnt[it]),cur_mA)+'\n')
        f.write('\t' + 'mA:        '+str(mA)+'\n')
        f.write('\t' + 'Accuracy:  '+str(accu)+'\n')
        f.write('\t' + 'Precision: '+str(prec)+'\n')
        f.write('\t' + 'Recall:    '+str(recall)+'\n')
        f.write('\t' + 'F1_Score:  '+str(f1)+'\n')
        f.write('=' * 100+'\n')
        f.write(str(vars(args)))
        
    return mA

def get_image_dirs_C6(dir, max_image_num=5000, ):
    '''
    対象のディレクトリ内の画像ファイルとpidを返す
    '''
    annotation = pd.read_csv('/home/ubuntu/workspace_yamanishi/ALM-pedestrian-attribute/data/label/attribute_C6_2023-12-16_13-00-00.txt',
                         names=['image_path', 'Female', 'AgeLess18', 'AgeYoung', 'AgeOld', 'gender_condition', 'age_condition'])
    annotation['image_dir'] = annotation['image_path'].apply(lambda x:x.split('/')[-2])
    target_ids = annotation['image_dir'].values
    image_files = []
    image_files_first = []
    image_dirs = []
    image_dirs_first = []
    for i,id in enumerate(target_ids):
        curDir = os.path.join('/home/ubuntu/workspace_yamanishi/Camera-App-ML/src/DATA/person_image/C6_2023-12-16_13-00-00', id)
        files = os.listdir(curDir)
        image_files+=[os.path.join(curDir, f) for f in files]
        image_files_first+=[os.path.join(curDir, f) for f in files if f=='0.jpg']
        #image_dirs+=[int(curDir.split('/')[-1]) for _ in range(len(files))]
        image_dirs+=[i for _ in range(len(files))]
        image_dirs_first+=[i for f in files if f=='0.jpg']
        #image_dirs_first+=[int(curDir.split('/')[-1]) for f in files if f=='0.jpg']
            #print(image_files_first)
        if len(image_files)>=max_image_num:break
    #print(image_dirs, target_ids)
    # image_dirs = 
    # _, image_dirs = np.unique(np.array(image_dirs), return_inverse=True)
    # _, image_dirs_first = np.unique(np.array(image_dirs_first), return_inverse=True)
    #print(image_dirs_first, target_ids)
    return image_files, image_dirs, image_files_first, image_dirs_first


def get_image_dirs(dir, max_image_num=5000, also_first_only=False):
    '''
    対象のディレクトリ内の画像ファイルとpidを返す
    '''
    image_files = []
    if also_first_only:image_files_first = []
    image_dirs = []
    for curDir, dirs, files in os.walk(dir):
        #print(curDir)
        if len(dirs)==0:
            image_files+=[os.path.join(curDir, f) for f in files]
            if also_first_only:image_files+=[os.path.join(curDir, f) for f in files if f=='0.jpg']
            image_dirs+=[int(curDir.split('/')[-1]) for _ in range(len(files))]
        if len(image_files)>=max_image_num:break
    _, image_dirs = np.unique(np.array(image_dirs), return_inverse=True)
    if not also_first_only:
        return image_files, image_dirs
    else:
        return image_files, image_dirs, image_files_first

def real_data_evaluate_age(eval_dirs, model, attr_num, dataset, max_image_num=5000, logger=None,
                           also_first_only=False):
    model.eval()
    attributes_all = []
    for eval_dir in eval_dirs:
        logger.info(f'evaluating {eval_dir}..')
        if not also_first_only:
            image_files, image_dirs = get_image_dirs(eval_dir, max_image_num=max_image_num)
        else:
            image_files, image_dirs, image_files_first = get_image_dirs(eval_dir, )
        dir_num = len(np.unique(image_dirs))
        eval_dataset = SimpleDataset(image_files)
        if also_first_only:eval_dataset_first = SimpleDataset(image_files_first)
        attributes = np.zeros((dir_num, attr_num))
        count_votes = np.zeros(dir_num)
        batch_size = 50
        data_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
        for i, inputs in tqdm(enumerate(data_loader)):
            # 推論
            with torch.no_grad():
                inputs = inputs.cuda()
                output = model(inputs)[0][0]
                output = torch.softmax(output.data, dim=-1).cpu().numpy()
            #output = torch.sigmoid(output.data).cpu().numpy()
            #output = np.where(output > 0.5, 1, 0)
            inds_tmp = image_dirs[i*batch_size:(i+1)*batch_size]
            # 属性の総和と登場回数をそれぞれ加算する
            np.add.at(attributes, inds_tmp, output)
            count_votes+=np.bincount(inds_tmp, minlength=count_votes.size)
            
        if also_first_only:data_loader_first = DataLoader(eval_dataset_first, batch_size=batch_size, shuffle=False)
        attributes_first = np.zeros((dir_num, attr_num))
        count_votes_first = np.zeros(dir_num)
        if also_first_only:
            for i, inputs in tqdm(enumerate(data_loader_first)):
                # 推論
                with torch.no_grad():
                    inputs = inputs.cuda()
                    output = model(inputs)[0][0]
                    output = torch.softmax(output.data, dim=-1).cpu().numpy()
                inds_tmp = image_dirs[i*batch_size:(i+1)*batch_size]
                # 属性の総和と登場回数をそれぞれ加算する
                np.add.at(attributes_first, inds_tmp, output)
                count_votes_first+=np.bincount(inds_tmp, minlength=count_votes.size)

        # majority vote
        attributes = attributes/count_votes.reshape(-1, 1)
        attributes = np.argmax(attributes, axis=-1)
        
        if also_first_only:
            attributes_first = attributes_first/count_votes_first.reshape(-1, 1)
            attributes_first = np.argmax(attributes_first, axis=-1)
        #attributes = np.where(attributes>0.5, 1, 0)
        # 全属性の平均
        logger.info('AgeLess18 ratio: %f', len(attributes[attributes == 0])/len(attributes))
        logger.info('AgeYoung ratio: %f', len(attributes[attributes == 1])/len(attributes))
        logger.info('AgeOld ratio: %f', len(attributes[attributes == 2])/len(attributes))
    if not also_first_only:
        return attributes
    else:
        return attributes, attributes_first

def real_data_evaluate(eval_dirs, model, attr_num, dataset, max_image_num=5000):
    model.eval()
    attributes_all = []
    for eval_dir in eval_dirs:
        print(f'evaluating {eval_dir}..')
        image_files, image_dirs = get_image_dirs(eval_dir, max_image_num=max_image_num)
        dir_num = len(np.unique(image_dirs))
        eval_dataset = SimpleDataset(image_files)
        attributes = np.zeros((dir_num, attr_num))
        count_votes = np.zeros(dir_num)
        batch_size = 50
        data_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
        for i, inputs in tqdm(enumerate(data_loader)):
            # 推論
            inputs = inputs.cuda(non_blocking=True)
            output = model(inputs)[0][0]
            output = torch.sigmoid(output.data).cpu().numpy()
            output = np.where(output > 0.5, 1, 0)
            inds_tmp = image_dirs[i*batch_size:(i+1)*batch_size]
            # 属性の総和と登場回数をそれぞれ加算する
            np.add.at(attributes, inds_tmp, output)
            count_votes+=np.bincount(inds_tmp, minlength=count_votes.size)

        # majority vote
        attributes = attributes/count_votes.reshape(-1, 1)
        attributes = np.where(attributes>0.5, 1, 0)
        # 全属性の平均
        attribute_ratio = np.sum(attributes, axis=0)/len(attributes)
        #print('dataset', dataset)
        if dataset == 'pa100k':
            female_ind, ageless_ind, ageover_ind = 22, 19, 21
        elif dataset == 'gender_age':
            female_ind, ageless_ind, ageover_ind = 0, 1, 4
        print_color_and_alert('Female Ratio', attribute_ratio[female_ind], 0.3, 0.4, 0.6, 0.7)
        print_color_and_alert('AgeLess18 Ratio', attribute_ratio[ageless_ind], 0.005, 0.01, 0.1, 0.2)
        print_color_and_alert('AgeOver60 Ratio', attribute_ratio[ageover_ind], 0.005, 0.01, 0.1, 0.2)
        attributes_all.append(attribute_ratio)
    return np.mean(np.stack(attributes_all), axis=0)

def get_pseudo_label_and_conf(model, args, labeled_dataset, unlabeled_dataset) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    '''
    get feature of labeled image and unlabeled image
    output:
        all_feature: [attr_num x N x feature_dim]
        pseudo_label: [N x attr_num]
        confidence: [N x attr_num]
    '''
    model.eval()
    unlabeled_loader_tmp = DataLoader(unlabeled_dataset, batch_size=100, num_workers=args.num_workers, shuffle=False, drop_last=False)
    pseudo_label, confidence = [], []
    attr_num = len(labeled_dataset.attributes[0])
    with torch.no_grad():
        for imgs, labels in tqdm(unlabeled_loader_tmp):
            output = model(imgs[0])[0][0]
            #print(output.shape)
            pred = torch.sigmoid(output)
            confidence.append(pred.cpu())
            pseudo_label.append(torch.ge(pred, 0.5).cpu().to(int))
            
    return torch.cat(pseudo_label, axis=0).numpy(), torch.cat(confidence, axis=0).numpy()