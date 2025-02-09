import pulp
import numpy as np
from torch.utils.data.sampler import BatchSampler

def get_target_ratio(ratio):
    if ratio>0.9:
        return ratio-0.1
    elif ratio>0.5:
        return ratio-0.15 #(ratio//0.2+1) * 0.2
    elif ratio>0.1:
        return ratio+0.15 #ratio//0.2*0.2
    else:
        return ratio+0.1
    
def get_target_ratio(balanced_ratio, current_ratio, attribute, epoch=None, max_epoch=None):
    #if attribute == 'Female':
    #    return min(max(balanced_ratio*2-current_ratio-0.07, current_ratio/2), balanced_ratio*1.5)
    #else:
    max_epoch = max_epoch
    epsilon = 0.1*(max_epoch-min(max_epoch, max_epoch))//max_epoch
    if attribute=='Female':
        epsilon = 0.05*(max_epoch-min(max_epoch, epoch))//max_epoch
        ratio = min(max(balanced_ratio*2-current_ratio, current_ratio/2), balanced_ratio*1.5)
        ratio = ratio - epsilon
    else:
        epsilon = 0.05*(max_epoch-min(max_epoch, epoch))//max_epoch
        ratio = min(max(balanced_ratio*2+current_ratio, current_ratio/2), balanced_ratio*1.5)
        ratio = ratio + epsilon
    return ratio
    
class BiasBalancer(BatchSampler):
    def __init__(self, dataset, epoch, batch_size, real_eval_attribute_ratio, unlabel=False):
        self.attributes = ['Hat','Glasses','ShortSleeve','LongSleeve','UpperStride',
            'UpperLogo','UpperPlaid','UpperSplice','LowerStripe','LowerPattern',
            'LongCoat','Trousers','Shorts','Skirt&Dress','boots',
            'HandBag','ShoulderBag','Backpack','HoldObjectsInFront','AgeOver60',
            'Age18-60','AgeLess18','Female','Front','Side','Back']
        self.batch_size = batch_size
        self.dataset = dataset
        # self.label: [n x num_attr]
        if unlabel:
            self.label = dataset.pseudo_label
            self.confidence = dataset.confidence
        else:
            self.label = np.array(dataset.attributes)
        self.epoch = epoch
        
        self.num_all_sample = self.label.shape[0]
        self.attr_num = self.label.shape[1]
        self.currnt_attribute_ratio = real_eval_attribute_ratio#np.sum(self.label, axis=0)/self.num_all_sample
        print(self.currnt_attribute_ratio)
        self.balanced_ratios = {
            22: 0.5,  # 女性
            19: 0.05, # 子供
            21: 0.1   # 60歳以上
        }
        
        problem = pulp.LpProblem("index_selection", pulp.LpMaximize)
        W = [pulp.LpVariable(f"W_{i}", lowBound=0, upBound=1, cat=pulp.LpInteger) for i in range(self.num_all_sample)]
        
        num_sample = int(self.num_all_sample * 0.5)
        for attr_j in range(self.attr_num):
            if attr_j not in self.balanced_ratios:continue
            target_ratio = get_target_ratio(self.balanced_ratios[attr_j],
                                            self.currnt_attribute_ratio[attr_j],
                                            self.attributes[attr_j],
                                            epoch=epoch,
                                            max_epoch=30)
            print(attr_j, target_ratio)
                                            
            epsilon = target_ratio * 0.05
            problem += pulp.lpSum([W[i] * self.label[i][attr_j] for i in range(self.num_all_sample)]) <= (target_ratio+epsilon)*num_sample
            problem += pulp.lpSum([W[i] * self.label[i][attr_j] for i in range(self.num_all_sample)]) >= (target_ratio-epsilon)*num_sample

        problem += pulp.lpSum(W) == int(num_sample)
        problem.solve(pulp.PULP_CBC_CMD(msg = False))
        W_optimal = np.array([pulp.value(var) for var in W]).astype(int)
        self.index = np.array([ind for ind, opt_w in enumerate(W_optimal) for _ in range(opt_w)])
        np.random.shuffle(self.index)
        #self.index = list(self.index)
        
    def __iter__(self):
        for i in range(len(self.index)):
            yield self.index[i]

class CurriculumSampler(BatchSampler):
    def __init__(self, dataset, epoch, batch_size, unlabel=False):
        self.batch_size = batch_size
        self.dataset = dataset
        # self.label: [n x num_attr]
        if unlabel:
            self.label = dataset.pseudo_label
            self.confidence = dataset.confidence
        else:
            self.label = np.array(dataset.attributes)
        self.max_epoch = 50
        self.epoch = epoch
        self.num_sample = self.label.shape[0]
        self.attr_num = self.label.shape[1]
        self.target_attribute_ratio = np.sum(self.label, axis=0)/self.num_sample
        self.final_attribute_ratio = [get_target_ratio(r) for r in self.target_attribute_ratio]
        self.target_attribute_ratio = [(self.epoch/self.max_epoch)*f + ((self.max_epoch-self.epoch)/self.max_epoch)*t for f, t in zip(self.final_attribute_ratio, self.target_attribute_ratio)]
        print(epoch, self.target_attribute_ratio)
        problem = pulp.LpProblem("index_selection", pulp.LpMaximize)
        W = [pulp.LpVariable(f"W_{i}", lowBound=1, upBound=2, cat=pulp.LpInteger) for i in range(self.num_sample)]
        if epoch<10:
            epsilon = 0.05 # 仮の値
        elif epoch<20:
            epsilon = 0.
        else:
            epsilon = 0.2
        
        for j in range(self.attr_num):
            problem += pulp.lpSum([W[i] * self.label[i][j] for i in range(self.num_sample)]) <= (self.target_attribute_ratio[j]+epsilon)*self.num_sample
            problem += pulp.lpSum([W[i] * self.label[i][j] for i in range(self.num_sample)]) >= (self.target_attribute_ratio[j]-epsilon)*self.num_sample

        problem += pulp.lpSum(W) == self.num_sample
        problem.solve(pulp.PULP_CBC_CMD(msg = False))
        W_optimal = np.array([pulp.value(var) for var in W]).astype(int)
        self.index = ([ind for ind, opt_w in enumerate(W_optimal) for _ in range(opt_w)])
        np.random.shuffle(self.index)
        #self.index = list(self.index)
        
    def __iter__(self):
        for i in range(len(self.index)):
            yield self.index[i]