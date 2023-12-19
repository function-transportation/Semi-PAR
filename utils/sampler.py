import numpy as np
import pulp
from torch.utils.data.sampler import BatchSampler, RandomSampler
def get_target_ratio(ratio):
    if ratio>0.9:
        return ratio-0.1
    elif ratio>0.5:
        return ratio-0.15#(ratio//0.2+1) * 0.2
    elif ratio>0.1:
        return ratio+0.15#ratio//0.2*0.2
    else:
        return ratio-0.1

class CurriculumSampler(BatchSampler):
    def __init__(self, dataset, epoch, batch_size, unlabel=False, max_epoch=30):
        #super().__init__()
        self.batch_size = batch_size
        self.dataset = dataset
        # self.label: [n x num_attr]
        if unlabel:
            self.label = dataset.pseudo_label
            self.confidence = dataset.confidence
        else:
            self.label = np.array(dataset.attributes)
        self.epoch = epoch
        self.max_epoch = max_epoch
        self.num_sample = self.label.shape[0]
        self.attr_num = self.label.shape[1]
        self.target_attribute_ratio = np.sum(self.label, axis=0)/self.num_sample
        self.final_attribute_ratio = [get_target_ratio(r) for r in self.target_attribute_ratio]
        self.target_attribute_ratio = [(self.epoch/self.max_epoch)*f + ((self.max_epoch-self.epoch)*t)/self.max_epoch for f, t in zip(self.final_attribute_ratio, self.target_attribute_ratio)]
        
        #print(epoch, self.target_attribute_ratio)
        #init_sum = sum(self.target_attribute_ratio)
        #self.target_attribute_ratio = self.target_attribute_ratio**np.cos(epoch*np.pi/30/2)
        #self.target_attribute_ratio = self.target_attribute_ratio*init_sum/sum(self.target_attribute_ratio)
        #print(epoch, self.target_attribute_ratio)
        #scale_ratio = np.cos(epoch*np.pi/50/2)
        #down_ratio = ((max(self.target_attribute_ratio)/(max(self.target_attribute_ratio)**scale_ratio)) + (min(self.target_attribute_ratio)/(min(self.target_attribute_ratio)**scale_ratio)))/2
        #print(down_ratio)
        #self.target_attribute_ratio = [i**scale_ratio/down_ratio for i in self.target_attribute_ratio]
        #print(epoch, self.target_attribute_ratio)
        #print(self.target_attribute_ratio)
        problem = pulp.LpProblem("index_selection", pulp.LpMaximize)
        W = [pulp.LpVariable(f"W_{i}", lowBound=0, upBound=2, cat=pulp.LpInteger) for i in range(self.num_sample)]
        if epoch<10:
            epsilon = 0.05 # 仮の値
        elif epoch<20:
            epsilon = 0.05
        else:
            epsilon = 0.1
        
        for j in range(self.attr_num):
            problem += pulp.lpSum([W[i] * self.label[i][j] for i in range(self.num_sample)]) <= (self.target_attribute_ratio[j]+epsilon)*self.num_sample
            problem += pulp.lpSum([W[i] * self.label[i][j] for i in range(self.num_sample)]) >= (self.target_attribute_ratio[j]-epsilon)*self.num_sample

        problem += pulp.lpSum(W) == self.num_sample
        problem.solve(pulp.PULP_CBC_CMD(msg = False))
        W_optimal = np.array([pulp.value(var) for var in W]).astype(int)
        print('optimal', W_optimal)
        self.index = ([ind for ind, opt_w in enumerate(W_optimal) for _ in range(opt_w)])
        np.random.shuffle(self.index)
        #self.index = list(self.index)
        
    def __iter__(self):
        for i in range(len(self.index)):
            yield self.index[i]