import torch
import utils as u
import numpy as np
from torch.utils.data import Dataset, DataLoader
 
'''Main purpose of this file:
 - define class to split the dataset
 - define Loss function
'''

class splitter():
    '''
    creates 3 splits
    train
    validation
    test
    '''
    def __init__(self,args,tasker):
 
        assert args.train_proportion + args.val_proportion < 1, \
            'there\'s no space for test samples'
        #only the training one requires special handling on start, the others are fine with the split IDX.
        start = tasker.data.min_time + args.num_hist_steps                   # 0+num_hist_Steps,
        end = args.train_proportion                                          # end 95, not training at G95      
        end = int(np.floor(tasker.data.max_time.type(torch.float) * end))    

        train = data_split(tasker, start, end, test = False, mode = 'train')
        train = DataLoader(train,**args.data_loading_params)
        
        # the validation     
        start = end                                                        
        end = args.val_proportion + args.train_proportion                    
        end = int(np.floor(tasker.data.max_time.type(torch.float) * end))    #[95, 108)
        
        val = data_split(tasker, start, end, test = True, all_edges=True)
        val = DataLoader(val,num_workers=args.data_loading_params['num_workers'])
        
        # the testing 
        start = end    
        #the +1 is because I assume that max_time exists in the dataset
        end = int(tasker.max_time) + 1
        test = data_split(tasker, start, end, test = True, all_edges=True)   #[108, 136)    
        test = DataLoader(test,num_workers=args.data_loading_params['num_workers'])
            
        print ('Dataset splits sizes:  train',len(train), 'val',len(val), 'test',len(test))
        self.tasker = tasker
        self.train = train
        self.val = val
        self.test = test     


class data_split(Dataset):
    def __init__(self, tasker, start, end, test, **kwargs):
        '''
        start and end are indices indicating what items belong to this split
        '''
        self.tasker = tasker
        self.start = start
        self.end = end
        self.test = test
        self.kwargs = kwargs

    def __len__(self):
        return self.end-self.start

    def __getitem__(self,idx):
        idx = self.start + idx
        t = self.tasker.get_sample(idx, test = self.test)
        return t

 
class Cross_Entropy(torch.nn.Module):
    """Define loss function"""
    def __init__(self, args, dataset):
        super().__init__()
        weights = torch.tensor(args.class_weights).to(args.device)
        self.weights = self.dyn_scale(args.task, dataset, weights)
    
    def dyn_scale(self,task,dataset,weights):
        def scale(labels):
            return weights
        return scale
    

    def logsumexp(self,logits):        #logits 2 cols
        m,_ = torch.max(logits,dim=1)  #dim=1 row, output each row max values and its index, a col 
        m = m.view(-1,1)               #make sure it is a col 
        sum_exp = torch.sum(torch.exp(logits-m),dim=1, keepdim=True) #each row sumup, a col 
        return m + torch.log(sum_exp)  #a col 
    
    
    def forward(self,logits,labels):
        '''
        logits is a matrix M by C where m is the number of classifications and C are the number of classes, c=2
        labels is a integer tensor of size M where each element corresponds to the class that prediction i
        should be matching to
        '''
        labels = labels.view(-1,1)  #view(-1, 1)transform a row to a col 
        alpha = self.weights(labels)[labels].view(-1,1) # weights tensor, a col
        loss = alpha * (- logits.gather(-1,labels) + self.logsumexp(logits)) #gather logits along dim=-1,row by index labels, a col 
        return loss.mean()  #loss is a col


if __name__ == '__main__':
    dataset = u.Namespace({'num_non_existing': torch.tensor(10)})
    args = u.Namespace({'class_weights': [1.0,2.0],
                        'task': 'no_link_pred'})
    args.device='cpu'
    labels = torch.tensor([1,0])
    ce_ref = torch.nn.CrossEntropyLoss(reduction='sum',weight=torch.Tensor(args.class_weights))
    ce = Cross_Entropy(args,dataset)
 
    logits = torch.tensor([[1.0,-1.0],
                           [1.0,-1.0]])
    logits = torch.rand((5,2))         #random uniform nums from [0, 1], size([5, 2])
    labels = torch.randint(0,2,(5,))   #random int from[0, 2], size([1,5])
    print(ce(logits,labels)- ce_ref(logits,labels))
    exit()
    
    ce.logsumexp(logits)
 
    x = torch.tensor([0,1])
    y = torch.tensor([1,0]).view(-1,1)    
    print(logits.gather(-1,y))