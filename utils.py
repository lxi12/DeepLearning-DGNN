import torch
import time
import random
import math
import argparse     #command line argument processing module 
import yaml
import numpy as np

'''Main purpose of this file:
  - get normalize adjacency matrix at a time from normalize_adj
  - get nodes features matrix from get_1_hot_deg_feats
'''

class Namespace(object):
    '''
    helps referencing object in a dictionary as dict.key instead of dict['key']
    '''
    def __init__(self, adict):
        self.__dict__.update(adict)

ECOLS = Namespace({'source': 0,
                     'target': 1,
                     'time': 2,
                     'label':3}) 
           
    
def sparse_prepare_tensor(tensor,torch_size, ignore_batch_dim = True):
    if ignore_batch_dim:
        tensor = sp_ignore_batch_dim(tensor)
    tensor = make_sparse_tensor(tensor,
                                tensor_type = 'float',
                                torch_size = torch_size)
    return tensor

def sp_ignore_batch_dim(tensor_dict):  
    tensor_dict['idx'] = tensor_dict['idx'][0]
    tensor_dict['vals'] = tensor_dict['vals'][0]
    return tensor_dict

def make_sparse_tensor(adj,tensor_type,torch_size): 
    # tensor is spatial, convert it to sparse_tensor {'idx': [matrix index], 'vals': vals}
    if len(torch_size) == 2:
        tensor_size = torch.Size(torch_size)
    elif len(torch_size) == 1:
        tensor_size = torch.Size(torch_size*2)

    if tensor_type == 'float':
        test = torch.sparse.FloatTensor(adj['idx'].t(),
                                      adj['vals'].type(torch.float),
                                      tensor_size)
        return torch.sparse.FloatTensor(adj['idx'].t(),
                                      adj['vals'].type(torch.float),
                                      tensor_size)
    elif tensor_type == 'long':
        return torch.sparse.LongTensor(adj['idx'].t(),
                                      adj['vals'].type(torch.long),
                                      tensor_size)
    else:
        raise NotImplementedError('only make floats or long sparse tensors')


def reset_param(t): #initialize weights 
    stdv = 2. / math.sqrt(t.size(0))
    t.data.uniform_(-stdv,stdv)

def random_param_value(param, param_min, param_max, type='int'):
    if str(param) is None or str(param).lower()=='none':
        if type=='int':
            return random.randrange(param_min, param_max+1)
        elif type=='logscale':
            interval=np.logspace(np.log10(param_min), np.log10(param_max), num=100)
            return np.random.choice(interval,1)[0]
        else:
            return np.random.uniform(param_min, param_max)
    else:
        return param


def create_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--config_file',default='parameters_bitcoin_alpha_linkpred_meta_gcn.yaml', type=argparse.FileType(mode='r'), help='optional, yaml file containing parameters to be used, overrides command line parameters')
    return parser

def parse_args(parser):
    args = parser.parse_args()
    if args.config_file:
        data = yaml.safe_load(args.config_file) 
        delattr(args, 'config_file')            #delete attribute args from object 'config_file'
        arg_dict = args.__dict__
        for key, value in data.items():
            arg_dict[key] = value

    if args.gcn_parameters['layer_2_feats_same_as_l1'] or args.gcn_parameters['layer_2_feats_same_as_l1'].lower()=='true':
        args.gcn_parameters['layer_2_feats'] = args.gcn_parameters['layer_1_feats']     
    if args.gcn_parameters['lstm_l2_feats_same_as_l1'] or args.gcn_parameters['lstm_l2_feats_same_as_l1'].lower()=='true':
        args.gcn_parameters['lstm_l2_feats'] = args.gcn_parameters['lstm_l1_feats']
   
    return args
 
 
def get_1_hot_deg_feats(adj,max_deg,num_nodes):
    #For now it'll just return a 1-hot vector,0 in most dimensions, and 1 in a single dimensions. Max_deg 67, num_nodes 3783
    new_vals = torch.ones(adj['idx'].size(0))
    new_adj = {'idx':adj['idx'], 'vals': new_vals}    #change adj vals all to 1
    degs_out, _ = get_degree_vects(new_adj,num_nodes) #get degree col vector of nodes 

    degs_out = {'idx': torch.cat([torch.arange(num_nodes).view(-1,1),
                                  degs_out.view(-1,1)],dim=1),
                'vals': torch.ones(num_nodes)}   #a dict{'idx':[0->3783 nodes index, 0->66 degrees of nodes], 'vals':[3783 of 1]}
    degs_out = make_sparse_tensor(degs_out,'long',[num_nodes,max_deg]) #sparse tensor, size([3783(row nodes index), 67(col degs index)]), nnz=3783

    hot_1 = {'idx': degs_out._indices().t(),
             'vals': degs_out._values()}
    return hot_1  #a dict{'idx':[0->3783 nodes index, 0->66 degrees of node], 'vals':[3783 of 1]}
 

def get_max_degs(args,dataset,all_window=False):
    #select the entire dataset (including validation and test) as one time
    cur_adj = get_sp_adj(edges = dataset.edges,
                         time = dataset.max_time,   #136
                         weighted = False,
                         time_window = dataset.max_time+1)
                             
    cur_out, cur_in = get_degree_vects(cur_adj,dataset.num_nodes)
    max_deg_out = int(cur_out.max().item()) + 1    #67
    max_deg_in = int(cur_in.max().item()) + 1      #67, some nodes have 0 degree at some time steps

    return max_deg_out, max_deg_in  


def get_degree_vects(adj,num_nodes):
    adj = make_sparse_tensor(adj,'long',[num_nodes])
    degs_out = adj.matmul(torch.ones(num_nodes,1,dtype = torch.long))    #([3783, 1])
    degs_in = adj.t().matmul(torch.ones(num_nodes,1,dtype = torch.long)) #([3783, 1])
    return degs_out, degs_in


def get_sp_adj(edges,time,weighted,time_window):
    #get different timesteps spatial adjacency matrix 
    idx = edges['idx']
    subset = idx[:,ECOLS.time] <= time
    subset = subset * (idx[:,ECOLS.time] > (time - time_window)) # T or F, select over time subset
    idx = edges['idx'][subset][:,[ECOLS.source, ECOLS.target]]  
    vals = edges['vals'][subset]
    out = torch.sparse.FloatTensor(idx.t(),vals).coalesce()
    
    idx = out._indices().t()
    if weighted:
        vals = out._values()
    else:
        vals = torch.ones(idx.size(0),dtype=torch.long) 
    return {'idx': idx, 'vals': vals}

def normalize_adj(adj,num_nodes):
    '''
    takes an adj matrix as a dict with idx and vals and normalize it by: 
        - adding an identity matrix, A`
        - computing the degree vector, D`
        - multiplying each element of the adj matrix (aij) by (di*dj)^-1/2, 
    '''
    idx = adj['idx']
    vals = adj['vals']

    sp_tensor = torch.sparse.FloatTensor(idx.t(),vals.type(torch.float),torch.Size([num_nodes,num_nodes])) 
    sparse_eye = make_sparse_eye(num_nodes)
    sp_tensor = sparse_eye + sp_tensor

    idx = sp_tensor._indices()
    vals = sp_tensor._values()

    degree = torch.sparse.sum(sp_tensor,dim=1).to_dense()
    di = degree[idx[0]]
    dj = degree[idx[1]]

    vals = vals * ((di * dj) ** -0.5)
    return {'idx': idx.t(), 'vals': vals} # a dict{'idx': [[source, target]](size([M >3783,2])), 'vals': [normalized adjacency values([M])}, sparse tensor 

def make_sparse_eye(size):
    eye_idx = torch.arange(size)
    eye_idx = torch.stack([eye_idx,eye_idx],dim=1).t()
    vals = torch.ones(size)
    eye = torch.sparse.FloatTensor(eye_idx,vals,torch.Size([size,size]))
    return eye 


def get_non_existing_edges(adj,number, tot_nodes, smart_sampling, existing_nodes=None): 
    t0 = time.time()
    idx = adj['idx'].t().numpy() # [[source-ids], [target_ids]]
    true_ids = get_edges_ids(idx,tot_nodes)
    true_ids = set(true_ids)
  
    #the maximum of edges would be all edges that don't exist between nodes that have edges
    num_edges = min(number,idx.shape[1] * (idx.shape[1]-1) - len(true_ids))     # n*(n-1)

    if smart_sampling:
        def sample_edges(num_edges):
            from_id = np.random.choice(idx[0],size = num_edges,replace = True)
            to_id = np.random.choice(existing_nodes,size = num_edges, replace = True)

            if num_edges>1:
                edges = np.stack([from_id,to_id])
            else:
                edges = np.concatenate([from_id,to_id])
            return edges
    else:           #simple sampling
        def sample_edges(num_edges):     
            if num_edges > 1:
                edges = np.random.randint(0,tot_nodes,(2,num_edges))
            else:
                edges = np.random.randint(0,tot_nodes,(2,))
            return edges

    edges = sample_edges(num_edges*4)         #oversampling_factor
    edge_ids = get_edges_ids(edges, tot_nodes)
    
    out_ids = set()
    num_sampled = 0
    sampled_indices = []
    for i in range(num_edges*4):
        eid = edge_ids[i]
        #ignore if any of these conditions happen
        if eid in out_ids or edges[0,i] == edges[1,i] or eid in true_ids:
            continue

        #add the eid and the index to a list
        out_ids.add(eid)
        sampled_indices.append(i)
        num_sampled += 1

        #if we have sampled enough edges break
        if num_sampled >= num_edges:
            break

    edges = edges[:,sampled_indices]
    edges = torch.tensor(edges).t()
    vals = torch.zeros(edges.size(0),dtype = torch.long)
    return {'idx': edges, 'vals': vals}
    
def get_edges_ids(sp_idx, tot_nodes):
    return sp_idx[0]*tot_nodes + sp_idx[1]