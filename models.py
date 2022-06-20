import torch
import utils as u
from argparse import Namespace 
from torch.nn.parameter import Parameter
import torch.nn as nn
import math
 
"""Design the GCN and EGCN_O architecture"""

class Sp_GCN(torch.nn.Module):
    def __init__(self,args,activation):
        super().__init__()
        self.activation = activation
        self.num_layers = args.num_layers
        self.w_list = nn.ParameterList()  #set w_list to parameterlist, can be trained 

        for i in range(self.num_layers): #2 layers
            if i==0: 
                w_i = Parameter(torch.Tensor(args.feats_per_node, args.layer_1_feats)) #(input, gcn layer1)
                u.reset_param(w_i)
            else:
                w_i = Parameter(torch.Tensor(args.layer_1_feats, args.layer_2_feats))  #(gcn layer1, gcn layer2)
                u.reset_param(w_i)
            self.w_list.append(w_i) #output weights parameterList, size([F,F]), F gcn parameter: layer_1_feats

    def forward(self, A_list, Nodes_list, nodes_mask_list, vars = None):
        if vars is None:
            vars = self.w_list
        node_feats = Nodes_list[-1] #take only last nodes feats matrix from nodes_list, 
        Ahat = A_list[-1]           #take only last adj matrix in time, sparse tensor ([3783, 3783])
        last_l = self.activation(Ahat.matmul(node_feats.matmul(vars[0]))) #H1=A^*X*W0, sparse multiplication, W0 size([D,F(gcn.layer_1_feats)]), H1 size([3783,F])

        for i in range(1, self.num_layers):
            last_l = self.activation(Ahat.matmul(last_l.matmul(vars[i]))) #H2=A^*H1*W1, W1 size([F,F])

        return last_l #output last layer H2 size([3783, F]), node embeddings at a time 
 
 
class Classifier(torch.nn.Module):
    def __init__(self,args,out_features=2, in_features = None): #output_features, 2 class, 1 or 0
        super(Classifier,self).__init__()
        self.activation = torch.nn.ReLU(inplace=True)  #use ReLU activation funciton 

        if in_features is not None:
            num_feats = in_features
        else:
            num_feats = args.gcn_parameters['layer_2_feats'] * 2
        print ('CLS num_feats',num_feats)

        #use sequential container to house multilayer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(in_features=num_feats,                           
                                                       out_features=args.gcn_parameters['cls_feats']),
                                       self.activation,
                                       torch.nn.Linear(in_features=args.gcn_parameters['cls_feats'],    
                                                       out_features=out_features)) 

    def forward(self, x):
        return self.mlp(x)

 
class EGCN_O(Sp_GCN):
    def __init__(self, args, activation):
        super().__init__(args, activation)                      # straight use torch.nn.LSTM
        self.rnn_l1 = nn.LSTM(input_size=args.layer_1_feats,    #num of input features F
                              hidden_size=args.lstm_l1_feats,   #num of output features F, need be the same 
                              num_layers=args.lstm_l1_layers)
                              
        self.rnn_l2 = nn.LSTM(input_size=args.layer_2_feats,    #as GCN has 2 layers, place GCN in LSTM
                              hidden_size=args.lstm_l2_feats,
                              num_layers=args.lstm_l2_layers)                         
        
        self.W2 = Parameter(torch.Tensor(args.lstm_l1_feats, args.layer_2_feats)) #size([F, F]), output features 
        u.reset_param(self.W2)  #W2 happens each timestep GCN, reset every time, no use 

        
    def forward(self, A_list, Nodes_list=None,nodes_mask_list=None):
        l1_seq = []                      #GCN layer1 H1 sequence
        l2_seq = []                      #GCN layer2 H2 sequence
        for t, Ahat in enumerate(A_list):
            node_feats = Nodes_list[t]
            l1 = self.activation(Ahat.matmul(node_feats.matmul(self.w_list[0])))  #H1=A^*X*W0, sparse multiplication
            l1_seq.append(l1)
            
        l1_seq = torch.stack(l1_seq)          #concatenates a seq along a new dimension 
        out_l1, _ = self.rnn_l1(l1_seq, None) #LSTM input(H1 sequence, hidden state),  
                                              #LSTM output(net time H1 seq, updated hidden state) 
        for i in range(len(A_list)):
            Ahat = A_list[i]         #size([3783, 3783]), need keep input A_list for a time 
            out_t_l1 = out_l1[i]     #size([3783, F]), F rnn_l1 out features 
            l2 = self.activation(Ahat.matmul(out_t_l1).matmul(self.w_list[1])) #H2=A^*H1*W1
            l2_seq.append(l2)

        l2_seq = torch.stack(l2_seq)
        out, _ = self.rnn_l2(l2_seq, None)  #input(H2 seq, hidden state), W1 initally None as well 
                                            #output(next time H2 seq, hidden state)

        return out[-1] #output last of H2 seq, the final node embedding for the whole model 
        