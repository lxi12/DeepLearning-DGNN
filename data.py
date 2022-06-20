import torch
import utils as u
import os

"""Main purpose of this file, get edges from dataset"""

class bitcoin_dataset():
    def __init__(self,args):
        #assert args.task in ['link_pred', 'edge_cls'], 
        self.ecols = u.Namespace({'FromNodeId': 0,
                                  'ToNodeId': 1,
                                  'Weight': 2,
                                  'TimeStep': 3
                                })
        args.bitcoin_args = u.Namespace(args.bitcoin_args)

        #build edge data structure
        edges = self.load_edges(args.bitcoin_args)

        edges = self.make_contigous_node_ids(edges)                 #make node ids contigous, 0->3782
        num_nodes = edges[:,[self.ecols.FromNodeId,
                            self.ecols.ToNodeId]].unique().size(0)  # 3783

        timesteps = self.aggregate_by_time(edges[:,self.ecols.TimeStep],args.bitcoin_args.aggr_time)
        self.max_time = timesteps.max() #135
        self.min_time = timesteps.min() #0
        edges[:,self.ecols.TimeStep] = timesteps  #convert TimeStep to every 3 weeks per unit 

        #modify all positive rating to 1, all negative rating to -1
        edges[:,self.ecols.Weight] = self.cluster_negs_and_positives(edges[:,self.ecols.Weight])
        

        #add the reversed link to make the graph undirected
        edges = torch.cat([edges,edges[:,[self.ecols.ToNodeId,
                                          self.ecols.FromNodeId,
                                          self.ecols.Weight,
                                          self.ecols.TimeStep]]])  # ([48372, 4])

        #separate classes
        sp_indices = edges[:,[self.ecols.FromNodeId,
                              self.ecols.ToNodeId,
                              self.ecols.TimeStep]].t()  #([3, 48372])
        sp_values = edges[:,self.ecols.Weight]           #([48372]), 1 or -1

        
        neg_mask = sp_values == -1
        neg_sp_indices = sp_indices[:,neg_mask]
        neg_sp_values = sp_values[neg_mask]
        neg_sp_edges = torch.sparse.LongTensor(neg_sp_indices
                                              ,neg_sp_values,
                                              torch.Size([num_nodes,
                                                          num_nodes,
                                                          self.max_time+1])).coalesce()
        # size([3783, 3783, 137]), nnz=2938, number non zero elements, values -1

        pos_mask = sp_values == 1
        pos_sp_indices = sp_indices[:,pos_mask]
        pos_sp_values = sp_values[pos_mask]
        pos_sp_edges = torch.sparse.LongTensor(pos_sp_indices
                                              ,pos_sp_values,
                                              torch.Size([num_nodes,
                                                          num_nodes,
                                                          self.max_time+1])).coalesce()

        #size([3783, 3783, 137]), nnz=28986, values {1,2}, Eij can be 2 in a timestep 
        
        #scale positive class to separate after adding, values*1000
        pos_sp_edges *= 1000

        #we substract the neg_sp_edges to make the values positive
        sp_edges = (pos_sp_edges - neg_sp_edges).coalesce()  # size([3783, 3783, 137]), nnz=31792, values{1000,2000,1001, 2001}

        #separating negs and positive edges per edge/timestamp
        vals = sp_edges._values()   #1000 or 1001
        neg_vals = vals%1000        #([31792]), values {0,1}, negative edges val 1 
        pos_vals = torch.div(vals, 1000, rounding_mode='trunc') #pos_vals = vals//1000, ([31792]), values {0,1,2}, positive edges val 1 or 2

        #We add the negative and positive scores and do majority voting
        vals = pos_vals - neg_vals   #([31792]), values{1,2,-1}

        #creating labels new_vals -> the label of the edges
        new_vals = torch.zeros(vals.size(0),dtype=torch.long)
        new_vals[vals>0] = 1   #postive edges to label 1        
        new_vals[vals<=0] = 0  #negative edges to label 0
        indices_labels = torch.cat([sp_edges._indices().t(),new_vals.view(-1,1)],dim=1) #([31792, 4]), cols(source, target, time, label_edge)

        #the weight of the edges (vals), is simply the number of edges between two entities at each time_step
        vals = pos_vals + neg_vals  #([31792]), vals{1,2}, no matter postive edges or negative edges 

        self.edges = {'idx': indices_labels, 'vals': vals} #a dict
        self.num_nodes = num_nodes
        self.num_classes = 2


    def cluster_negs_and_positives(self,ratings):
        pos_indices = ratings > 0
        neg_indices = ratings <= 0
        ratings[pos_indices] = 1
        ratings[neg_indices] = -1
        return ratings

    def get_num_nodes(self,edges):
        all_ids = edges[:,[self.ecols.FromNodeId,self.ecols.ToNodeId]]
        num_nodes = all_ids.max() + 1
        return num_nodes  #tensor 7605

    def load_edges(self,bitcoin_args):
        file = os.path.join(bitcoin_args.folder,bitcoin_args.edges_file)
        with open(file) as f:
            lines = f.read().splitlines()
        edges = [[float(r) for r in row.split(',')] for row in lines]
        edges = torch.tensor(edges,dtype = torch.long)
        return edges  #(24186, 4), kind of list of list 

    def make_contigous_node_ids(self,edges):
        new_edges = edges[:,[self.ecols.FromNodeId,self.ecols.ToNodeId]]
        _, new_edges = new_edges.unique(return_inverse=True)
        edges[:,[self.ecols.FromNodeId,self.ecols.ToNodeId]] = new_edges
        return edges #([24186, 4])

    def aggregate_by_time(self,time_vector,time_win_aggr):
        time_vector = time_vector - time_vector.min() 
        time_vector = torch.div(time_vector, time_win_aggr, rounding_mode='trunc') #time_vector//time_win_aggr
        return time_vector
