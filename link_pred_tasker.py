import torch
import utils as u

"""Main purpose of this file, defile link prediciton task class"""

class Link_Pred_Tasker():
	'''
	Creates a tasker object which computes the required inputs for training on a link prediction
	task. It receives a dataset object which should have two attributes: nodes_feats and edges, this
	makes the tasker independent of the dataset being used (as long as mentioned attributes have the same
	structure).

	Based on the dataset it implements the get_sample function required by edge_cls_trainer.
	This is a dictionary with:
		- time_step: the time_step of the prediction
		- hist_adj_list: the input adjacency matrices until t, each element of the list 
						 is a sparse tensor with the current edges. For link_pred they're
						 unweighted. In the model, it has used num_hist_steps for prediction
		- nodes_feats_list: the input nodes for the GCN models, each element of the list is a tensor
						  two dimmensions: node_idx and node_feats
		- label_adj: a sparse representation of the target edges. A dict with two keys: idx: M by 2 
					 matrix with the indices of the nodes conforming each edge, vals: 1 if the node exists
					 , 0 if it doesn't

	There's a test difference in the behavior, on test (or development), the number of sampled non existing 
	edges should be higher.
	'''
	def __init__(self,args,dataset):
		self.data = dataset
		#max_time for link pred should be one before
		self.max_time = dataset.max_time - 1        #135
		self.args = args
		self.num_classes = 2
		if not args.use_1_hot_node_feats:
			self.feats_per_node = dataset.feats_per_node

		self.get_node_feats = self.build_get_node_feats(args,dataset)
		self.prepare_node_feats = self.build_prepare_node_feats(args,dataset)

	def build_prepare_node_feats(self,args,dataset):
		if args.use_1_hot_node_feats:
			def prepare_node_feats(node_feats):
				return u.sparse_prepare_tensor(node_feats,
											   torch_size= [dataset.num_nodes, self.feats_per_node])
		else:
			prepare_node_feats = self.data.prepare_node_feats

		return prepare_node_feats


	def build_get_node_feats(self,args,dataset):
		if args.use_1_hot_node_feats:
			max_deg,_ = u.get_max_degs(args,dataset)
			self.feats_per_node = max_deg             # max_deg for node feature
			def get_node_feats(adj):
				return u.get_1_hot_deg_feats(adj, max_deg, dataset.num_nodes)
		else:
			def get_node_feats(adj):
				return dataset.nodes_feats

		return get_node_feats
	
	
	def get_sample(self,idx,test):
		hist_adj_list = []
		hist_ndFeats_list = []
		hist_mask_list = []
		existing_nodes = []
		for i in range(idx - self.args.num_hist_steps, idx+1):

			cur_adj = u.get_sp_adj(edges = self.data.edges,   #current adjacency matrix 
								   time = i,
								   weighted = True,
								   time_window = self.args.adj_mat_time_window)

			if self.args.smart_neg_sampling:
				existing_nodes.append(cur_adj['idx'].unique())
			else:
				existing_nodes = None

			node_mask = torch.zeros(self.data.num_nodes) - float("Inf")
			non_zero = cur_adj['idx'].unique()
			node_mask[non_zero] = 0
			
			#node_feats is get_1_hot_deg_feats
			node_feats = self.get_node_feats(cur_adj) # a dict, {'idx': [source 0->3782, degree 0->66] size([3783,67]), 'vals': [3783 of 1]}
			cur_adj = u.normalize_adj(adj = cur_adj, num_nodes = self.data.num_nodes) 
			#cur_adj, a dict, {'idx': [source 0->3782, target 0->3782] size([num_edges,2]), 'vals': [num_edges of normalize adj values]}

			hist_adj_list.append(cur_adj) 
			hist_ndFeats_list.append(node_feats)
			hist_mask_list.append(node_mask)

		# Get labels for outer loop, training on all the edges in the time_window 
		label_adj = u.get_sp_adj(edges = self.data.edges,     # a dict,{'idx': [source, target], 'vals':[1 or 0]}
								 time = idx+1,                # idx is an edge list of the edges we test on 
								 weighted = False,
								 time_window =  self.args.adj_mat_time_window)
		if test:
			neg_mult = self.args.negative_mult_test
		else:
			neg_mult = self.args.negative_mult_training
			
		if self.args.smart_neg_sampling:
			existing_nodes = torch.cat(existing_nodes)

		non_exisiting_adj = u.get_non_existing_edges(adj = label_adj,
													     number = label_adj['vals'].size(0) * neg_mult,
													     tot_nodes = self.data.num_nodes,
													     smart_sampling = self.args.smart_neg_sampling,
													     existing_nodes = existing_nodes)
		
		label_adj['idx'] = torch.cat([label_adj['idx'].long(),non_exisiting_adj['idx'].long()])
		label_adj['vals'] = torch.cat([label_adj['vals'],non_exisiting_adj['vals']])
		return {'idx': idx,                            # an interger tensor, timestep 
				'hist_adj_list': hist_adj_list,        #a list of (num_hist_steps+1) dicts, dict{'idx': tensor([source, target], 'vals': tensor([normalize adjacency value])
				'hist_ndFeats_list': hist_ndFeats_list,#a list of (num_hist_steps+1) dicts, dict{'idx': tensor([node index0->3782, deg of node0->66], 'vals': tensor([3783 of 1])
				'label_sp': label_adj,                 #a dict, {'idx': tensor([source, target], 'vals': tensor([1 exists, 0 No])
				'node_mask_list': hist_mask_list}      #len (num_hist_steps+1),a list of tensor, 0 or -inf, 
