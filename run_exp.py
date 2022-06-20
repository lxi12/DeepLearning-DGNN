# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import utils as u
import torch
import torch.distributed as dist
import numpy as np
import time
import random

#datasets
import data as bc
 
#taskers
import link_pred_tasker as lpt
 
#models
import models as mls
import Analysis as ana
import trainer as tr
import logger


def build_random_hyper_params(args):
	"""build learning_rate, num_hist_steps, some gcn_parameters """ 
	if args.model == 'all':
		model_types = ['gcn', 'egcn_o']
		args.model=model_types[args.rank]

	args.learning_rate =u.random_param_value(args.learning_rate, args.learning_rate_min, args.learning_rate_max, type='logscale')

	args.num_hist_steps = u.random_param_value(args.num_hist_steps, args.num_hist_steps_min, args.num_hist_steps_max, type='int')

	args.gcn_parameters['feats_per_node'] =u.random_param_value(args.gcn_parameters['feats_per_node'], args.gcn_parameters['feats_per_node_min'], args.gcn_parameters['feats_per_node_max'], type='int')
	args.gcn_parameters['layer_1_feats'] =u.random_param_value(args.gcn_parameters['layer_1_feats'], args.gcn_parameters['layer_1_feats_min'], args.gcn_parameters['layer_1_feats_max'], type='int')
	if args.gcn_parameters['layer_2_feats_same_as_l1'] or args.gcn_parameters['layer_2_feats_same_as_l1'].lower()=='true':
		args.gcn_parameters['layer_2_feats'] = args.gcn_parameters['layer_1_feats']
	else:
		args.gcn_parameters['layer_2_feats'] =u.random_param_value(args.gcn_parameters['layer_2_feats'], args.gcn_parameters['layer_1_feats_min'], args.gcn_parameters['layer_1_feats_max'], type='int')
	args.gcn_parameters['lstm_l1_feats'] =u.random_param_value(args.gcn_parameters['lstm_l1_feats'], args.gcn_parameters['lstm_l1_feats_min'], args.gcn_parameters['lstm_l1_feats_max'], type='int')
	if args.gcn_parameters['lstm_l2_feats_same_as_l1'] or args.gcn_parameters['lstm_l2_feats_same_as_l1'].lower()=='true':
		args.gcn_parameters['lstm_l2_feats'] = args.gcn_parameters['lstm_l1_feats']
	else:
		args.gcn_parameters['lstm_l2_feats'] =u.random_param_value(args.gcn_parameters['lstm_l2_feats'], args.gcn_parameters['lstm_l1_feats_min'], args.gcn_parameters['lstm_l1_feats_max'], type='int')
	args.gcn_parameters['cls_feats']=u.random_param_value(args.gcn_parameters['cls_feats'], args.gcn_parameters['cls_feats_min'], args.gcn_parameters['cls_feats_max'], type='int')
	return args

def build_dataset(args):
	if args.data == 'bitcoinalpha':
		args.bitcoin_args = args.bitcoinalpha_args
		return bc.bitcoin_dataset(args)
	else:
		raise NotImplementedError('Whoops, no valid dataset found')
	 
def build_tasker(args,dataset):
	if args.task == 'link_pred':
		return lpt.Link_Pred_Tasker(args,dataset)
	else:
		raise NotImplementedError('Only doing link prediciton task')

	 
def build_gcn(args,tasker):
	gcn_args = u.Namespace(args.gcn_parameters)
	gcn_args.feats_per_node = tasker.feats_per_node
	if args.model == 'gcn':
		return mls.Sp_GCN(gcn_args,activation = torch.nn.RReLU()).to(args.device)
	else:
		assert args.num_hist_steps > 0, 'more than one step is necessary to train LSTM'
		if args.model == 'egcn_o':
			return mls.EGCN_O(gcn_args,activation = torch.nn.RReLU()).to(args.device)
		else:
			raise NotImplementedError('other models not implemented yet in this project')

def build_classifier(args,tasker):
	mult = 2
	if 'egcn_o' in args.model:
		in_feats = args.gcn_parameters['lstm_l2_feats'] * mult
	else:
		in_feats = args.gcn_parameters['layer_2_feats'] * mult

	return mls.Classifier(args,in_features = in_feats, out_features = tasker.num_classes).to(args.device)


if __name__ == '__main__':
	parser = u.create_parser()
	args = u.parse_args(parser)

	global rank, wsize, use_cuda
	args.use_cuda = (torch.cuda.is_available() and args.use_cuda)
	args.device='cpu'
	if args.use_cuda:
		args.device='cuda'
	print ("use CUDA:", args.use_cuda, "- device:", args.device)
	try:
		dist.init_process_group(backend='mpi') #, world_size=4
		rank = dist.get_rank()                 #,rank of each process, so they will know whether it is the master of a worker
		wsize = dist.get_world_size()          #, the total number of processes, so that the master knows how many workers to wait for 
		print('Hello from process {} (out of {})'.format(dist.get_rank(), dist.get_world_size()))
		if args.use_cuda:
			torch.cuda.set_device(rank )  # are we sure of the rank+1???
			print('using the device {}'.format(torch.cuda.current_device()))
	except:
		rank = 0
		wsize = 1
		print(('MPI backend not preset. Set process rank to {} (out of {})'.format(rank, wsize)))

	if args.seed is None and args.seed!='None':
		seed = 123+rank
	else:
		seed=args.seed

	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	args.seed=seed
	args.rank=rank
	args.wsize=wsize

	# Assign the requested random hyper parameters
	args = build_random_hyper_params(args)

	#build the dataset
	dataset = build_dataset(args)
	#build the tasker
	tasker = build_tasker(args,dataset)
	#build the splitter
	splitter = ana.splitter(args,tasker)
	#build the models
	gcn = build_gcn(args, tasker)

	classifier = build_classifier(args,tasker)
	classifier2 = build_classifier(args,tasker)

	cross_entropy = ana.Cross_Entropy(args,dataset).to(args.device)

	trainer = tr.Trainer(args,
						 splitter = splitter,
						 gcn = gcn,
						 classifier = classifier,
						 classifier2 = classifier2, 
						 comp_loss = cross_entropy,
						 dataset = dataset,
						 num_classes = tasker.num_classes)

	trainer.train()
