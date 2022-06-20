import os
import torch
import utils as u
import logger
import time
import pandas as pd
import numpy as np
from torch.nn import functional as F

"""Main purpose of this file, define Training process: optimizer step, run_epoch, predict"""

class Trainer():
	def __init__(self,args, splitter, gcn, classifier, classifier2, comp_loss, dataset, num_classes):
		self.args = args
		self.splitter = splitter
		self.tasker = splitter.tasker
		self.gcn = gcn
		self.classifier = classifier
		self.classifier2 = classifier2
		self.comp_loss = comp_loss

		self.num_nodes = dataset.num_nodes
		self.data = dataset
		self.num_classes = num_classes
		self.logger = logger.Logger(args, self.num_classes)
		self.init_optimizers(args)

	def init_optimizers(self,args):
		#use Adam to optimizer all parameters
		params = self.gcn.parameters()
		self.gcn_opt = torch.optim.Adam(params, lr = args.learning_rate)
		params = self.classifier.parameters()
		self.classifier_opt = torch.optim.Adam(params, lr = args.learning_rate)
		params = self.classifier2.parameters()
		self.classifier_opt2 = torch.optim.Adam(params, lr = args.learning_rate)

	def train(self):
		self.tr_step = 0
		best_eval_valid = 0
		eval_valid = 0
		epochs_without_impr = 0

		for e in range(self.args.num_epochs):
			eval_train, nodes_embs = self.run_epoch(self.splitter.train, e, 'TRAIN', grad = True)
			if len(self.splitter.val)>0 and e>self.args.eval_after_epochs:
				eval_valid, _ = self.run_epoch(self.splitter.val, e, 'VALID', grad = True)
				if eval_valid>best_eval_valid:
					best_eval_valid = eval_valid
					epochs_without_impr = 0
					print ('### w'+str(self.args.rank)+') ep '+str(e)+' - Best valid measure:'+str(eval_valid))
				else:
					epochs_without_impr+=1
					if epochs_without_impr>self.args.early_stop_patience:
						print ('### w'+str(self.args.rank)+') ep '+str(e)+' - Early stop.')
						break
			if len(self.splitter.test)>0 and eval_valid==best_eval_valid and e>self.args.eval_after_epochs:
				eval_test, _ = self.run_epoch(self.splitter.test, e, 'TEST', grad = True)


	def run_epoch(self, split, epoch, set_name, grad):
		t0 = time.time()      #return current time 
		log_interval=999
		if set_name=='TEST':
			log_interval=1

		self.logger.log_epoch_start(epoch, len(split), set_name, minibatch_log_interval=log_interval)
		torch.set_grad_enabled(grad) #calculate gradient 
		rand_tensor = torch.rand((3783, 50))             #3783 nodes, 50 from gcn parameters layer-1-feats, use a stable one for one epoch
		for s in split:              #split is the output of task.get_sample
			s = self.prepare_sample(s, set_name)
			predictions, nodes_embs = self.predict(s.hist_adj_list,         
												   s.hist_ndFeats_list,
												   s.label_sp['idx'],
												   s.node_mask_list,
												   rand_tensor)

			loss = self.comp_loss(predictions,s.label_sp['vals']) #predictions 2 cols [source, targtet] size([num_edges, 2]); label_sp['vals'] a row ([num_edges])

			if set_name in ['TEST', 'VALID'] and self.args.task == 'link_pred':
				self.logger.log_minibatch(predictions, s.label_sp['vals'], loss.detach(), adj = s.label_sp['idx'])
			else:
				self.logger.log_minibatch(predictions, s.label_sp['vals'], loss.detach())
			if grad:
				self.optim_step(loss)

		torch.set_grad_enabled(True)
		eval_measure = self.logger.log_epoch_done()
		return eval_measure, nodes_embs #nodes_embs, size([3783, F]), F gcn parameters: layer_1_feats


	def predict(self,hist_adj_list,hist_ndFeats_list,node_indices,mask_list,random_tensor):
		nodes_embs = self.gcn(hist_adj_list,
							  hist_ndFeats_list,
							  mask_list)
 
		#Ahat = hist_adj_list[-1]                  #skip-GCN,, just generate a random matrix reduce the dimension of A hat              		
        #nodes_embs =  Ahat @ random_tensor            

		predict_batch_size = 100000
		gather_predictions=[]
		#node_indices([[source], [target]]),size(2, num_edges]), 2 row)
		for i in range(1 +(node_indices.size(1)//predict_batch_size)):
			cls_input = self.gather_node_embs(nodes_embs, node_indices[:, i*predict_batch_size:(i+1)*predict_batch_size])
			predictions = self.classifier(cls_input)
			gather_predictions.append(predictions)
		gather_predictions=torch.cat(gather_predictions, dim=0) 
		#gather_predicitons(size([num_edges, 2]), 2 cols 
		return gather_predictions, nodes_embs

	def gather_node_embs(self,nodes_embs,node_indices):
		cls_input = []
		for node_set in node_indices:
			cls_input.append(nodes_embs[node_set])
		return torch.cat(cls_input,dim = 1)


	def optim_step(self,loss):
		#clear gradients for next train
		self.gcn_opt.zero_grad()          
		self.classifier_opt.zero_grad()
		self.classifier_opt2.zero_grad()

		loss.backward()  #back propagation, compute gradients
		
		#perform gradient clipping, to mitigate gradients explosure 
		torch.nn.utils.clip_grad_norm_(self.gcn.parameters(), 1)
		torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), 1)
		torch.nn.utils.clip_grad_norm_(self.classifier2.parameters(), 1)
		
		#apply gradients, update parameters, gradient descent  
		self.gcn_opt.step()
		self.classifier_opt.step()
		self.classifier_opt2.step()

	def prepare_sample(self,sample, set_name):
		sample = u.Namespace(sample)
		for i,adj in enumerate(sample.hist_adj_list):
			adj = u.sparse_prepare_tensor(adj,torch_size = [self.num_nodes])    # torch sparse tensor, edge index 2 * N_e, normalized adj values, 1 * N_e
			sample.hist_adj_list[i] = adj.to(self.args.device)

			nodes = self.tasker.prepare_node_feats(sample.hist_ndFeats_list[i]) # node features, one-hot degree feature for sbm

			sample.hist_ndFeats_list[i] = nodes.to(self.args.device)
			node_mask = sample.node_mask_list[i]
			sample.node_mask_list[i] = node_mask.to(self.args.device).t() #transposed to have same dimensions as scorer

		label_sp = self.ignore_batch_dim(sample.label_sp)  # label of all edges, 1 for existing / 0 for non-existing in link prediction
		label_sp['idx'] = label_sp['idx'].to(self.args.device).t()   
		label_sp['vals'] = label_sp['vals'].type(torch.long).to(self.args.device)
		sample.label_sp = label_sp
		return sample

	def ignore_batch_dim(self,adj):
		if self.args.task in ["link_pred", "edge_cls"]:
			adj['idx'] = adj['idx'][0]
		adj['vals'] = adj['vals'][0]
		return adj

