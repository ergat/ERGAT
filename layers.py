import numpy as np
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
from torch.autograd import Variable
import math
import random
import copy

CUDA = torch.cuda.is_available()


class ConvKB(nn.Module):
    def __init__(self, input_dim, input_seq_len, in_channels, out_channels, drop_prob, alpha_leaky):
        super().__init__()

        self.conv_layer = nn.Conv2d(
            in_channels, out_channels, (1, input_seq_len))  # kernel size -> 1*input_seq_length(i.e. 2)
        self.dropout = nn.Dropout(drop_prob)
        self.non_linearity = nn.ReLU()
        self.fc_layer = nn.Linear((input_dim) * out_channels, 1)

        nn.init.xavier_uniform_(self.fc_layer.weight, gain=1.414)
        nn.init.xavier_uniform_(self.conv_layer.weight, gain=1.414)

    def forward(self, conv_input):

        batch_size, length, dim = conv_input.size()
        # assuming inputs are of the form ->
        conv_input = conv_input.transpose(1, 2)
        # batch * length(which is 3 here -> entity,relation,entity) * dim
        # To make tensor of size 4, where second dim is for input channels
        conv_input = conv_input.unsqueeze(1)

        out_conv = self.dropout(
            self.non_linearity(self.conv_layer(conv_input)))

        input_fc = out_conv.squeeze(-1).view(batch_size, -1)
        output = self.fc_layer(input_fc)
        return output

#ctx is a context object that can be used to stash information for backward computation
class SpecialSpmmFunctionFinal(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, edge, edge_w, N, E, out_features, sum_type):
        # assert indices.requires_grad == False

        a = torch.sparse_coo_tensor(
            edge, edge_w, torch.Size([N, N, out_features]))
        if sum_type == 'out':
            b = torch.sparse.sum(a, dim=1)
            ctx.N = b.shape[0]
            ctx.outfeat = b.shape[1]
            ctx.E = E
            ctx.indices = a._indices()[0, :]
        elif sum_type == 'in':
            b = torch.sparse.sum(a, dim=0)
            ctx.N = b.shape[1]
            ctx.outfeat = b.shape[0]
            ctx.E = E
            ctx.indices = a._indices()[1, :]
        return b.to_dense()

    @staticmethod
    def backward(ctx, grad_output):
        grad_values = None
        if ctx.needs_input_grad[1]:
            edge_sources = ctx.indices

            if(CUDA):
                edge_sources = edge_sources.cuda()

            grad_values = grad_output[edge_sources]
            # grad_values = grad_values.view(ctx.E, ctx.outfeat)
            # print("Grad Outputs-> ", grad_output)
            # print("Grad values-> ", grad_values)
        return None, grad_values, None, None, None, None


class SpecialSpmmFinal(nn.Module):
    def forward(self, edge, edge_w, N, E, out_features, sum_type):
        return SpecialSpmmFunctionFinal.apply(edge, edge_w, N, E, out_features, sum_type)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, num_nodes, in_features, out_features, nrela_dim, dropout, alpha, gamma, eta, epsilon, d, concat=True):
#        super(SpGraphAttentionLayer, self).__init__()
        super().__init__()
        self.in_features = in_features 
        self.out_features = out_features
        self.num_nodes = num_nodes
        self.alpha = alpha
        self.concat = concat
        self.nrela_dim = nrela_dim
        
        self.gamma = gamma
        self.eta = eta
        self.epsilon = epsilon
        self.d = d

        self.a = nn.Parameter(torch.zeros(
            size=(out_features, 2 * in_features + nrela_dim)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)
        self.a_2 = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_normal_(self.a_2.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm_final = SpecialSpmmFinal()


    def forward(self, input, edge, edge_embed, edge_list_nhop, edge_embed_nhop, confidence, entity_rank, Corpus_):
        N = input.size()[0]
        # Self-attention on the nodes - Shared attention mechanism
        if len(edge_list_nhop) != 0:
            edge = torch.cat((edge[:, :], edge_list_nhop[:, :]), dim=1)
            edge_embed = torch.cat(
                (edge_embed[:, :], edge_embed_nhop[:, :]), dim=0)

        edge_h = torch.cat(
            (input[edge[0, :], :], input[edge[1, :], :], edge_embed[:, :]), dim=1).t()
        # edge_h: (2*in_dim + nrela_dim) x E

        edge_m = self.a.mm(edge_h)
        # edge_m: D * E
        
        #Modified Absolute Attention Value: add LeakrReLU as activation function
        edge_m=self.leakyrelu(edge_m)
        # edge_m: leakyrelu(D * E)
        
        # [Please ignore]
        if confidence != False:
            #create noise
            noise_edge=copy.deepcopy(edge)
            for i in range(edge.size()[1]):
                if random.random() < 0.5:#randomly replace head or tail entity
                    noise_edge[0][i]=Corpus_.unique_entities_train[math.floor(random.random()*len(Corpus_.unique_entities_train))]
                else:
                    noise_edge[1][i]=Corpus_.unique_entities_train[math.floor(random.random()*len(Corpus_.unique_entities_train))]
            #generate noise enytity embedding
            noise_edge_h = torch.cat(
                (input[noise_edge[0, :], :], input[noise_edge[1, :], :], edge_embed[:, :]), dim=1).t()
            noise_edge_m = self.a.mm(noise_edge_h)                
            noise_edge_m = self.leakyrelu(noise_edge_m)
        
            quality = self.gamma + (edge_m - noise_edge_m).sum(0)
            #for quality <= 0
            q1 = torch.le(quality, 0).float() * self.eta * confidence
            #for quality > 0
            q2 = torch.gt(quality, 0).float() * (self.epsilon + confidence)
            
            confidence = q1 + q2
    
            # to be checked later
            powers = -self.leakyrelu(self.a_2.mm(edge_m).squeeze())
            #add confidence here
            edge_e = torch.exp(powers * confidence).unsqueeze(1)
            
        
        else:
            powers = -self.leakyrelu(self.a_2.mm(edge_m).squeeze())
            edge_e = torch.exp(powers).unsqueeze(1)
        
           
        assert not torch.isnan(edge_e).any()
        # edge_e: E
    
        e_rowsum = self.special_spmm_final(
            edge, edge_e, N, edge_e.shape[0], 1, 'in')
        e_rowsum[e_rowsum == 0.0] = 1e-12
    
        in_edge_sum = e_rowsum[edge[0]]
        relative_attention_value = edge_e.div(in_edge_sum)

            

        denominator_rowsum = self.special_spmm_final(
            edge, relative_attention_value, N, relative_attention_value.shape[0], 1, 'out')#denominator of aijk
        denominator_rowsum[denominator_rowsum == 0.0] = 1e-12
        
        denominator_edge_sum = denominator_rowsum[edge[0]]#denominator before summation 
        numerator_entity_rank = (relative_attention_value.squeeze() * entity_rank[edge[0]]).unsqueeze(1)
        numerator_entity_rank = numerator_entity_rank/denominator_edge_sum

        numerator_rowsum = self.special_spmm_final(
            edge, numerator_entity_rank, N, numerator_entity_rank.shape[0], 1, 'in')

        
        # renew entity rank
        entity_rank = (1-self.d) + self.d * numerator_rowsum.squeeze()

        
        if confidence != False:
            #propagation
            edge_prop = relative_attention_value.squeeze() * edge_m
            edge_prop = (entity_rank[edge[0]] * edge_prop).t()
    
            
            h_prime = self.special_spmm_final(
                edge, edge_prop, N, edge_prop.shape[0], self.out_features, 'in')
#            print(h_prime)
        else:
            edge_e = edge_e.squeeze(1)
    
            edge_e = self.dropout(edge_e)

    
            edge_w = (F.softmax(entity_rank, dim=0)[edge[0]] * (edge_e * edge_m)).t()
#            edge_w = (edge_e * edge_m).t()
    
            h_prime = self.special_spmm_final(
                edge, edge_w, N, edge_w.shape[0], self.out_features, 'in')
#            print(h_prime)
                
            assert not torch.isnan(h_prime).any()
            # h_prime: N x out
            h_prime = h_prime.div(e_rowsum)
            # h_prime: N x out
            


        assert not torch.isnan(h_prime).any()
        if self.concat:
            # if this layer is not last layer,
            if confidence != False:
                return F.elu(h_prime), entity_rank, confidence
            else:
                return F.elu(h_prime), entity_rank
        else:
            # if this layer is last layer,
            if confidence != False:
                return h_prime, entity_rank, confidence
            else:
                return h_prime, entity_rank

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
