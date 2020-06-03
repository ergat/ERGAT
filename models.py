import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
from layers import SpGraphAttentionLayer, ConvKB

CUDA = torch.cuda.is_available()


class SpGAT(nn.Module):
    def __init__(self, num_nodes, nfeat, nhid, relation_dim, dropout, alpha, gamma, eta, epsilon, d, nheads):
        """
            Sparse version of GAT
            nfeat -> Entity Input Embedding dimensions
            nhid  -> Entity Output Embedding dimensions
            relation_dim -> Relation Embedding dimensions
            num_nodes -> number of nodes in the Graph
            nheads -> Used for Multihead attention

        """
        super(SpGAT, self).__init__()
        self.dropout = dropout
        self.gamma = gamma
        self.dropout_layer = nn.Dropout(self.dropout)
        self.nheads = nheads
        self.attentions = [SpGraphAttentionLayer(num_nodes, nfeat,
                                                 nhid,
                                                 relation_dim,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 gamma=self.gamma,
                                                 eta=eta, 
                                                 epsilon=epsilon,
                                                 d = d,
                                                 concat=True)
                           for _ in range(nheads)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # W matrix to convert h_input to h_output dimension
        self.W = nn.Parameter(torch.zeros(size=(relation_dim, nheads * nhid)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414) 

        self.out_att = SpGraphAttentionLayer(num_nodes, nhid * nheads,
                                             nheads * nhid, nheads * nhid,
                                             dropout=dropout,
                                             alpha=alpha,
                                             gamma=gamma,
                                             eta=eta, 
                                             epsilon=epsilon,
                                             d=d,
                                             concat=False
                                             )

    def forward(self, Corpus_, batch_inputs, entity_embeddings, relation_embed,
                edge_list, edge_type, edge_embed, edge_list_nhop, edge_type_nhop, confidence, entity_rank):
        x = entity_embeddings
        if len(edge_type_nhop) != 0:
            edge_embed_nhop = relation_embed[
                edge_type_nhop[:, 0]] + relation_embed[edge_type_nhop[:, 1]]
        else: 
            edge_embed_nhop = edge_type_nhop

#attention forward
        if confidence != False:
            x = torch.cat([att(x, edge_list, edge_embed, edge_list_nhop, edge_embed_nhop, confidence, entity_rank, Corpus_)
                           for att in self.attentions], dim=1)
        else:
            h_prime_list = []
            entity_rank_list = []

            for att in self.attentions:
                h_prime, entity_rank = att(x, edge_list, edge_embed, edge_list_nhop, edge_embed_nhop, False, entity_rank, Corpus_)
                h_prime_list.append(h_prime)
                entity_rank_list.append(entity_rank)
            x = torch.cat([h_p for h_p in h_prime_list], dim=1)
            entity_rank = sum(entity_rank_list)/self.nheads
                                
        x = self.dropout_layer(x)

        out_relation_1 = relation_embed.mm(self.W)

        edge_embed = out_relation_1[edge_type]
        if len(edge_type_nhop) != 0:
            edge_embed_nhop = out_relation_1[
                edge_type_nhop[:, 0]] + out_relation_1[edge_type_nhop[:, 1]]

        if confidence != False:
            x = F.elu(self.out_att(x, edge_list, edge_embed,
                                   edge_list_nhop, edge_embed_nhop, confidence, entity_rank, Corpus_))
        else:
            x, entity_rank = self.out_att(x, edge_list, edge_embed,
                                   edge_list_nhop, edge_embed_nhop, False, entity_rank, Corpus_)
            x = F.elu(x)

        return x, out_relation_1, entity_rank


class SpKBGATModified(nn.Module):
    def __init__(self, initial_entity_emb, initial_relation_emb, entity_out_dim, relation_out_dim,
                 drop_GAT, alpha, gamma, eta, epsilon, d, nheads_GAT):
        '''Sparse version of KBGAT
        entity_in_dim -> Entity Input Embedding dimensions
        entity_out_dim  -> Entity Output Embedding dimensions, passed as a list
        num_relation -> number of unique relations
        relation_dim -> Relation Embedding dimensions
        num_nodes -> number of nodes in the Graph
        nheads_GAT -> Used for Multihead attention, passed as a list '''

        super().__init__()

        self.num_nodes = initial_entity_emb.shape[0] 
        self.entity_in_dim = initial_entity_emb.shape[1]
        self.entity_out_dim_1 = entity_out_dim[0]
        self.nheads_GAT_1 = nheads_GAT[0]
        self.entity_out_dim_2 = entity_out_dim[1]
        self.nheads_GAT_2 = nheads_GAT[1]

        # Properties of Relations
        self.num_relation = initial_relation_emb.shape[0]
        self.relation_dim = initial_relation_emb.shape[1]
        self.relation_out_dim_1 = relation_out_dim[0]

        self.drop_GAT = drop_GAT
        self.alpha = alpha 
        self.gamma = gamma
        self.eta = eta
        self.epsilon = epsilon
        self.d = d

        self.final_entity_embeddings = nn.Parameter(
            torch.randn(self.num_nodes, self.entity_out_dim_1 * self.nheads_GAT_1))

        self.final_relation_embeddings = nn.Parameter(
            torch.randn(self.num_relation, self.entity_out_dim_1 * self.nheads_GAT_1))

        self.final_entity_rank = nn.Parameter(
            torch.randn(self.num_nodes))

        self.entity_embeddings = nn.Parameter(initial_entity_emb)
        self.relation_embeddings = nn.Parameter(initial_relation_emb)

        self.sparse_gat_1 = SpGAT(self.num_nodes, self.entity_in_dim, self.entity_out_dim_1, self.relation_dim,
                                  self.drop_GAT, self.alpha, self.gamma, self.eta, self.epsilon, self.d, self.nheads_GAT_1)

        self.W_entities = nn.Parameter(torch.zeros(
            size=(self.entity_in_dim, self.entity_out_dim_1 * self.nheads_GAT_1)))
        nn.init.xavier_uniform_(self.W_entities.data, gain=1.414)

    def forward(self, Corpus_, adj, batch_inputs, train_indices_nhop, confidence, entity_rank):
        # getting edge list
        edge_list = adj[0]
        edge_type = adj[1]
        if len(train_indices_nhop) != 0:
            edge_list_nhop = torch.cat(
                (train_indices_nhop[:, 3].unsqueeze(-1), train_indices_nhop[:, 0].unsqueeze(-1)), dim=1).t()
            edge_type_nhop = torch.cat(
                [train_indices_nhop[:, 1].unsqueeze(-1), train_indices_nhop[:, 2].unsqueeze(-1)], dim=1)
        else:
            edge_list_nhop = train_indices_nhop
            edge_type_nhop = train_indices_nhop
        
        if confidence != False:
            confidence=nn.Parameter(confidence)
        entity_rank=nn.Parameter(entity_rank)

        if(CUDA):
            edge_list = edge_list.cuda()
            edge_type = edge_type.cuda()
            edge_list_nhop = edge_list_nhop.cuda()
            edge_type_nhop = edge_type_nhop.cuda()



        edge_embed = self.relation_embeddings[edge_type]



        self.entity_embeddings.data = F.normalize(
            self.entity_embeddings.data, p=2, dim=1).detach()

        # self.relation_embeddings.data = F.normalize(
        #     self.relation_embeddings.data, p=2, dim=1)
        
        if confidence != False:
            out_entity_1, out_relation_1, entity_rank = self.sparse_gat_1(
                Corpus_, batch_inputs, self.entity_embeddings, self.relation_embeddings,
                edge_list, edge_type, edge_embed, edge_list_nhop, edge_type_nhop, confidence, entity_rank)
        else:
            out_entity_1, out_relation_1, entity_rank = self.sparse_gat_1(
                Corpus_, batch_inputs, self.entity_embeddings, self.relation_embeddings,
                edge_list, edge_type, edge_embed, edge_list_nhop, edge_type_nhop, False, entity_rank)

        if(CUDA):
            mask_indices = torch.unique(batch_inputs[:, 2]).cuda()
            mask = torch.zeros(self.entity_embeddings.shape[0]).cuda()
        else:
            mask_indices = torch.unique(batch_inputs[:, 2]) 
            mask = torch.zeros(self.entity_embeddings.shape[0])           
        mask[mask_indices] = 1.0 


        entities_upgraded = self.entity_embeddings.mm(self.W_entities)
      
        out_entity_1 = entities_upgraded + \
            mask.unsqueeze(-1).expand_as(out_entity_1) * out_entity_1

        out_entity_1 = F.normalize(out_entity_1, p=2, dim=1)
        self.final_entity_embeddings.data = out_entity_1.data
        self.final_relation_embeddings.data = out_relation_1.data

        self.final_entity_rank.data = entity_rank.data

        return out_entity_1, out_relation_1

class SpKBGATConvOnly(nn.Module):
    def __init__(self, initial_entity_emb, initial_relation_emb, entity_out_dim, relation_out_dim,
                 drop_GAT, drop_conv, alpha, alpha_conv, nheads_GAT, conv_out_channels):
        '''Sparse version of KBGAT
        entity_in_dim -> Entity Input Embedding dimensions
        entity_out_dim  -> Entity Output Embedding dimensions, passed as a list
        num_relation -> number of unique relations
        relation_dim -> Relation Embedding dimensions
        num_nodes -> number of nodes in the Graph
        nheads_GAT -> Used for Multihead attention, passed as a list '''

        super().__init__()

        self.num_nodes = initial_entity_emb.shape[0]
        self.entity_in_dim = initial_entity_emb.shape[1]
        self.entity_out_dim_1 = entity_out_dim[0]
        self.nheads_GAT_1 = nheads_GAT[0]
        self.entity_out_dim_2 = entity_out_dim[1]
        self.nheads_GAT_2 = nheads_GAT[1]

        # Properties of Relations
        self.num_relation = initial_relation_emb.shape[0]
        self.relation_dim = initial_relation_emb.shape[1]
        self.relation_out_dim_1 = relation_out_dim[0]

        self.drop_GAT = drop_GAT
        self.drop_conv = drop_conv
        self.alpha = alpha 
        self.alpha_conv = alpha_conv
        self.conv_out_channels = conv_out_channels

        self.final_entity_embeddings = nn.Parameter(
            torch.randn(self.num_nodes, self.entity_out_dim_1 * self.nheads_GAT_1))

        self.final_relation_embeddings = nn.Parameter(
            torch.randn(self.num_relation, self.entity_out_dim_1 * self.nheads_GAT_1))

        self.final_entity_rank = nn.Parameter(
            torch.randn(self.num_nodes))

        self.convKB = ConvKB(self.entity_out_dim_1 * self.nheads_GAT_1, 3, 1,
                             self.conv_out_channels, self.drop_conv, self.alpha_conv)

    def forward(self, Corpus_, adj, batch_inputs):
        conv_input = torch.cat((self.final_entity_embeddings[batch_inputs[:, 0], :].unsqueeze(1), self.final_relation_embeddings[
            batch_inputs[:, 1]].unsqueeze(1), self.final_entity_embeddings[batch_inputs[:, 2], :].unsqueeze(1)), dim=1)
        out_conv = self.convKB(conv_input)
        return out_conv

    def batch_test(self, batch_inputs):
        conv_input = torch.cat((self.final_entity_embeddings[batch_inputs[:, 0], :].unsqueeze(1), self.final_relation_embeddings[
            batch_inputs[:, 1]].unsqueeze(1), self.final_entity_embeddings[batch_inputs[:, 2], :].unsqueeze(1)), dim=1)
        out_conv = self.convKB(conv_input)
        return out_conv
