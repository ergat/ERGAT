import torch
import os
import numpy as np
import math
import random



def read_entity_from_id(filename='./data/WN18RR/entity2id.txt'):
    entity2id = {}
    with open(filename, 'r') as f:
        for line in f:
            if len(line.strip().split()) > 1:
                entity, entity_id = line.strip().split(
                )[0].strip(), line.strip().split()[1].strip()
                entity2id[entity] = int(entity_id)
    return entity2id


def read_relation_from_id(filename='./data/WN18RR/relation2id.txt'):
    relation2id = {}
    with open(filename, 'r') as f:
        for line in f:
            if len(line.strip().split()) > 1:
                relation, relation_id = line.strip().split(
                )[0].strip(), line.strip().split()[1].strip()
                relation2id[relation] = int(relation_id)
    return relation2id


def init_embeddings(entity_file, relation_file):
    entity_emb, relation_emb = [], []

    with open(entity_file) as f:
        for line in f:
            entity_emb.append([float(val) for val in line.strip().split()])

    with open(relation_file) as f:
        for line in f:
            relation_emb.append([float(val) for val in line.strip().split()])

    return np.array(entity_emb, dtype=np.float32), np.array(relation_emb, dtype=np.float32)


def parse_line(line):
    line = line.strip().split()
    e1, relation, e2 = line[0].strip(), line[1].strip(), line[2].strip()
    return e1, relation, e2


def load_data(filename, noise_file, entity2id, relation2id, is_unweigted=False, directed=True):
    with open(filename) as f:
        lines = f.readlines()
        
    if noise_file != False:
        with open(noise_file) as f:
            noise_lines = f.readlines()
        lines = lines + noise_lines

    # this is list for relation triples
    triples_data = []

    # for sparse tensor, rows list contains corresponding row of sparse tensor, cols list contains corresponding
    # columnn of sparse tensor, data contains the type of relation
    # Adjacecny matrix of entities is undirected, as the source and tail entities should know, the relation
    # type they are connected with
    rows, cols, data = [], [], []
    unique_entities = set()
    for line in lines:
        e1, relation, e2 = parse_line(line)
        unique_entities.add(e1)
        unique_entities.add(e2)
        triples_data.append(
            (entity2id[e1], relation2id[relation], entity2id[e2]))
        if not directed:
                # Connecting source and tail entity
            rows.append(entity2id[e1])
            cols.append(entity2id[e2])
            if is_unweigted:
                data.append(1)
            else:
                data.append(relation2id[relation])

        # Connecting tail and source entity
        rows.append(entity2id[e2])
        cols.append(entity2id[e1])
        if is_unweigted:
            data.append(1)
        else:
            data.append(relation2id[relation])

    print("number of unique_entities ->", len(unique_entities))
    return triples_data, (rows, cols, data), list(unique_entities)


def build_data(path='./data/WN18RR/', use_noise = False, noice_percent=0.2, is_unweigted=False, directed=True):
    entity2id = read_entity_from_id(path + 'entity2id.txt')
    relation2id = read_relation_from_id(path + 'relation2id.txt')
    
    if use_noise != False:
        noise_file = 'noise/'+path.split('/')[-2]+'-'+str(int(noice_percent*100))+'%.txt'
        train_triples, train_adjacency_mat, unique_entities_train = load_data(os.path.join(
            path, 'train.txt'), os.path.join(path, noise_file), entity2id, relation2id, is_unweigted, directed)    

    # Adjacency matrix only required for training phase
    # Currenlty creating as unweighted, undirected
    else:
        train_triples, train_adjacency_mat, unique_entities_train = load_data(os.path.join(
            path, 'train.txt'), False, entity2id, relation2id, is_unweigted, directed)
    validation_triples, valid_adjacency_mat, unique_entities_validation = load_data(
        os.path.join(path, 'valid.txt'), False, entity2id, relation2id, is_unweigted, directed)
    test_triples, test_adjacency_mat, unique_entities_test = load_data(os.path.join(
        path, 'test.txt'), False, entity2id, relation2id, is_unweigted, directed)

    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}
    left_entity, right_entity = {}, {}

    with open(os.path.join(path, 'train.txt')) as f:
        lines = f.readlines()

    for line in lines:
        e1, relation, e2 = parse_line(line)

        # Count number of occurences for each (e1, relation)
        if relation2id[relation] not in left_entity:
            left_entity[relation2id[relation]] = {}
        if entity2id[e1] not in left_entity[relation2id[relation]]:
            left_entity[relation2id[relation]][entity2id[e1]] = 0
        left_entity[relation2id[relation]][entity2id[e1]] += 1

        # Count number of occurences for each (relation, e2)
        if relation2id[relation] not in right_entity:
            right_entity[relation2id[relation]] = {}
        if entity2id[e2] not in right_entity[relation2id[relation]]:
            right_entity[relation2id[relation]][entity2id[e2]] = 0
        right_entity[relation2id[relation]][entity2id[e2]] += 1

    left_entity_avg = {}
    for i in range(len(relation2id)):
        left_entity_avg[i] = sum(
            left_entity[i].values()) * 1.0 / len(left_entity[i])

    right_entity_avg = {}
    for i in range(len(relation2id)):
        right_entity_avg[i] = sum(
            right_entity[i].values()) * 1.0 / len(right_entity[i])

    headTailSelector = {}
    for i in range(len(relation2id)):
        headTailSelector[i] = 1000 * right_entity_avg[i] / \
            (right_entity_avg[i] + left_entity_avg[i])

    return (train_triples, train_adjacency_mat), (validation_triples, valid_adjacency_mat), (test_triples, test_adjacency_mat), \
        entity2id, relation2id, headTailSelector, unique_entities_train

def generate_noise_data(path='./data/WN18RR/',percent=0.1):
    filename=path+'train.txt'
    with open(filename) as f:
        lines = f.readlines()
    # this is list for relation triples
    triples_data = []
    e1re={}#distinct tuple for e1 and relation
    e2re={}
    #build entity-relation list
    for line in lines:
        e1, relation, e2 = parse_line(line)
        triples_data.append((e1, relation, e2))
        if relation not in e1re:
            e1re[relation]=[e1]
        elif e1 not in e1re[relation]:
            e1re[relation].append(e1)
        if relation not in e2re:
            e2re[relation]=[e2]
        elif e2 not in e2re[relation]:
            e2re[relation].append(e2)
    #generate noise data
    random.shuffle(triples_data)
    noise_triples=[]
    cnt=0
    for triple in triples_data[:round(len(triples_data)*percent)]:
        e1, relation, e2 = triple[0], triple[1], triple[2]
        if random.random() < 0.5:#randomly replace head or tail entity
            noise=e1re[relation][math.floor(random.random()*len(e1re[relation]))]# do not use random.choice(e1re[relation])!
            while noise==e1:
                noise=e1re[relation][math.floor(random.random()*len(e1re[relation]))]
            noise_triples.append((noise, relation, e2))
        else:
            noise=e1re[relation][math.floor(random.random()*len(e1re[relation]))]
            while noise==e2:
                noise=e1re[relation][math.floor(random.random()*len(e1re[relation]))]
            noise_triples.append((e1, relation, noise))
        cnt+=1
        print('Now: '+str(cnt)+' total: '+str(round(len(triples_data)*percent)))
        
    #save noise data
    if not os.path.exists(path+'noise/'):
        os.mkdir(path+'noise/')
    noise_file_name=path+'noise/'+path.split('/')[-2]+'-'+str(int(percent*100))+'%.txt'
    with open(noise_file_name,'w') as f:
        for line in noise_triples:
            f.write(line[0]+'\t'+line[1]+'\t'+line[2]+'\n')
        f.close()
