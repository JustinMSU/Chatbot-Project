import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import torch.autograd as autograd
from torch.nn import init
import torch.nn.utils.rnn 
from torch.nn.utils.rnn import pad_sequence
import datetime
import operator
import codecs
from datasets import load_dataset
import unicodedata
import string
import re
import random
import os
import itertools
import warnings
warnings.filterwarnings('ignore')
np.random.seed(0)

def shuffle_list(list):
    random.shuffle(list)

def load_id(sentence, word_to_id):
    sentence_ids = []

    max_sentence_len = 160
    
    sentence_words = sentence.split()
    if len(sentence_words) > max_sentence_len:
        sentence_words = sentence_words[:max_sentence_len]
    for word in sentence_words:
        if word in word_to_id:
            sentence_ids.append(word_to_id[word])
        else: 
            sentence_ids.append(0) #UNK

    return sentence_ids

def load_ids(pair, word_to_id):
    context_ids = []
    response_ids = []

    context_cell = pair[0]
    response_cell = pair[1]

    max_context_len = 160
    
    context_words = context_cell.split()
    if len(context_words) > max_context_len:
        context_words = context_words[:max_context_len]
    for word in context_words:
        if word in word_to_id:
            context_ids.append(word_to_id[word])
        else: 
            context_ids.append(0) #UNK
    
    response_words = response_cell.split()
    for word in response_words:
        if word in word_to_id:
            response_ids.append(word_to_id[word])
        else: 
            response_ids.append(0)

    return context_ids, response_ids

class Voc:
    def __init__(self):
        self.vocab = {}
        self.sentences = []
        self.word2id = {}
        self.id2vec = None
        
    def save(self):
        torch.save({
                'voc_dict': self.__dict__,
            }, os.path.join('Tutorials/dataset/models', 'save_voc.tar'))
    
    def load(self, filename):
        checkpoint = torch.load(filename)
        self.__dict__ = checkpoint['voc_dict']

class Encoder(nn.Module):

    def __init__(self, 
            emb_size, 
            hidden_size, 
            vocab_size,
            id_to_vec,
            p_dropout): 
    
            super(Encoder, self).__init__()
            self.emb_size = emb_size
            self.hidden_size = hidden_size
            self.vocab_size = vocab_size
            self.id_to_vec = id_to_vec
            self.p_dropout = p_dropout
            self.embedding = nn.Embedding(self.vocab_size, self.emb_size)
            self.lstm = nn.LSTM(self.emb_size, self.hidden_size)
            self.dropout_layer = nn.Dropout(self.p_dropout) 

            self.init_weights()
    def init_weights(self):
        init.uniform(self.lstm.weight_ih_l0, a = -0.01, b = 0.01)
        init.orthogonal(self.lstm.weight_hh_l0)
        self.lstm.weight_ih_l0.requires_grad = True
        self.lstm.weight_hh_l0.requires_grad = True
        
        embedding_weights = torch.FloatTensor(self.vocab_size, self.emb_size)
        embedding_weights = embedding_weights.to(device)
        for id, vec in self.id_to_vec.items():
            embedding_weights[id] = vec
        
        self.embedding.weight = nn.Parameter(embedding_weights, requires_grad = True)
            
    def forward(self, input_seq, input_lengths, hidden=None):
        embeddings = self.embedding(input_seq)
        embeddings = embeddings.to(device)
        packed = nn.utils.rnn.pack_padded_sequence(embeddings, input_lengths, enforce_sorted = False)
        
        _, (last_hidden, _) = self.lstm(packed, hidden)
        last_hidden = self.dropout_layer(last_hidden[-1])
        return last_hidden
    
    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        embeddings = embeddings.to(device)
        _, (last_hidden, _) = self.lstm(embeddings)
        last_hidden = self.dropout_layer(last_hidden[-1])

        return last_hidden

    
class DualEncoder(nn.Module):
     
    def __init__(self, encoder):
        super(DualEncoder, self).__init__()
        self.encoder = encoder
        self.hidden_size = self.encoder.hidden_size
        M = torch.FloatTensor(self.hidden_size, self.hidden_size)
        M = M.to(device)
        init.xavier_normal(M)
        self.M = nn.Parameter(M, requires_grad = True)

    def forward(self, input_seq, response_seq, input_lengths, response_lengths):
        context_last_hidden = self.encoder(input_seq, input_lengths) #dimensions: (batch_size x hidden_size)
        context_last_hidden = context_last_hidden.to(device)
        response_last_hidden = self.encoder(response_seq, response_lengths) #dimensions: (batch_size x hidden_size)
        response_last_hidden = response_last_hidden.to(device)
        
        context = context_last_hidden.mm(self.M) #dimensions: (batch_size x hidden_size)
        context = context.view(-1, 1, self.hidden_size) #dimensions: (batch_size x 1 x hidden_size)
        context = context.to(device)
        
        response = response_last_hidden.view(-1, self.hidden_size, 1) #dimensions: (batch_size x hidden_size x 1)
        response = response.to(device)
        
        score = torch.bmm(context, response).view(-1, 1) #dimensions: (batch_size x 1 x 1) and lastly --> (batch_size x 1)
        score = score.to(device)

        return score
    
    def forward(self, context_tensor, response_tensor):
        
        context_last_hidden = self.encoder(context_tensor) #dimensions: (batch_size x hidden_size)
        context_last_hidden = context_last_hidden.to(device)
        
        response_last_hidden = self.encoder(response_tensor) #dimensions: (batch_size x hidden_size)
        context = context_last_hidden.mm(self.M) #dimensions: (batch_size x hidden_size)
        context = context.view(-1, 1, self.hidden_size) #dimensions: (batch_size x 1 x hidden_size)
        context = context.to(device)
        
        response = response_last_hidden.view(-1, self.hidden_size, 1) #dimensions: (batch_size x hidden_size x 1)
        response = response.to(device)
        
        score = torch.bmm(context, response).view(-1, 1) #dimensions: (batch_size x 1 x 1) and lastly --> (batch_size x 1)
        score = score.to(device)

        return score
    
def creating_model(emb_dim, hidden_size, vocab_size, id_to_vec, p_dropout):

    encoder = Encoder(
            emb_size = emb_dim,
            hidden_size = hidden_size,
            vocab_size = vocab_size,
            id_to_vec = id_to_vec,
            p_dropout = p_dropout)

    dual_encoder = DualEncoder(encoder)
    
    return encoder, dual_encoder

def get_response_sample(context, num_responses):
    
    fileName = 'Tutorials/dataset/models/save_voc.tar'
    voc = Voc()
    voc.load(fileName)
    word_to_id = voc.word2id
    context_ids = load_id(context, word_to_id)
    i = 0
    best_response = ''
    best_score = 0
    
    # Set up encoder
    encoder, dual_encoder = creating_model(emb_dim = 50, hidden_size = 50, vocab_size = len(voc.vocab), id_to_vec = voc.id2vec, p_dropout = 0.85)
    encoder = encoder.to(device)
    dual_encoder = dual_encoder.to(device)
    dual_encoder.load_state_dict(torch.load('Tutorials/dataset/models/retrieval_encoder_dict.pt'))
    dual_encoder.eval();
    
    responses = random.sample(voc.sentences, num_responses)
    for response in responses:
        if not isinstance(response[0], str):
            continue
        response_ids = load_id(response[0], word_to_id)
        contextt = autograd.Variable(torch.LongTensor(context_ids).view(-1,1))
        responset = autograd.Variable(torch.LongTensor(response_ids).view(-1, 1))
        contextt, responset = contextt.to(device), responset.to(device)
        score = torch.sigmoid(dual_encoder(contextt, responset))
        if score > best_score:
            
            best_score = score
            best_response = response[0]
        i += 1
    return best_response