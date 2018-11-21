# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 15:55:06 2018

@author: HQ
"""

import os
import pickle
import random
import numpy as np
import pandas as pd

'''
sen_file = './data/sentence.txt'
sum_file = './data/summary.txt'
s_vocab = './data/token_summary.csv'
t_vocab = './data/token_text.csv'
'''

def build_mapping(vocab_file):
    
    vocab = pd.read_csv(vocab_file,encoding='CP949')
    vocab = list(vocab)
    
    idx2word = {}
    idx2word[0] = '<pad>'
    idx2word[1] = '<unk>'
    idx2word[2] = '<bos>'
    idx2word[3] = '<eos>'
    
    for i, w in enumerate(vocab):
        idx2word[i+4] = w
    
    word2idx = {w : i for i, w in idx2word.items()}
    print(">> built word to index (index to word) mapping")
    return word2idx, idx2word

          
def token2idx(text_file, word2idx):
    
    index_text = []
    unk_count = 0
    
    document = np.array(pd.read_table(text_file, encoding = 'CP949', header=-1))
    
    for doc in document:
        index_line = []
        index_line.append(word2idx['<bos>'])
        
        doc = ''.join(doc).split(" ")
        for word in doc:
            if word in word2idx:
                index_line.append(word2idx[word])
            else:
                index_line.append(word2idx['<unk>'])
                unk_count += 1
        
        index_line.append(word2idx['<eos>'])
        index_text.append(index_line)
    print(">> tokenized text converted into sequence of index, number of unk words: ", unk_count)
    return index_text

#word2idx, idx2word = build_mapping(vocab_file)
#convert_text = token2idx(text_file, word2idx)


    

    



        
    













    