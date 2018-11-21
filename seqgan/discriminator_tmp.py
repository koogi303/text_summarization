# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 10:19:48 2018

@author: HQ
"""

import math
import tensorflow as tf

## A CNN discriminator model for text classification
vocab_size = 23198
max_len = 400
batch_size = 16
lr = 0.0001
emb_dim = 64
num_class = 2
dropout_prob = .5
l2_reg_lambda = 1

vocab_size = vocab_size
max_seq_len = max_len
batch_size = batch_size
lr = lr
embedding_dim = emb_dim
num_class = num_class
l2_reg_lambda = l2_reg_lambda
dropout_keep_prob = tf.get_variable(name = 'dropout_prob', shape = [], 
                               initializer=tf.constant_initializer(dropout_prob))

filter_sizes = [1, 2, 5, 10, 15, 20]
num_filters = [100, 200, 200, 100, 160, 160]

input_x = tf.placeholder('int32', [None, max_seq_len], name = 'input_X')
input_y = tf.placeholder('float32', [None, num_class], name = 'input_Y')

build_model()


def build_model():
    with tf.variable_scope('discriminator'):
        
        #tf.reset_default_graph()
        # -- embedding -- 
        with tf.variable_scope('word_embedding'):
            emb_W = tf.get_variable(name = 'W', 
                                shape = [vocab_size, embedding_dim],
                                initializer=tf.truncated_normal_initializer(stddev=6/math.sqrt(embedding_dim)))
            word_emb = tf.nn.embedding_lookup(params = emb_W, ids = input_x)
            word_emb_expand = tf.expand_dims(word_emb, axis=-1)
    
        pooled_output = []
        
        #filter_size = filter_sizes[0]
        #num_filter = num_filters[0]
        
        for filter_size, num_filter in zip(filter_sizes, num_filters):
            with tf.variable_scope('conv_maxpool_%s' % filter_size):
                filter_shape = [filter_size, embedding_dim, 1, num_filter]
                conv_W = tf.get_variable(name = 'conv_W', shape = filter_shape,
                                    initializer=tf.truncated_normal_initializer(stddev=.1))
                conv_b = tf.get_variable(name = 'conv_b', shape=[num_filter],
                                    initializer=tf.constant_initializer(0.1))
                conv = tf.nn.conv2d(input = word_emb_expand,
                                    filter = conv_W,
                                    strides=[1, 1, 1, 1],
                                    padding='VALID',
                                    name = 'conv')
                conv_bias = tf.nn.bias_add(value = conv, bias = conv_b, name ='conv_bias')
                h = tf.nn.relu(conv_bias, name = 'relu')
                
                pooled = tf.nn.max_pool(value = h,
                                        ksize=[1, max_seq_len - filter_size + 1, 1, 1],
                                        strides = [1, 1, 1, 1],
                                        padding = 'VALID',
                                        name = 'max_pooling')
                pooled_output.append(pooled)
        
        total_num_filters = sum(num_filters)
        h_pool = tf.concat(values = pooled_output, axis = 3)
        h_pool_flat = tf.reshape(h_pool, [-1, total_num_filters])
        
        with tf.name_scope('highway'):
            h_highway = _highway(input_ = h_pool_flat, size = h_pool_flat.get_shape()[1],
                                 num_layers=1, bias=0.)
            
        with tf.name_scope('dropout'):
            h_drop = tf.nn.dropout(x = h_highway, keep_prob=dropout_keep_prob)
            
        l2_loss = tf.constant(0.0)
        
        with tf.name_scope('output'):
            W = tf.get_variable(name = 'W', shape = [total_num_filters, num_class],
                                initializer=tf.truncated_normal_initializer(stddev=.1))
            b = tf.get_variable(name = 'b', shape = [num_class],
                                initializer=tf.constant_initializer(0.1))
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            scores = tf.nn.xw_plus_b(h_drop, W, b, name ='scores')
            ypred_for_auc = tf.nn.softmax(scores)
            predictions = tf.argmax(scores, 1, name = 'predictions')
            
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits = scores, labels = input_y)
            loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
        
    optimizer = tf.train.AdamOptimizer(lr)
    params = [param for param in  tf.trainable_variables() if 'discriminator' in param.name]
    gradients = optimizer.compute_gradients(loss = loss, var_list = params, aggregation_method=2)
    train_op = optimizer.apply_gradients(gradients)
    
    
def _highway(input_, size, num_layers = 1, bias = -2.0, scope = 'Highway'):
    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = tf.nn.relu(_linear(input_, size, scope = 'highway_lin_%d' % idx))
            t = tf.sigmoid(_linear(input_, size, scope = 'highway_gate_%d' % idx) + bias)
            output = t * g + (1. - t) * input_
            input_ = output
    return output

def _linear(input_, output_size, scope = None):
    shape = input_.get_shape().as_list()
    input_size = shape[1]
    
    with tf.variable_scope(scope or 'SimpleLinear'):
        matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
        bias_term = tf.get_variable("Bias", [output_size], dtype = input_.dtype)
    
    return tf.matmul(input_, tf.transpose(matrix)) + bias_term            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            