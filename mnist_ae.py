# -*- coding: utf-8 -*-
"""
This program builds an Autoencoder using TensorFlow
MNIST dataset has been used.
Code based on tutorial from Big Data Univeristy.

Author: Ankit Bansal
Date: 03-30-2017
"""
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("mnist_data/", one_hot = True)

learning_rate = 0.01
batch_size = 256
no_examples = 10
input_dim = 28*28
no_epochs = 20

# Network Parameters
nodes_hidden_1 = 256 # 1st layer num features
nodes_hidden_2 = 128 # 2nd layer num features
nodes_input = input_dim # MNIST data input (img shape: 28*28)

# tf Graph input (only images)
X = tf.placeholder("float", [None, nodes_input])

# Weights and Biases for the encoder and decoder
"""
O                                   O
O        O                 O        O
O        O        O        O        O
O W_e_h1 O W_e_h2 O W_d_h1 O W_d_h2 O  
O        O        O        O        O
O        O                 O        O
O                                   O
"""

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([nodes_input, nodes_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([nodes_hidden_1, nodes_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([nodes_hidden_2, nodes_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([nodes_hidden_1, nodes_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([nodes_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([nodes_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([nodes_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([nodes_input])),
}

def encoder(x):
    print('Encoding...')
    encoder_layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x,weights['encoder_h1']), biases['encoder_b1']))
    encoder_layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(encoder_layer_1, weights['encoder_h2']), biases['encoder_b2']))
    print('Encoded!')
    return encoder_layer_2

def decoder(x):
    print('Decoding...')
    decoder_layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x,weights['decoder_h1']), biases['decoder_b1']))
    decoder_layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(decoder_layer_1, weights['decoder_h2']), biases['decoder_b2']))
    print('Decoded!')
    return decoder_layer_2   

def autoencoder(x):
    
    encoder_output = encoder(x)
    decoder_output = decoder(encoder_output)
    
    cost = tf.reduce_mean(tf.square(x - decoder_output))
    
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
    
    with tf.Session() as sess:
        
        tf.global_variables_initializer().run()
        
        for epoch in range(no_epochs):
            
            # Loop over all batches
            for i in range(int(mnist.train.num_examples/batch_size)):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x})
                
            print('############################')     
            print('Epoch: ', epoch+1, ' out of ', no_epochs)
            print('Cost: ', c)
        
        encode_decode = sess.run(decoder_output, feed_dict={x: mnist.test.images[:no_examples]})
        
        f, a = plt.subplots(2, 10, figsize=(10, 2))
        
        for i in range(no_examples):
            a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
            a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
            
autoencoder(X)
               
        
    
    
    
    
    
    