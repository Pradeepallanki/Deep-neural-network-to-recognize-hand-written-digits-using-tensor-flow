#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 22:19:25 2018

@author: pradi
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from PIL import Image


mnist=input_data.read_data_sets("/tmp/data/", one_hot=True)

layer1=500
layer2=500
layer3=500

classes=10

batch_size=100

x=tf.placeholder('float',[None,784])
y=tf.placeholder('float')

def neural_model(data):
    h_layer1={'weights':tf.Variable(tf.random_normal([784, layer1])),
              'biases': tf.Variable(tf.random_normal([layer1])) }
    
    h_layer2={'weights':tf.Variable(tf.random_normal([layer1, layer2])),
              'biases': tf.Variable(tf.random_normal([layer2])) }
    
    h_layer3={'weights':tf.Variable(tf.random_normal([layer2, layer3])),
              'biases': tf.Variable(tf.random_normal([layer3])) }
    
    output_layer={'weights':tf.Variable(tf.random_normal([layer3, classes])),
              'biases': tf.Variable(tf.random_normal([classes])) }
    
    
    l1=tf.add(tf.matmul(data,h_layer1['weights']), h_layer1['biases']) 
    l1=tf.nn.relu(l1)
    
    l2=tf.add(tf.matmul(l1,h_layer2['weights']), h_layer2['biases'])
    l2=tf.nn.relu(l2)
    
    l3=tf.add(tf.matmul(l2,h_layer3['weights']), h_layer3['biases'])
    l3=tf.nn.relu(l3)
    
    output=tf.matmul(l3, output_layer['weights'])+output_layer['biases']
    
    return output

def train_network(x):
    prediction=neural_model(x)
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    optimizer= tf.train.AdamOptimizer().minimize(cost)   
    hm_epochs =10
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        
        for epoch in range(hm_epochs):
            epoch_loss=0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x,epoch_y = mnist.train.next_batch(batch_size)
                _,c= sess.run([optimizer,cost], feed_dict={x:epoch_x, y:epoch_y})
                epoch_loss+=c
                print('Epoch', epoch, 'completed out of', hm_epochs,'loss:',epoch_loss)
                
            correct=tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
            print(correct)
            accuracy=tf.reduce_mean(tf.cast(correct,'float'))
            print('Accuracy:', accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))
            
            img=np.invert(Image.open('testimage.png').convert('L')).ravel()
            prediction1=sess.run(tf.argmax(prediction,1),feed_dict={x:[img]})

            print("Prediction for test image:",np.squeeze(prediction1))   

    
train_network(x)


    
    
    
