# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 20:41:01 2018

@author: Ankan
"""

from PIL import Image
import numpy as np
import pandas as pd
import os
from os.path import isfile, join
import tensorflow as tf

##### PREPARING THE DATA #####

def numpy_fillna(data):
    lens = np.array([len(i) for i in data])
    c = 0
    max_len = max(lens)
    for i in lens: 
        npad = max_len - i 
        data[c] = np.pad(data[c], pad_width=npad, mode='constant', constant_values=0)[npad:]
        c = c +1
    data = np.array(data)
    return data

def build_data(dir):
    path = "DESTINATION_PATH\\%s"%(dir)
    dir_name = [name for name in os.listdir(path)]
    tempData = []
    y = []
    
    for directory in dir_name:
        my_path = "DESTINATION_PATH\\%s\\%s"%(dir,directory)
        files = [f for f in os.listdir(my_path) if isfile(join(my_path, f))]
        
        for ff in files:
            array = np.array(Image.open("DESTINATION_PATH\\%s\\%s\\%s"%(dir,directory,ff)).convert("1"))
            b = array.ravel()
            b = 1*b
            b = b.tolist()
            tempData.append(b)
            class_name = directory
            if len(class_name) > 1:
                class_name = class_name[0]
            classarr = [0]*6
            classarr[ord(class_name) - ord('A')]=1
            y.append(classarr)
    
    x = np.array(tempData)      
    req_x = numpy_fillna(tempData)
    req_y_ = np.array(y)
    l_y = len(req_y_)
    return req_x, req_y_
    
x_train, y_train = build_data("Samsung-Marcel-Train")
x_test, y_test = build_data("Samsung-Marcel-Test")


##### DESIGNING THE NEURAL NETWORK #####

l1_nodes = 400
l2_nodes = 200
l3_nodes = 200
l4_nodes = 400
final_layer_nodes = 6

X = tf.placeholder(dtype=tf.float32, shape=[None, 6048])
Y_ = tf.placeholder(dtype=tf.float32)

w1 = tf.Variable(initial_value=tf.truncated_normal([6048,l1_nodes], stddev=0.1))
b1 = tf.Variable(initial_value=tf.zeros([l1_nodes]))
Y1 = tf.nn.relu(tf.matmul(X,w1)+b1)

w2 = tf.Variable(initial_value=tf.truncated_normal([l1_nodes,l2_nodes], stddev=0.1))
b2 = tf.Variable(tf.zeros([l2_nodes]))
Y2 = tf.nn.relu(tf.matmul(Y1,w2)+b2)

w3 = tf.Variable(initial_value=tf.truncated_normal([l2_nodes,l3_nodes], stddev=0.1))
b3 = tf.Variable(tf.zeros([l3_nodes]))
Y3 = tf.nn.relu(tf.matmul(Y2,w3)+b3)

w4 = tf.Variable(initial_value=tf.truncated_normal([l3_nodes,l4_nodes], stddev=0.1))
b4 = tf.Variable(tf.zeros([l4_nodes]))
Y4 = tf.nn.relu(tf.matmul(Y3,w4)+b4)

w5 = tf.Variable(initial_value=tf.truncated_normal([l4_nodes, final_layer_nodes], stddev=0.1))
b5 = tf.Variable(tf.zeros([final_layer_nodes]))
Y = tf.nn.softmax(tf.matmul(Y4,w5)+b5)

#define cost function and evaluation
cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))
is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000003)
train_step = optimizer.minimize(loss=cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

##### TRAINING AND TESTING #####

for i in range(1000):
    
    # associate data with placeholders
    train_data = {X: x_train, Y_: y_train}

    # train model
    sess.run(train_step, feed_dict=train_data)
    # capture accuracy and loss metrics
    train_a, train_c = sess.run([accuracy, cross_entropy], feed_dict=train_data)
    # measure performance on test data
    test_data = {X: x_test , Y_: y_test }
    test_a, test_c = sess.run([accuracy, cross_entropy], feed_dict=test_data)

    if i % 100 == 0:
        print("Train accuracy: {}, Test accuracy: {}".format(train_a,test_a))

