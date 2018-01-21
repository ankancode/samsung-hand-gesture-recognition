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
import math
import time

max_len = 0
ceil_val = 0
ceil_max = 0

##### PREPARING THE DATA #####

def setting_global(len_data):
    global ceil_val
    global ceil_max
    square_root = math.sqrt(len_data)
    ceil_val = math.ceil(square_root)
    ceil_max = ceil_val * ceil_val
    return
    
def numpy_fillna(data):
    global max_len
    global ceil_val
    global ceil_max
    lens = np.array([len(i) for i in data])
    c = 0
    max_len = max(lens)
    
    if max_len<= max(lens):
        max_len = max(lens)
        setting_global(max_len)
   
    for i in lens: 
        npad = ceil_max - i 
        #npad = max_len - i 
        data[c] = np.pad(data[c], pad_width=npad, mode='constant', constant_values=0)[npad:]
        c = c + 1
    data = np.array(data)
    print(ceil_val)
    print(ceil_max)
    return data

def build_data(dir):
    path = "C:\\Users\\Ankan\\Desktop\\samsung_hand_gesture\\%s"%(dir)
    dir_name = [name for name in os.listdir(path)]
    tempData = []
    y = []
    
    for directory in dir_name:
        my_path = "C:\\Users\\Ankan\\Desktop\\samsung_hand_gesture\\%s\\%s"%(dir,directory)
        files = [f for f in os.listdir(my_path) if isfile(join(my_path, f))]
        
        for ff in files:
            array = np.array(Image.open("C:\\Users\\Ankan\\Desktop\\samsung_hand_gesture\\%s\\%s\\%s"%(dir,directory,ff)).convert("1"))
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
    return req_x, req_y_
    
x_train, y_train = build_data("Samsung-Marcel-Train")
x_test, y_test = build_data("Samsung-Marcel-Test")

##### DESIGNING THE NEURAL NETWORK #####

# Placeholder variable for the input images
x = tf.placeholder(tf.float32, shape=[None, ceil_max], name='x')
# Reshape it into [num_images, img_height, img_width, num_channels]
x_image = tf.reshape(x, [-1, ceil_val, ceil_val, 1])

# Placeholder variable for the true labels associated with the images
y_true = tf.placeholder(tf.float32, shape=[None, 6], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)


def new_conv_layer(input, num_input_channels, filter_size, num_filters, name):
    
    with tf.variable_scope(name) as scope:
        # Shape of the filter-weights for the convolution
        shape = [filter_size, filter_size, num_input_channels, num_filters]

        # Create new weights (filters) with the given shape
        weights = tf.Variable(tf.truncated_normal(shape, stddev=0.05))

        # Create new biases, one for each filter
        biases = tf.Variable(tf.constant(0.05, shape=[num_filters]))

        # TensorFlow operation for convolution
        layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='VALID')

        # Add the biases to the results of the convolution.
        layer += biases
        
        return layer, weights
    
def new_pool_layer(input, name):
    
    with tf.variable_scope(name) as scope:
        # TensorFlow operation for convolution
        layer = tf.nn.avg_pool(value=input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        
        return layer    

def new_relu_layer(input, name):
    
    with tf.variable_scope(name) as scope:
        # TensorFlow operation for convolution
        layer = tf.nn.relu(input)
        
        return layer    
    
def new_fc_layer(input, num_inputs, num_outputs, name):
    
    with tf.variable_scope(name) as scope:

        # Create new weights and biases.
        weights = tf.Variable(tf.truncated_normal([num_inputs, num_outputs], stddev=0.05))
        biases = tf.Variable(tf.constant(0.05, shape=[num_outputs]))
        
        # Multiply the input and weights, and then add the bias-values.
        layer = tf.matmul(input, weights) + biases
        
        return layer
    
# Convolutional Layer 1
layer_conv1, weights_conv1 = new_conv_layer(input=x_image, num_input_channels=1, filter_size=5, num_filters=6, name ="conv1")

# Pooling Layer 1
layer_pool1 = new_pool_layer(layer_conv1, name="pool1")

# RelU layer 1
layer_relu1 = new_relu_layer(layer_pool1, name="relu1")
#layer_relu1 = new_relu_layer(layer_conv1, name="relu1")

# Convolutional Layer 2
layer_conv2, weights_conv2 = new_conv_layer(input=layer_relu1, num_input_channels=6, filter_size=5, num_filters=16, name= "conv2")

# Pooling Layer 2
layer_pool2 = new_pool_layer(layer_conv2, name="pool2")

# RelU layer 2
layer_relu2 = new_relu_layer(layer_pool2, name="relu2")


# Flatten Layer
num_features = layer_relu2.get_shape()[1:4].num_elements()
layer_flat = tf.reshape(layer_relu2, [-1, num_features])

# Fully-Connected Layer 1
layer_fc1 = new_fc_layer(layer_flat, num_inputs=num_features, num_outputs=128, name="fc1")

# RelU layer 3
layer_relu3 = new_relu_layer(layer_fc1, name="relu3")

#layer_relu3 = tf.nn.dropout(layer_relu3, 0.5)

# Fully-Connected Layer 2
layer_fc2 = new_fc_layer(input=layer_relu3, num_inputs=128, num_outputs=6, name="fc2")    


# Use Softmax function to normalize the output
with tf.variable_scope("Softmax"):
    y_pred = tf.nn.softmax(layer_fc2)
    y_pred_cls = tf.argmax(y_pred, dimension=1)
    
# Use Cross entropy cost function
with tf.name_scope("cross_entropy"):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
    cost = tf.reduce_mean(cross_entropy)   
    
# Use Adam Optimizer
with tf.name_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

# Accuracy
with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    
##### TRAINING AND TESTING #####
    
num_epochs = 100   

with tf.Session() as sess:
    # Initialize all variables
    sess.run(tf.global_variables_initializer())
    
    # Loop over number of epochs
    for i in range(num_epochs):
        
        start_time = time.time()
        train_accuracy = 0
        
        train_data = {x: x_train, y_true: y_train}

        # train model
        sess.run(optimizer, feed_dict=train_data)
        
        train_a, train_c = sess.run([accuracy, cross_entropy], feed_dict=train_data)
        # measure performance on test data
        test_data = {x: x_test , y_true: y_test }
        test_a, test_c = sess.run([accuracy, cross_entropy], feed_dict=test_data)

        if i % 5 == 0:
            print("{} Train accuracy: {}, Test accuracy: {}".format(i,train_a,test_a))


       