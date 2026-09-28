################################################################################    
################## SELF DRIVING CAR ENGINEER NANO DEGREE #######################
################################################################################

##### Project three: Traffic signs classifier
##### by Felipe Rojas
##### In this project, I will create a Convolutional Neural Network for Traffic Signs
##### recognition using TensorFlow and the LeNet Architecture


############################ Step 0. Load the Data ############################

# Load pickled data
import pickle

training_file = 'traffic-signs-data/train.p'
validation_file= 'traffic-signs-data/valid.p'
testing_file = 'traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

##################### Step 1. Dataset Summary & Exploration #####################

#The pickled data is a dictionary with 4 key/value pairs:

# 'features'` is a 4D array containing raw pixel data of the traffic sign images, 
# (num examples, width, height, channels).

# 'labels'` is a 1D array containing the label/class id of the traffic sign.
# The file `signnames.csv` contains id -> name mappings for each id.

# 'sizes'` is a list containing tuples, (width, height) representing the original
# width and height the image.

# 'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a
# bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. 
# THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**

n_train = len(X_train)
n_validation =len(X_valid)
n_test = len(X_test)
image_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
n_classes = len(set(y_train))
total_data = n_train + n_validation + n_test

print("Number of training examples =", n_train)
print("Number of validation examples =", n_validation)
print("Number of testing examples =", n_test)
print("Total Number of data examples =", total_data)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

#### Data exploration visualization 
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

index = random.randint(0, len(X_train))
image = X_train[index].squeeze()

plt.figure(figsize=(1,1))
plt.imshow(image)
y_label = y_train[index]
print(y_label)
# Here we can Check the plot and label number with the actual Sign number assignation
# in the traffic_signs database 
traffic_signs = np.array(pd.read_csv('signnames.csv'))
print(traffic_signs[y_label][1])


################ Step 2. Design and Test a Model Architecture #################

# Now, here I will Build the LeNet Architecture

from tensorflow.contrib.layers import flatten

def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the
    # weights and biases for each layer
    # The number of Channels is 3, and the number of outputs or Classes is 43
    mu = 0
    sigma = 0.1
    dropout = 0.5
    
    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    
    # Activation of Layer 1
    conv1 = tf.nn.relu(conv1) #Rectified Linear Unit activation

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # Activation of Layer 2
    conv2 = tf.nn.relu(conv2) #Rectified Linear Unit activation
    
    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # TODO: Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)
    
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # TODO: Activation of layer 3
    fc1    = tf.nn.relu(fc1)
    #fc1 = tf.nn.dropout(fc1, rate =(1 - dropout))
    
    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # TODO: Activation.
    fc2    = tf.nn.relu(fc2)
    
    # TODO: Layer 5: Fully Connected. Input = 84. Output = 43
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits

## With the LeNet Architecture defined, time to get the data ready for it!

## First, the preprocessing section
## Shuffle the data 
from sklearn.utils import shuffle
import cv2
X_train, y_train = shuffle(X_train, y_train)

### I will try to convert the datasets into grayscale images
### SOLUTION > https://stackoverflow.com/questions/56390917/convert-a-list-of-images-to-grayscale-using-opencv

X_train_grayscale = np.zeros(X_train.shape[:-1])
X_valid_grayscale = np.zeros(X_valid.shape[:-1])
X_test_grayscale = np.zeros(X_test.shape[:-1])

for i in range(X_train.shape[0]): 
    X_train_grayscale[i] = cv2.cvtColor(X_train[i], cv2.COLOR_BGR2GRAY) 

for i in range(X_valid.shape[0]): 
    X_valid_grayscale[i] = cv2.cvtColor(X_valid[i], cv2.COLOR_BGR2GRAY)     

for i in range(X_test.shape[0]): 
    X_test_grayscale[i] = cv2.cvtColor(X_test[i], cv2.COLOR_BGR2GRAY) 

### The LeNet algorithm works with arrays of the size: (None, height, width, channel)
### Since these images are grayscale images, channel = 1. So, I need to expand one dimension for the
### new grayscale variables
X_train_grayscale =np.expand_dims(X_train_grayscale, axis=3)
X_valid_grayscale =np.expand_dims(X_valid_grayscale, axis=3)
X_test_grayscale =np.expand_dims(X_test_grayscale, axis=3)

plt.imshow(X_train[0])
plt.imshow(X_train_grayscale[0].squeeze(), cmap = 'gray')

## I need to Normalize the image data so that the data has mean zero and equal variance
## For image data, (pixel - 128)/ 128 is a quick way to approximately normalize 
## the data and can be used in this project. 
X_train_norm = X_train_grayscale/128 - 1
X_valid_norm = X_valid_grayscale/128 - 1
X_test_norm = X_test_grayscale/128 - 1

#### Here I will be testing the LeNet architecture for Image classification
import tensorflow as tf
EPOCHS = 50
BATCH_SIZE = 128

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)

rate = 0.0025

# Best results for each +5 epochs WITHOUT dropout
# 10 epochs rate = 0.0013 accuracy = 0.89 / 
# 10 epochs rate = 0.0015 accuracy = 0.878 / 
# 10 epochs rate = 0.0017 accuracy = 0.902/ 
# 10 epochs rate = 0.002 accuracy = 0.916 / 
# 10 epochs rate = 0.0022 accuracy = 0.910 / 
# 10 epochs rate = 0.0025 accuracy = 0.935 / 0.917
# best accuracy = 0.935

# Best results for each +5 epochs 
# 15 epochs rate = 0.0013 accuracy = 0.897 / 
# 15 epochs rate = 0.0015 accuracy = 0.912 / 
# 15 epochs rate = 0.0017 accuracy = 0.919/ 0.909
# 15 epochs rate = 0.002 accuracy = 0.913 / 
# 15 epochs rate = 0.0022 accuracy = 0.910 / 
# 15 epochs rate = 0.0025 accuracy = 0.91
# best accuracy = 0.919

# Best results for each +5 epochs 
# 20 epochs rate = 0.0013 accuracy = 0.893 / 
# 20 epochs rate = 0.0015 accuracy = 0.917 / 
# 20 epochs rate = 0.0017 accuracy = 0.901/ 0.895
# 20 epochs rate = 0.002 accuracy = 0.920 / 0.902
# 20 epochs rate = 0.0022 accuracy = 0.92 / 0.92
# 20 epochs rate = 0.0025 accuracy = 0.907 / 
# best accuracy = 0.92
# with dropuout best accuracy = 0.927 / 0.92

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

####################### Training the Model with SESSION #######################
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train_norm)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train_norm, y_train = shuffle(X_train_norm, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train_norm[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_valid_norm, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './Saved_model/lenet')
    print("Model saved")
    
###################### Evaluating the Model with SESSION ######################    

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('./Saved_model'))

    test_accuracy = evaluate(X_test_norm, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))    
    
   
#################### Testing the model on brand new images ####################   

# To give yourself more insight into how your model is working, download at least
#five pictures of German traffic signs from the web and use your model to predict
#the traffic sign type.
#You may find `signnames.csv` useful as it contains mappings from the class id (integer)
# to the actual sign name.

### Load the images and plot them here.
### Feel free to use as many code cells as needed.
import glob
import cv2

images = glob.glob('test_images/test*.png')
test_images = []
# Each of the images I chose correspond to the following labels:
test_labels = [17, 25, 2, 13, 14]
for idx, fname in enumerate(images):
    test_image = cv2.imread(fname)
    test_image = cv2.resize(test_image, dsize=(32, 32), interpolation=cv2.INTER_LINEAR)
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    test_image = test_image/128 - 1
    test_images.append(test_image)
   
test_images =np.expand_dims(test_images, axis=3)   

### Checking the images
for img in range(len(test_images)):
    plt.figure()
    plt.imshow(test_images[img].squeeze(), cmap = 'gray')
    print(traffic_signs[test_labels[img]][1])

### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('./Saved_model'))

    test_accuracy = evaluate(test_images, test_labels)
    print("Test Accuracy = {:.3f}".format(test_accuracy))     