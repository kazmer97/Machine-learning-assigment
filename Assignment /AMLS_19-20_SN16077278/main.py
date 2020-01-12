from numpy.random import seed
seed(1)
import numpy as np 
import matplotlib.pyplot as plt 
import os
import cv2
from tqdm import tqdm
import pandas as pd
import sys
import signal
import pickle
import math

import Data_Processing
import ModelA
# ======================================================================================================================
#### Data preprocessing
IMG_SIZE = 50

a = Data_Processing.Data_A()
# The below module has already been executed and saved into pickles

# a.processing_CNN_A1(IMG_SIZE) # create training data

#### Load preprocessed data from pickle

# The pickles contain the seperated test and training data set. The split has been done as instructed.
# The images saved in the pickle have been croped to only contain faces.
# The pickles contain numerical image data prepared as input for a convolutional neural neutwork
# 



# Non-cropped images

# pickle_in = open("A1/X_train_{}x{}.pickle".format(IMG_SIZE,IMG_SIZE),"rb")
# X_train = np.asarray(pickle.load(pickle_in))

# pickle_in = open("A1/y_train_{}x{}.pickle".format(IMG_SIZE,IMG_SIZE),"rb")
# y_train = np.asarray(pickle.load(pickle_in))

# pickle_in = open("A1/X_test_{}x{}.pickle".format(IMG_SIZE,IMG_SIZE),"rb")
# X_test = np.asarray(pickle.load(pickle_in))

# pickle_in = open("A1/y_test_{}x{}.pickle".format(IMG_SIZE,IMG_SIZE),"rb")
# y_test = np.asarray(pickle.load(pickle_in))

# Cropped images

# pickle_in = open("A1/X_train_c.pickle","rb")
# X_train = pickle.load(pickle_in)

# pickle_in = open("A1/y_train_c.pickle","rb")
# y_train = pickle.load(pickle_in)

# pickle_in = open("A1/X_test_c.pickle","rb")
# X_test = pickle.load(pickle_in)

# pickle_in = open("A1/y_test_c.pickle","rb")
# y_test = pickle.load(pickle_in)



# # ======================================================================================================================
# # Task A1
m = ModelA.Model_A1()

# best_model = m.find_best_CNN(X_train, y_train, X_test, y_test) # The line was already run, it creates 27 different networks with logs


X,y,X_test,y_test,X_test1,y_test1 = a.processing_CNN_A1(1)

m.gender_CNN_final(X,y) #train the model

res_A1, res_A1_1 = m.test_gender_CNN_final(X_test,y_test,X_test1,y_test1) #test the model on the 2 test sets

# print(res_A1, res_A1_1) 




# # ======================================================================================================================
# # Task A2


m = ModelA.Model_A2()

X_train, y_train, X_test, y_test = a.processing_CNN_A2_crop()
m.smile_CNN_final(X_train,y_train)




X_test, y_test, X_test1, y_test1 = a.processing_CNN_A2_crop(1)
res_A2, res_A2_1 = m.test_smile_CNN_final(X_test, y_test, X_test1, y_test1)


# print(result, result1)


# # ======================================================================================================================
# # Task B1

from ModelB import Model_B1

from Data_Processing import Data_B

a=Data_B()
m=Model_B1()

X_train,y_train, X_test, y_test = a.process_B1_svm()
m.train_face_shape_svm(X_train,y_train, X_test, y_test)


X_test, y_test, X_test1, y_test1 = a.process_B1_svm(1) #input1 is additional test set needed no/yes (0 or 1) input2 percentage of the Dataset needed, default is 100%
res_B1, res_B1_1 = m.test_face_shape_svm(X_test, y_test, X_test1, y_test1)

# print(result)





# # ======================================================================================================================
# # Task B2
# model_B2 = B2(args...)
# acc_B2_train = model_B2.train(args...)
# acc_B2_test = model_B2.test(args...)
# Clean up memory/GPU etc...

from ModelB import Model_B2


m = Model_B2()


# X_train,y_train, X_test, y_test = a.process_B2_svm(0,0.05)
# m.find_eye_svm(X_train,y_train,X_test,y_test)




X_train,y_train, X_test, y_test = a.process_B2_svm()
result = m.train_eye_svm(X_train,y_train,X_test,y_test)


X_test, y_test, X_test1, y_test1 = a.process_B2_svm(1)


res_B2, res_B2_1 = m.test_eye_svm(X_test, y_test, X_test1, y_test1)


# print (result)



# # ======================================================================================================================
# ## Print out your results with following format:

print("Results of each model on test set seperated from original dataset and additional test set provided: \n")

print("TA1 test set 1: test_accuracy: {} ; test_loss: {}".format(res_A1[1],res_A1[0]))

print("TA1 test set 1: test_accuracy: {} ; test_loss: {}".format(res_A1_1[1],res_A1_1[0]))

print("TA2 test set 1: test_accuracy: {} ; test_loss: {}".format(res_A2[1],res_A2[0]))

print("TA2 test set 1: test_accuracy: {} ; test_loss: {}".format(res_A2_1[1],res_A2_1[0]))

print("TB1 test set 1: test_accuracy: {} ; test_loss: {}".format(res_B1[1],res_B1[0]))
print("TB1 test set 1: test_accuracy: {} ; test_loss: {}".format(res_B1_1[1],res_B1_1[0]))

print("TB2 test set 1: test_accuracy: {} ; test_loss: {}".format(res_B2[1],res_B2[0]))
print("TB2 test set 1: test_accuracy: {} ; test_loss: {}".format(res_B2_1[1],res_B2_1[0]))


