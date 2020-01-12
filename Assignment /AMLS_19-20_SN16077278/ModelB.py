import os
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from numpy.random import seed 
seed(1)

import tensorflow as tf
tf.random.set_seed(2)
import cv2
from tqdm import tqdm
import numpy as np
import pandas as pd
import sys
import pickle
import dlib
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint


import time 


class Model_B1:

	img_dir = "{}/Datasets/cartoon_set/img".format(os.getcwd())
	label = "{}/Datasets/cartoon_set/labels.csv".format(os.getcwd())

	def find_face_shape_svm(self,X,y,X_t,y_t):

		from sklearn.svm import SVC
		from sklearn.model_selection import GridSearchCV
		from sklearn.metrics import classification_report

		tuned_parameters = [{'kernel':['linear','rbf','poly'],'gamma':[0.1 ,0.01,1e-3,1e-4,1e-5], 'C':[0.01 ,0.1,1,10,100,1000]}]

		clf = GridSearchCV(SVC(), tuned_parameters, scoring = "recall_weighted")

		clf.fit(X,y)

		print("Best parameters set found on development set:")

		print()
		print(clf.best_params_)
		print()
		print("Grid scores on development set:")
		print()
		means = clf.cv_results_['mean_test_score']
		stds = clf.cv_results_['std_test_score']
		for mean, std, params in zip(means, stds, clf.cv_results_['params']):
		    print("%0.3f (+/-%0.03f) for %r"
		          % (mean, std * 2, params))
		print()
		y_true, y_pred = y_t, clf.predict(X_t)
		report = classification_report(y_true, y_pred)

		print("Detailed classification report:")
		print()
		print(report)
		print("The model is trained on the full development set.")
		print("The scores are computed on the full evaluation set.")
		print()

		return report
		



	def train_face_shape_svm(self,X,y,X_t,y_t):
		from sklearn.metrics import classification_report
		from sklearn.svm import SVC

		clf = SVC(kernel='linear', C=0.01, gamma = 0.1, verbose = True)

		clf.fit(X,y)
		y_true, y_pred = y_t, clf.predict(X_t)
		report = classification_report(y_true, y_pred)

		print(report)

		pickle_out = open("B1/SVM_model_face_shape.pickle","wb")
		pickle.dump(clf,pickle_out)
		pickle_out.close()

		
	def test_face_shape_svm(self, X,y, X1,y1):
		from sklearn.metrics import classification_report

		model = pickle.load(open("{}/B1/SVM_model_face_shape.pickle".format(os.getcwd()),"rb"))
		y_true, y_pred = y, model.predict(X)
		report = classification_report(y_true, y_pred)

		print(report)

		y_true, y_pred = y1, model.predict(X1)
		report1 = classification_report(y_true, y_pred)
		
		print(report1)	
	
		return report, report1


class Model_B2:

	img_dir = "{}/Datasets/cartoon_set/img".format(os.getcwd())
	label = "{}/Datasets/cartoon_set/labels.csv".format(os.getcwd())

	def train_eye_svm(self,IMG_SIZE = 70):

		

		X = []

		lower = np.array([0, 0 , 0], dtype = "uint8")
		upper = np.array([60, 100, 255], dtype = "uint8")

		for i in tqdm(range(8000)):
			img = "{}/{}".format(self.img_dir, "{}.png".format(i))
			image = cv2.imread(img)
			image = image[240:290, 150:350]
			# img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
			# mask = cv2.inRange(img,lower,upper)
			# result = cv2.bitwise_and(image, image, mask = mask)
			# image = result
			image = cv2.resize(image,(IMG_SIZE,IMG_SIZE))
			image = np.reshape(image,(IMG_SIZE*IMG_SIZE*3))
			X.append(image)

		X = np.asarray(X)/255
		y = pd.read_csv(self.label,delimiter="	", header = 0)

		from sklearn.svm import SVC
		from sklearn.metrics import accuracy_score, recall_score, classification_report


		clf = SVC(kernel='linear', C = 0.01, gamma = 0.1 ,probability=True, tol=1e-3, verbose = True)

		clf.fit(X[0:6000],y.eye_color[0:6000])
		pred = clf.predict(X[6000:8000])

		print(np.asarray(y.eye_color[6000:8000]),"\n",pred)

		print(classification_report(y.eye_color[6000:8000],pred))


		pickle_out = open("B2/SVM_model_eye_color_cropped+masked_res-{}.pickle".format(IMG_SIZE),"wb")
		pickle.dump(clf,pickle_out)
		pickle_out.close()		
		return classification_report

	
	def test_eye_svm(self,IMG_SIZE = 70):
		from sklearn.metrics import recall_score, classification_report

		lower = np.array([0, 0 , 0], dtype = "uint8")
		upper = np.array([60, 100, 255], dtype = "uint8")

		X = []
		for i in tqdm(range(8000,10000)):
			img = "{}/{}".format(self.img_dir, "{}.png".format(i))
			image = cv2.imread(img)
			image = image[240:290, 150:350]
			# img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
			# mask = cv2.inRange(img,lower,upper)
			# result = cv2.bitwise_and(image, image, mask = mask)
			# image = result
			image = cv2.resize(image,(IMG_SIZE,IMG_SIZE))
			image = np.reshape(image,(IMG_SIZE*IMG_SIZE*3))
			X.append(image)

		X = np.asarray(X)/255
		y = pd.read_csv(self.label,delimiter="	", header = 0)
		y = y.eye_color[8000:10000]

		model = pickle.load(open("{}/B2/SVM_model_eye_color_cropped+masked_res-{}.pickle".format(os.getcwd(),IMG_SIZE),"rb"))
		# result = model.score(X,y.eye_color)

		# recall = recall_score(y.eye_color,model.predict(X),average = 'weighted')
		print(classification_report(y,model.predict(X)))

		# f = open("{}/B2/model_performance.txt".format(os.getcwd()),"a")
		# f.write("SVM_model_res-{}".format(IMG_SIZE)+" tested over: "+str(len(X))+" images, achieved: "
		# 	+str(result)+" accuracy and achieved: "+str(recall)+" recall score \n")
		# f.close()

		return classification_report


	def find_eye_svm(self,X,y,X_t,y_t):

		from sklearn.svm import SVC
		from sklearn.model_selection import GridSearchCV
		from sklearn.metrics import classification_report

		tuned_parameters = [{'kernel':['linear','rbf','poly'],'gamma':[0.1 ,0.01,1e-3,1e-4,1e-5], 'C':[0.01 ,0.1,1,10,100,1000]}]

		clf = GridSearchCV(SVC(), tuned_parameters, scoring = "recall_weighted")

		clf.fit(X,y)

		print("Best parameters set found on development set:")

		print()
		print(clf.best_params_)
		print()
		print("Grid scores on development set:")
		print()
		means = clf.cv_results_['mean_test_score']
		stds = clf.cv_results_['std_test_score']
		for mean, std, params in zip(means, stds, clf.cv_results_['params']):
		    print("%0.3f (+/-%0.03f) for %r"
		          % (mean, std * 2, params))
		print()
		y_true, y_pred = y_t, clf.predict(X_t)
		report = classification_report(y_true, y_pred)

		print("Detailed classification report:")
		print()
		print(report)
		print("The model is trained on the full development set.")
		print("The scores are computed on the full evaluation set.")
		print()

		return report



	def train_eye_svm(self,X,y,X_t,y_t):
		from sklearn.metrics import classification_report
		from sklearn.svm import SVC

		clf = SVC(kernel='linear', C=0.1, gamma = 0.1, verbose = True)

		clf.fit(X,y)
		y_true, y_pred = y_t, clf.predict(X_t)
		report = classification_report(y_true, y_pred)

		print(report)

		pickle_out = open("B2/SVM_model_eye.pickle","wb")
		pickle.dump(clf,pickle_out)
		pickle_out.close()

		return report

		
	def test_eye_svm(self, X,y, X1,y1):
		from sklearn.metrics import classification_report

		model = pickle.load(open("{}/B2/SVM_model_eye.pickle".format(os.getcwd()),"rb"))
		y_true, y_pred = y, model.predict(X)
		report = classification_report(y_true, y_pred)

		print(report)

		y_true, y_pred = y1, model.predict(X1)
		report1 = classification_report(y_true, y_pred)
		
		print(report1)	

		return report, report1






