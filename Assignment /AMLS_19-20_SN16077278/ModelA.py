import os
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from numpy.random import seed 
seed(1)
import tensorflow as tf
tf.random.set_seed(2)
import numpy as np
import pandas as pd
import sys
import pickle
import csv


from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks.tensorboard_v2 import TensorBoard
from keras.optimizers import Adam

# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
# from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint


import time 

class Model_A1():

	# Function to create multiple CNN-s and find the best one
	def find_best_CNN(self,X,y,X_test,y_test):
		
		dense_layers = [0, 1, 2]
		layer_sizes = [32, 64, 128]
		conv_layers = [1, 2 , 3]
		X = np.asarray(X)/255.0
		y = np.asarray(y)
		# X_test = np.asarray(X_test)/255.0
		# y_test = np.asarray(y_test)
		# best_model = pd.DataFrame(columns=["name","accuracy","loss"])

		for dense_layer in dense_layers:
			for layer_size in layer_sizes:
				for conv_layer in conv_layers:
					NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer,int(time.time()))
		            
					print(NAME)

					model = Sequential()

					model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
					model.add(Activation('relu'))
					model.add(MaxPooling2D(pool_size=(2, 2)))

					for l in range(conv_layer-1):
						model.add(Conv2D(layer_size, (3, 3)))
						model.add(Activation('relu'))
						model.add(MaxPooling2D(pool_size=(2, 2)))

					model.add(Flatten())
					for l in range(dense_layer):
						model.add(Dense(layer_size))
						model.add(Activation('relu'))

					model.add(Dense(1))
					model.add(Activation('sigmoid'))


					model.compile(loss='binary_crossentropy',
						optimizer='adam',
						metrics=['accuracy'])

					tensorboard = TensorBoard(log_dir="{}/A1/logs/non-cropped_images/{}".format(sys.path[0],NAME))
					NAME1 = "{}-conv-{}-nodes-{}-dense".format(conv_layer, layer_size, dense_layer)
					es = EarlyStopping(monitor = 'val_loss', mode='min', verbose=1, patience= 3)

					mc = ModelCheckpoint("{}/A1/models_no-crop/{}.h5".format(sys.path[0],NAME1),monitor='val_loss',mode='min',save_best_only=True)



					model.fit(X, y, batch_size=32, epochs=10, validation_split=0.25,callbacks=[tensorboard,es,mc])
		            
					model.save("{}/A1/models_no-crop/{}".format(sys.path[0],NAME))
					# result = model.evaluate(X_test,y_test,batch_size=32)

					# df = pd.DataFrame([[NAME, result[1], result[0]]] ,columns=["name","accuracy","loss"])
		     
					# best_model = best_model.append(df, ignore_index=True)

		# best_model.to_csv("{}/A1/best_model_nocrop1.csv".format(os.getcwd()),sep=",")
		# print(best_model)

		pass
	

	def gender_CNN_refine(self, X,y):

		learning_rates = [0.001]
		droputs = [0.15]

		best_model = pd.DataFrame(columns=["name","accuracy","loss"])
		for learn in learning_rates:
			for drop in droputs:

				NAME = "gender_detect_CNN_im-{}_l{}_dropout{}_time-{}".format(X.shape[1],learn,drop,int(time.time()))

				model = Sequential()

				model.add(Conv2D(32,(3,3),input_shape=X.shape[1:]))
				model.add(Activation('relu'))
				model.add(MaxPooling2D(pool_size=(2,2)))

				model.add(Conv2D(32,(3,3)))
				model.add(Activation('relu'))
				model.add(MaxPooling2D(pool_size=(2,2)))

				model.add(Dropout(drop))

				model.add(Conv2D(32,(3,3)))
				model.add(Activation('relu'))
				model.add(MaxPooling2D(pool_size=(2,2)))

				model.add(Flatten())

				# model.add(Dense(32))
				# model.add(Activation('relu'))

				model.add(Dense(1))
				model.add(Activation('sigmoid'))

				opt = Adam(lr = learn)

				model.compile(
					loss = 'binary_crossentropy',
					optimizer = opt,
					metrics = ['accuracy'])

				print(model.summary())

				tb = TensorBoard(log_dir="{}/A1/model_refine/logs/{}".format(os.getcwd(),NAME))

				es = EarlyStopping(
					monitor = 'val_loss', 
					mode='min', 
					verbose=1, 
					patience= 5)

				mc = ModelCheckpoint(
					"{}/A1/model_refine/{}.h5".format(os.getcwd(),NAME),
					monitor='val_loss',
					mode='min',
					save_best_only=True)


				model.fit(
					X,y, 
					epochs = 20,
					batch_size = 32,
					validation_split = 0.25,
					callbacks = [tb, es, mc])

	def gender_CNN_final(self,X,y):
		NAME = "gender_detect_CNN_final"

		model = Sequential()

		model.add(Conv2D(32,(3,3),input_shape=X.shape[1:]))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2,2)))

		model.add(Conv2D(32,(3,3)))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2,2)))

		model.add(Dropout(0.15))

		model.add(Conv2D(32,(3,3)))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2,2)))

		model.add(Flatten())

		# model.add(Dense(32))
		# model.add(Activation('relu'))

		model.add(Dense(1))
		model.add(Activation('sigmoid'))

		opt = Adam(lr = 0.001)

		model.compile(
			loss = 'binary_crossentropy',
			optimizer = opt,
			metrics = ['accuracy'])

		print(model.summary())

		tb = TensorBoard(log_dir="{}/A1/model_final/logs/{}".format(os.getcwd(),NAME))

		es = EarlyStopping(
			monitor = 'val_loss', 
			mode='min', 
			verbose=1, 
			patience= 5)

		mc = ModelCheckpoint(
			"{}/A1/{}.h5".format(os.getcwd(),NAME),
			monitor='val_loss',
			mode='min',
			save_best_only=True)


		model.fit(
			X,y, 
			epochs = 25,
			batch_size = 32,
			validation_split = 0.25,
			callbacks = [tb, es, mc])


	def test_gender_CNN_final(self,X,y,X1,y1):

		# Load additional test data



		model = load_model("{}/A1/gender_detect_CNN_final.h5".format(os.getcwd()))

		result = model.evaluate(X,y,batch_size=32)

		result1 = model.evaluate(X1,y1,batch_size=32)

		from keras.utils.vis_utils import plot_model
		plot_model(model, to_file='A1/gender_CNN_final.png',show_shapes = True, dpi = 200)

		return result, result1
		pass

	# returns the performance of each created model on the test data
	def test_CNN(self,X,y):

		X_test = np.asarray(X)/255.0
		y_test = np.asarray(y)

		acc = 0


		dense_layers = [0, 1, 2]
		layer_sizes = [32, 64, 128]
		conv_layers = [1, 2, 3]
		# best_model = ["1",""]

		best_model = pd.DataFrame(columns=["name","accuracy","loss"])



		for dense_layer in dense_layers:
		    for layer_size in layer_sizes:
		        for conv_layer in conv_layers:
		            NAME = "{}-conv-{}-nodes-{}-dense".format(conv_layer, layer_size, dense_layer)
		            model = tf.keras.models.load_model("{}/A1/models_no-crop/{}.h5".format(os.getcwd(),NAME))
		            result = model.evaluate(X_test, y_test, batch_size=32)
		            df = pd.DataFrame([[NAME, result[1], result[0]]] ,columns=["name","accuracy","loss"])
		     
		            best_model = best_model.append(df, ignore_index=True)

		best_model.to_csv("{}/A1/best_model_nocrop.csv".format(os.getcwd()),sep=",")
		print(best_model)
		
		return best_model





class Model_A2():


	def svm1(self,):
		
		from sklearn.svm import SVC


		pickle_in = open("A2/X_train_svm.pickle","rb")
		X_train = pickle.load(pickle_in)

		pickle_in = open("A2/y_train_svm.pickle","rb")
		y_train = pickle.load(pickle_in)

		pickle_in = open("A2/X_test_svm.pickle","rb")
		X_test = pickle.load(pickle_in)

		pickle_in = open("A2/y_test_svm.pickle","rb")
		y_test = pickle.load(pickle_in)

		clf = SVC(kernel='linear', probability=True, tol=1e-3, verbose = True)

		clf.fit(X_train,y_train)


		pred_lin = clf.score(X_test, y_test)

		pickle_out = open("A2/SVM_model1.pickle","wb")
		pickle.dump(clf,pickle_out)
		pickle_out.close()

		return pred_lin

	def test_svm(self,):
		pickle_in = open("A2/X_test_svm.pickle","rb")
		X_test = pickle.load(pickle_in)

		pickle_in = open("A2/y_test_svm.pickle","rb")
		y_test = pickle.load(pickle_in)

		from sklearn.svm import SVC
		from sklearn.metrics import classification_report

		model = pickle.load(open("{}/A2/SVM_model1.pickle".format(os.getcwd()),"rb"))
		result = classification_report(model.predict(X_test),y_test)
		print(result)

		return result


	def find_smile_detection_svm(self,X,y,X_test,y_test):
		from sklearn.svm import SVC
		from sklearn.model_selection import GridSearchCV
		from sklearn.metrics import classification_report
		
		tuned_parameters = [{'kernel':['linear','rbf','poly'],'gamma':[0.1 ,0.01,1e-3,1e-4,1e-5], 'C':[0.01 ,0.1,1,10,100,1000]}]

		score = 'recall'

		# clf = GridSearchCV(SVC(), tuned_parameters, scoring = "{}_weighted".format(score))

		# clf.fit(X,y)

		# print("Best parameters set found on development set:")

		# print()
		# print(clf.best_params_)
		# print()
		# print("Grid scores on development set:")
		# print()
		# means = clf.cv_results_['mean_test_score']
		# stds = clf.cv_results_['std_test_score']
		# for mean, std, params in zip(means, stds, clf.cv_results_['params']):
		#     print("%0.3f (+/-%0.03f) for %r"
		#           % (mean, std * 2, params))
		# print()

		# print("Detailed classification report:")
		# print()
		# print("The model is trained on the full development set.")
		# print("The scores are computed on the full evaluation set.")
		# print()
		clf = SVC(kernel = 'linear', C = 10, gamma = 0.1, verbose = True)
		clf.fit(X,y)
		print(clf.score(X_test,y_test))
		y_true, y_pred = y_test, clf.predict(X_test)
		print(classification_report(y_true, y_pred))
		# print()
    	




	def find_best_CNN(self,X_train,y_train):
		# pickle_in = open("A2/X_train_c.pickle","rb")
		# X_train = pickle.load(pickle_in)

		# pickle_in = open("A2/y_train_c.pickle","rb")
		# y_train = pickle.load(pickle_in)

		# pickle_in = open("A2/X_test_c.pickle","rb")
		# X_test = pickle.load(pickle_in)

		# pickle_in = open("A2/y_test_c.pickle","rb")
		# y_test = pickle.load(pickle_in)

			
		dense_layers = [0,1,2]
		layer_sizes = [32,64,128]
		conv_layers = [1,2,3]
		X = np.asarray(X_train)
		y = np.asarray(y_train)
		# best_model = pd.DataFrame(columns=["name","accuracy","loss"])

		for dense_layer in dense_layers:
			for layer_size in layer_sizes:
				for conv_layer in conv_layers:
					NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer,int(time.time()))
		            
					print(NAME)

					model = Sequential()

					model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
					model.add(Activation('relu'))
					model.add(MaxPooling2D(pool_size=(2, 2)))

					for l in range(conv_layer-1):
						model.add(Conv2D(layer_size, (3, 3)))
						model.add(Activation('relu'))
						model.add(MaxPooling2D(pool_size=(2, 2)))

					model.add(Flatten())
					for l in range(dense_layer):
						model.add(Dense(layer_size))
						model.add(Activation('relu'))

					model.add(Dense(1))
					model.add(Activation('sigmoid'))


					model.compile(loss='binary_crossentropy',
						optimizer='adam',
						metrics=['accuracy'])

					tensorboard = TensorBoard(log_dir="{}/A2/logs/{}".format(sys.path[0],NAME))
					NAME1 = "{}-conv-{}-nodes-{}-dense".format(conv_layer, layer_size, dense_layer)
					es = EarlyStopping(monitor = 'val_loss', mode='min', verbose=1, patience= 3)

					mc = ModelCheckpoint("{}/A2/models/{}.h5".format(sys.path[0],NAME1),monitor='val_loss',mode='min',save_best_only=True)



					model.fit(X, y, batch_size=32, epochs=10, validation_split=0.25,callbacks=[tensorboard,es,mc])
		pass

	def test_CNN(self,):

		pickle_in = open("A2/X_test.pickle","rb")
		X_test = pickle.load(pickle_in)

		pickle_in = open("A2/y_test.pickle","rb")
		y_test = pickle.load(pickle_in)

		X_test = np.asarray(X_test)/255.0
		y_test = np.asarray(y_test)


		dense_layers = [1, 2, 3]
		layer_sizes = [32, 64, 128]
		conv_layers = [1, 2, 3]

		best_model = pd.DataFrame(columns=["name","accuracy","loss"])



		for dense_layer in dense_layers:
		    for layer_size in layer_sizes:
		        for conv_layer in conv_layers:
		            NAME = "{}-conv-{}-nodes-{}-dense".format(conv_layer, layer_size, dense_layer)
		            model = tf.keras.models.load_model("{}/A2/models/{}.h5".format(os.getcwd(),NAME))
		            result = model.evaluate(X_test, y_test, batch_size=32)
		            df = pd.DataFrame([[NAME, result[1], result[0]]] ,columns=["name","accuracy","loss"])
		     
		            best_model = best_model.append(df, ignore_index=True)

		best_model.to_csv("{}/A2/best_model.csv".format(os.getcwd()),sep=",")
		print(best_model)
		
		return best_model



	def smile_CNN_refine(self,X,y):

		# X = pickle.load(open("{}/A2/X_train.pickle".format(os.getcwd()),"rb"))
		# y = pickle.load(open("{}/A2/y_train.pickle".format(os.getcwd()),"rb"))

		learning_rates = [0.001]
		droputs1 = [0.2]
		droputs2 = [0.01,0.1, 0.15, 0.2]

		
		for learn in learning_rates:
			for drop1 in droputs1:
				
				NAME = "smile_detect_CNN_im-{}_l{}_do1-{}_{}_time-{}".format(X.shape[1],learn,0,"",int(time.time()))

				model = Sequential()

				model.add(Conv2D(32,(3,3),input_shape=X.shape[1:],padding = 'same'))
				model.add(Activation('relu'))
				model.add(MaxPooling2D(pool_size=(2,2)))

				model.add(Conv2D(32,(3,3)))
				model.add(Activation('relu'))
				model.add(MaxPooling2D(pool_size=(2,2)))

				# model.add(Dropout(drop1))

				model.add(Conv2D(32,(3,3)))
				model.add(Activation('relu'))
				model.add(MaxPooling2D(pool_size=(2,2)))

				model.add(Flatten())

				

				# model.add(Dense(32))
				# model.add(Activation('relu'))

				# model.add(Dropout(drop1))

				# model.add(Dense(32))
				# model.add(Activation('relu'))

				model.add(Dense(1))
				model.add(Activation('sigmoid'))

				opt = Adam(lr = learn)

				model.compile(
					loss = 'binary_crossentropy',
					optimizer = opt,
					metrics = ['accuracy'])

				print(model.summary())

				tb = TensorBoard(log_dir="{}/A2/model_refine/logs/{}".format(os.getcwd(),NAME))

				es = EarlyStopping(
					monitor = 'val_loss', 
					mode='min', 
					verbose=1, 
					patience= 5)

				mc = ModelCheckpoint(
					"{}/A2/model_refine/{}.h5".format(os.getcwd(),NAME),
					monitor='val_loss',
					mode='min',
					save_best_only=True)


				model.fit(
					X,y, 
					epochs = 15,
					batch_size = 32,
					validation_split = 0.25,
					callbacks = [tb, es, mc])


	def smile_CNN_final(self,X,y):

		NAME = "smile_detect_CNN_final-{}".format(int(time.time()))
		NAME1 = "smile_detect_CNN_final"

		model = Sequential()

		model.add(Conv2D(32,(3,3),input_shape=X.shape[1:],padding = 'same'))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2,2)))

		model.add(Conv2D(32,(3,3)))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2,2)))

		# model.add(Dropout(drop1))

		model.add(Conv2D(32,(3,3)))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2,2)))

		model.add(Flatten())

		

		# model.add(Dense(32))
		# model.add(Activation('relu'))

		# model.add(Dropout(drop1))

		# model.add(Dense(32))
		# model.add(Activation('relu'))

		model.add(Dense(1))
		model.add(Activation('sigmoid'))

		opt = Adam(lr = 0.001)

		model.compile(
			loss = 'binary_crossentropy',
			optimizer = opt,
			metrics = ['accuracy'])

		print(model.summary())

		tb = TensorBoard(log_dir="{}/A2/model_final/logs/{}".format(os.getcwd(),NAME))

		es = EarlyStopping(
			monitor = 'val_loss', 
			mode='min', 
			verbose=1, 
			patience= 5)

		mc = ModelCheckpoint(
			"{}/A2/{}.h5".format(os.getcwd(),NAME1),
			monitor='val_loss',
			mode='min',
			save_best_only=True)


		model.fit(
			X,y, 
			epochs = 15,
			batch_size = 32,
			validation_split = 0.25,
			callbacks = [tb, es, mc])		

	def test_smile_CNN_final(self,X,y,X1,y1):

		# Load additional test data



		model = load_model("{}/A2/smile_detect_CNN_final.h5".format(os.getcwd()))

		result = model.evaluate(X,y,batch_size=32)

		result1 = model.evaluate(X1,y1,batch_size=32)

		from keras.utils.vis_utils import plot_model
		plot_model(model, to_file='A2/smile_CNN_final.png',show_shapes = True, dpi = 200)

		return result, result1
		pass


