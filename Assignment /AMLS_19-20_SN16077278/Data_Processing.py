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
from sklearn.model_selection import train_test_split 
import pickle
import dlib
import math






class Data_A():

	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

	DATADIR = "{}/Datasets/celeba/img".format(os.getcwd())
	DATADIR_test = "{}/Datasets/celeba_test/img".format(os.getcwd())

	csv_path = "{}/Datasets/celeba/labels.csv".format(os.getcwd())
	csv_path_test = "{}/Datasets/celeba_test/labels.csv".format(os.getcwd())
		

	# Function to detect a face within a single picture and crop it around the face
	def face_detection_dlib(self,image,face_detector):
		
		detected_faces = face_detector(image,1)
		if len(detected_faces)>0:
			for face in detected_faces:
				x = face.left()
				y = face.top()
				w = face.right() - x
				h = face.bottom() - y

				if x <0:
					x=0

				if y <0:
					y=0

				if w>178:
					w=178
				if h>218:
					h=218

				cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
				
				crop_img = image[y:y+h,x:x+w]
				
				return crop_img
		
		else: 
			return image


	# not in use, tested cropped data as input
	def processing_CNN_A1_crop(self,):
		
		################## Definitions and data read in #########################

		training_data = []	

		# initialise face detector
		detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")

		IMG_SIZE = 50

		here =  os.path.abspath(os.getcwd())

		DATADIR = os.path.join(here,'Datasets/celeba/img')

		csv_path = os.path.join(here,'Datasets/celeba/labels.csv')

		d_label = pd.read_csv(csv_path,index_col=0,names=["name","gender","smile"],sep="	",header=0)
		d_label.gender[(d_label.gender == -1)] = 0

		
		################## Picture Processing ###########################
		# Resizing images by first detecting faces in them using dlib and cropping the images to only include 
		# the face, then resizing each image to 50x50 colour image to reduce the load when training the CNN
		
		

		# print(d_label.gender[(d_label.gender == 0)])
		for i in tqdm(range(0,len(d_label.index))):
			
			image = cv2.imread(os.path.join(DATADIR,d_label.name[i]),cv2.IMREAD_COLOR)
			image = self.face_detection_dlib(image,detector)
			image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
			training_data.append([image,d_label.gender[i]])
		pass


		################## Splitting the Data into Train and Test set saving it to Pickle #############################
		

		# import random

		# random.shuffle(training_data)

		# train, validate, test = np.split(training_data.sample(frac=1), [int(.6*len(training_data)), int(.8*len(training_data))]) # 60/20/20 split

		
		print(len(training_data))
		X = []
		y = []

		for features,label in training_data:
		    X.append(features)
		    y.append(label)

		X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

		X_training = X[1000:]
		y_training = y[1000:]
		X_test = X[0:1000]
		y_test = y[0:1000]


		pickle_out = open("A1/X_train_c.pickle","wb")
		pickle.dump(X_training,pickle_out)
		pickle_out.close()

		pickle_out = open("A1/y_train_c.pickle","wb")
		pickle.dump(y_training,pickle_out)
		pickle_out.close()

		pickle_out = open("A1/X_test_c.pickle","wb")
		pickle.dump(X_test,pickle_out)
		pickle_out.close()

		pickle_out = open("A1/y_test_c.pickle","wb")
		pickle.dump(y_test,pickle_out)
		pickle_out.close()


	# function used for preprocessing uncropped data 
	def processing_CNN_A1(self, test = 0):
		IMG_SIZE = 50
		X = []

		label = pd.read_csv(self.csv_path,index_col=0,names=["name","gender","smile"],sep="	",header=0)
		y = (np.array(label.gender)+1)/2
		
		for i in tqdm(range(0,5000)):
			
			image = cv2.imread("{}/{}".format(self.DATADIR,"{}.jpg".format(i)))
			image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
			X.append(image)
		pass


		X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
		X = X/255

		split = round(len(X)*0.2)

		X_train = X[0:len(X)-split]
		y_train = y[0:len(X)-split]
		X_test = X[len(X)-split:]
		y_test = y[len(X)-split:len(X)]

		# load additional test data
		if test == 1:

			X = []

			label = pd.read_csv(self.csv_path_test,index_col=0,names=["name","gender","smile"],sep="	",header=0)
			y = (np.array(label.gender)+1)/2
			
			for i in tqdm(range(0,1000)):
				
				image = cv2.imread("{}/{}".format(self.DATADIR_test,"{}.jpg".format(i)))
				image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
				X.append(image)
			pass


			X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
			X = X/255

			X_test1 = X
			y_test1 = y

			return X_train, y_train, X_test, y_test, X_test1, y_test1

		return X_train, y_train, X_test, y_test


	# return the facial landmarks of the main detected face 
	def facial_landmark_vectors_dlib(self,image):
		detections = self.detector(image, 1)
		for k,d in enumerate(detections): #For all detected face instances individually
			shape = self.predictor(image,d) #Draw Facial Landmarks with the predictor class
			x_list = []
			y_list = []
			for i in range(1,68): #Store X and Y coordinates in two lists
				x_list.append(float(shape.part(i).x))
				y_list.append(float(shape.part(i).y))

			xmean = np.mean(x_list)
			ymean = np.mean(y_list)

			x_centre = [(x-xmean) for x in x_list]
			y_centre = [(y-ymean) for y in y_list]

			landmark_vectors = []

			for x,y,w,z in zip(x_centre,y_centre,x_list,y_list):
				landmark_vectors.append(w)
				landmark_vectors.append(z)
				mean = np.asarray((ymean,xmean))
				coord = np.asarray((z,w))
				dist = np.linalg.norm(coord - mean)
				landmark_vectors.append(dist)
				landmark_vectors.append((math.atan2(y,x)*360)/(2*math.pi))

		if len(detections) >= 1:
			return landmark_vectors

		else:
			return None


	def bounding_box(self,f):
		x = f.left()
		y = f.top()
		w = f.right() - x
		h = f.bottom() - y
		return x, y, w, h

	def facial_coord(self,d):
		landmarks = np.zeros((d.num_parts,2),dtype = "int")
		for i in range(1,68):
			landmarks[i] = (d.part(i).x,d.part(i).y)
		return landmarks

	def extract_landmark(self, image):
		det = self.detector(image,1)
		
		face_area = np.zeros((1,len(det)))
		face_landmarks = np.zeros((136, len(det)),dtype=np.int64)
		for k,d in enumerate(det):
			

			x,y,w,h = self.bounding_box(d)
			shape = self.predictor(image,d)
			face_landmarks[:,k] = np.reshape(self.facial_coord(shape),[136])
			face_area[0,k] = w*h
		if len(det)>0:
			landmarks = np.reshape(np.transpose(face_landmarks[:, np.argmax(face_area)]), [68, 2])

			return landmarks
		else:
			return None

			
	# NOT USED in FINAL version function agragate facial landmarks for SVM input.
	# SVM implementation not succesful
	def facial_landmarks_A2(self,):
		
		all_features = []
		all_labels = []

		label = pd.read_csv(self.csv_path,index_col=0,names=["name","gender","smile"],sep="	",header=0)
		all_labels = (np.array(label.smile)+1)/2
		
		for i in tqdm(range(0,500)):
			image = cv2.imread("{}/{}".format(self.DATADIR, "{}.jpg".format(i)))
			landmarks = self.facial_landmark_vectors_dlib(image)

			if landmarks is not None:
				all_features.append(landmarks)
			else:
				all_labels = np.delete(all_labels, i , axis = None)

		pass

		X = np.array(all_features)/255
		y = all_labels

		# X = np.reshape(X,(-1,136))
		print(X.shape)
		print(y.shape)

		split = round(len(X)*0.25)

		X_train = X[0:len(X)-split]
		y_train = y[0:len(X)-split]
		X_test = X[len(X)-split:]
		y_test = y[len(X)-split:len(X)]

		return X_train, y_train , X_test, y_test


		# pickle_out = open("A2/X_train_landmarks.pickle","wb")
		# pickle.dump(X_training,pickle_out)
		# pickle_out.close()

		# pickle_out = open("A2/y_train_landmarks.pickle","wb")
		# pickle.dump(y_training,pickle_out)
		# pickle_out.close()

		# pickle_out = open("A2/X_test_landmarks.pickle","wb")
		# pickle.dump(X_test,pickle_out)
		# pickle_out.close()

	# NOT IN USE CNN for A2 works better with cropped data
	def processing_CNN_A2(self,IMG_SIZE):

		training_data = []	

		

		here =  os.path.abspath(os.getcwd())

		DATADIR = os.path.join(here,'Datasets/celeba/img')

		csv_path = os.path.join(here,'Datasets/celeba/labels.csv')

		d_label = pd.read_csv(csv_path,index_col=0,names=["name","gender","smile"],sep="	",header=0)
		d_label.smile[(d_label.smile == -1)] = 0
		# print(d_label.gender[(d_label.gender == 0)])
		for i in tqdm(range(0,len(d_label.index))):
			
			image = cv2.imread(os.path.join(DATADIR,d_label.name[i]),cv2.IMREAD_COLOR)
			# image = self.face_detection(image)
			image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
			training_data.append([image,d_label.smile[i]])

		
		print(len(training_data))
		X = []
		y = []

		for features,label in training_data:
		    X.append(features)
		    y.append(label)

		X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

		X_training = X[1000:]
		y_training = y[1000:]
		X_test = X[0:1000]
		y_test = y[0:1000]


		pickle_out = open("A2/X_train.pickle","wb")
		pickle.dump(X_training,pickle_out)
		pickle_out.close()

		pickle_out = open("A2/y_train.pickle","wb")
		pickle.dump(y_training,pickle_out)
		pickle_out.close()

		pickle_out = open("A2/X_test.pickle","wb")
		pickle.dump(X_test,pickle_out)
		pickle_out.close()

		pickle_out = open("A2/y_test.pickle","wb")
		pickle.dump(y_test,pickle_out)
		pickle_out.close()

	# Active, in use
	def processing_CNN_A2_crop(self, test = 0):
		
		################## Definitions and data read in #########################
		IMG_SIZE = 50
		X = []	

		# initialise face detector
		# detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")

		label = pd.read_csv(self.csv_path,index_col=0,names=["name","gender","smile"],sep="	",header=0)
		all_labels = (np.array(label.smile)+1)/2
		
		################## Picture Processing ###########################
		# Resizing images by first detecting faces in them using dlib and cropping the images to only include 
		# the face, then resizing each image to IMG_SIZE colour image to reduce the load when training the CNN

		for i in tqdm(range(0,5000)):
			
			image = cv2.imread("{}/{}".format(self.DATADIR,"{}.jpg".format(i)))
			image = self.face_detection_dlib(image,self.detector)
			image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
			X.append(image)
		pass


		################## Splitting the Data into Train and Test set saving it to Pickle #############################

		X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
		X = X/255

		y = np.array(all_labels)

		split = round(len(X)*0.2)

		X_train = X[0:len(X)-split]
		y_train = y[0:len(X)-split]
		X_test = X[len(X)-split:]
		y_test = y[len(X)-split:len(X)]

		# pickle_out = open("A2/X_train_100.pickle","wb")
		# pickle.dump(X_train,pickle_out)
		# pickle_out.close()

		# pickle_out = open("A2/y_train_100.pickle","wb")
		# pickle.dump(y_train,pickle_out)
		# pickle_out.close()

		# pickle_out = open("A2/X_test_100.pickle","wb")
		# pickle.dump(X_test,pickle_out)
		# pickle_out.close()

		# pickle_out = open("A2/y_test_100.pickle","wb")
		# pickle.dump(y_test,pickle_out)
		# pickle_out.close()

		# load additional test set
		if test == 1:

			X = []

			label = pd.read_csv(self.csv_path_test,index_col=0,names=["name","gender","smile"],sep="	",header=0)
			y = (np.array(label.smile)+1)/2
			
			for i in tqdm(range(0,1000)):
				
				image = cv2.imread("{}/{}".format(self.DATADIR_test,"{}.jpg".format(i)))
				image = self.face_detection_dlib(image,self.detector)
				image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
				X.append(image)
			pass


			X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
			X = X/255

			X_test1 = X
			y_test1 = y

			return X_test, y_test, X_test1, y_test1


		return X_train, y_train, X_test, y_test
		

class Data_B():

	img_dir = "{}/Datasets/cartoon_set/img".format(os.getcwd())
	label = "{}/Datasets/cartoon_set/labels.csv".format(os.getcwd())

	img_dir_test = "{}/Datasets/cartoon_set_test/img".format(os.getcwd())
	label_test = "{}/Datasets/cartoon_set_test/labels.csv".format(os.getcwd())

	# NOT in USE used during development for data serialisation for debugging
	def all_faces_svm(self,IMG_SIZE = 50):

		

		y = pd.read_csv(self.label, index_col = 0, names = ['eye_color','face_shape','name'], header = 0, delimiter="	")

		X = []		
		for file in tqdm(os.listdir(os.fsencode(self.img_dir))):
			img = "{}/{}".format(self.img_dir, os.fsdecode(file))
			image = cv2.imread(img)
			# image = image[250:400, 150:350]
			# image = self.detect_skin(image)
			image = cv2.resize(image,(IMG_SIZE,IMG_SIZE))
			image = np.reshape(image,(IMG_SIZE*IMG_SIZE*3))
			X.append(image)

		X = np.asarray(X)/255

		y_eyes = y['eye_color']
		y_faces = y['face_shape']

		X_test = X[0:2000]
		X_train = X[2000:]
		y_eyes_test = y_eyes[0:2000]
		y_eyes_train = y_eyes[2000:]
		y_faces_train = y_faces[2000:]
		y_faces_test = y_faces[0:2000]

		print (X_train.shape)
		print(X_test.shape)

		pickle_out = open("B1/X_train_{}_svm.pickle".format(IMG_SIZE),"wb")
		pickle.dump(X_train,pickle_out,protocol=4)
		pickle_out.close()

		pickle_out = open("B2/X_train_{}_svm.pickle".format(IMG_SIZE),"wb")
		pickle.dump(X_train,pickle_out,protocol=4)
		pickle_out.close()


		pickle_out = open("B1/y_train_{}_svm.pickle".format(IMG_SIZE),"wb")
		pickle.dump(y_faces_train,pickle_out,protocol=4)
		pickle_out.close()

		pickle_out = open("B1/X_test_{}_svm.pickle".format(IMG_SIZE),"wb")
		pickle.dump(X_test,pickle_out,protocol=4)
		pickle_out.close()

		pickle_out = open("B2/X_test_{}_svm.pickle".format(IMG_SIZE),"wb")
		pickle.dump(X_test,pickle_out,protocol=4)
		pickle_out.close()

		pickle_out = open("B1/y_test_{}_svm.pickle".format(IMG_SIZE),"wb")
		pickle.dump(y_faces_test,pickle_out,protocol=4)
		pickle_out.close()

		pickle_out = open("B2/y_train_{}_svm.pickle".format(IMG_SIZE),"wb")
		pickle.dump(y_eyes_train,pickle_out,protocol=4)
		pickle_out.close()

		pickle_out = open("B2/y_test_{}_svm.pickle".format(IMG_SIZE),"wb")
		pickle.dump(y_eyes_test,pickle_out,protocol=4)
		pickle_out.close()

	# Presprocessing for B1
	def process_B1_svm(self, test =0 ,size = 1):
		IMG_SIZE = 100
		from sklearn.model_selection import train_test_split
		X = []
		for i in tqdm(range(0,round(8000*size))):
		    img = "{}/{}".format(self.img_dir, "{}.png".format(i))
		    image = cv2.imread(img)
		    image = cv2.resize(image,(IMG_SIZE,IMG_SIZE))
		    image = np.reshape(image,(IMG_SIZE*IMG_SIZE*3))
		    X.append(image)

		X = np.asarray(X)/255
		y = pd.read_csv(self.label,delimiter="	", header = 0)

		y = y.face_shape[0:round(8000*size)]

		X_train, X_test,y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 1 )

		if test == 1:

			X = []
			for i in tqdm(range(8000,10000)):
			    img = "{}/{}".format(self.img_dir, "{}.png".format(i))
			    image = cv2.imread(img)
			    image = cv2.resize(image,(IMG_SIZE,IMG_SIZE))
			    image = np.reshape(image,(IMG_SIZE*IMG_SIZE*3))
			    X.append(image)

			X_test = np.asarray(X)/255
			y = pd.read_csv(self.label,delimiter="	", header = 0)

			y_test = y.face_shape[8000:10000]

			
			X = []
			for i in tqdm(range(0,2500)):
			    img = "{}/{}".format(self.img_dir_test, "{}.png".format(i))
			    image = cv2.imread(img)
			    image = cv2.resize(image,(IMG_SIZE,IMG_SIZE))
			    image = np.reshape(image,(IMG_SIZE*IMG_SIZE*3))
			    X.append(image)

			X_test1 = np.asarray(X)/255
			y = pd.read_csv(self.label_test,delimiter="	", header = 0)
			y_test1 = y.face_shape

			return X_test, y_test, X_test1, y_test1


		return X_train,y_train, X_test, y_test
		pass

	# Presprocessing for B2
	def process_B2_svm(self,test =0 ,size = 1):
		IMG_SIZE = 100
		from sklearn.model_selection import train_test_split
		X = []
		for i in tqdm(range(0,round(8000*size))):
		    img = "{}/{}".format(self.img_dir, "{}.png".format(i))
		    image = cv2.imread(img)
		    image = cv2.resize(image,(IMG_SIZE,IMG_SIZE))
		    image = np.reshape(image,(IMG_SIZE*IMG_SIZE*3))
		    X.append(image)

		X = np.asarray(X)/255
		y = pd.read_csv(self.label,delimiter="	", header = 0)

		y = y.eye_color[0:round(8000*size)]

		X_train, X_test,y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 1 )

		if test == 1:

			X = []
			for i in tqdm(range(8000,10000)):
			    img = "{}/{}".format(self.img_dir, "{}.png".format(i))
			    image = cv2.imread(img)
			    image = cv2.resize(image,(IMG_SIZE,IMG_SIZE))
			    image = np.reshape(image,(IMG_SIZE*IMG_SIZE*3))
			    X.append(image)

			X_test = np.asarray(X)/255
			y = pd.read_csv(self.label,delimiter="	", header = 0)

			y_test = y.eye_color[8000:10000]


			
			X = []
			for i in tqdm(range(0,2500)):
			    img = "{}/{}".format(self.img_dir_test, "{}.png".format(i))
			    image = cv2.imread(img)
			    image = cv2.resize(image,(IMG_SIZE,IMG_SIZE))
			    image = np.reshape(image,(IMG_SIZE*IMG_SIZE*3))
			    X.append(image)

			X_test1 = np.asarray(X)/255
			y = pd.read_csv(self.label_test,delimiter="	", header = 0)
			y_test1 = y.eye_color

			return X_test, y_test, X_test1, y_test1


		return X_train,y_train, X_test, y_test


		pass


	

