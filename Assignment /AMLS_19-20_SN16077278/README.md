# README
### Organisation of the folders

Most folders were used during development. 
Methods with final in their name save to folders with final in their name. Same goes for refine. 

All the code is within the main folder. And each tasks results are saved within their dedicated folder. 
The methods from Data_Processing should be used to return processed images for model training. 

### Organisation of code modules
* Data_Processing.py
  * Data_A
    * preprocessing_CNN_A1
    * preprocessing_CNN_A2
    * face_detection_dlib
  * Data_B
    * process_B1_svm
    * process_B2_svm
* ModelA.py
  * Model_A1
    * find_best_CNN
    * gender_CNN_refine
    * gender_CNN_final
    * test_gender_CNN_final
  * Model_A2
    * find_best_CNN
    * smile_CNN_refine
    * smile_CNN_final
    * test_smile_CNN_final
* ModelB.py
  * Model_B1
    * find_face_shape_svm
    * train_face_shape_svm
    * test_face_shape_svm
  * Model_B2
    * find_eye_svm
    * train_eye_svm
    * test_eye_svm
* main.py 

### Folder organisation

``` bash
.
├── Assignment\ 
│   └── AMLS_19-20_SN16077278
│       ├── A1
│       │   ├── final_model
│       │   │   └── logs
│       │   ├── logs
│       │   │   ├── cropped_images
│       │   │   └── non-cropped_images
│       │   ├── model_refine
│       │   │   └── logs
│       │   ├── models_croped
│       │   └── models_no-crop
│       ├── A2
│       │   ├── logs
│       │   ├── model_final
│       │   │   └── logs
│       │   ├── model_refine
│       │   └── models
│       ├── B1
│       ├── B2
│       ├── Data_Processing.py
│       ├── Datasets
│       │   └── README\ --\ dataset_AMLS_19-20.md
│       ├── ModelA.py
│       ├── ModelB.py
│       ├── README.md
│       ├── __pycache__
│       │   ├── Data_Processing.cpython-37.pyc
│       │   ├── ModelA.cpython-37.pyc
│       │   └── ModelB.cpython-37.pyc
│       ├── main.py
│       ├── mmod_human_face_detector.dat
│       └── shape_predictor_68_face_landmarks.dat
└── README.md
```
