#!/usr/bin/env python
# coding: utf-8

# In[17]:


#Guidelines before running the code

#Folder Desciriptions:-

# My trianing data is in the folder Desktop\LIMG\Traintest\Training
# Validation Data is in Desktop\Test
# Create two files called data.h5 and labels.h5 in Desktop\LIMG\Output


# In[ ]:


#importing the necessary libraries


import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from skimage.feature import greycomatrix, greycoprops
import pandas as pd
import mahotas
import os
import h5py


# In[ ]:


#initializing some important variables

fixed_size = tuple((100,100))
#initializing training path
train_path = "Desktop\LIMG\Traintest\Training"
num_trees = 100
bins = 8
test_size =0.10
seed = 9


# In[ ]:


#feature no:1. Extracting Shape Features

def fd_hu_moments(image):
    #converting to greyscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature


# In[ ]:


#feature no:2. Extracting Greyscale Features

def fd_haralick(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick


# In[ ]:


#feature no:3. Extracting Colour Features

def fd_histogram(image, mask=None):
   
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


# In[ ]:


#initializing the two training labels( folders )

train_labels = os.listdir(train_path)

#train_labels.sort()
print(train_labels)

global_features = []
labels = []

images_per_class = 10890


# In[ ]:




for training_name in train_labels:
   
    dir = os.path.join(train_path, training_name)

    current_label = training_name
    
    
    #going through each photo and extracting the three features
    for x in range(1,images_per_class+1):
        
        #each photo is made in the format dir\(x).jpg where x is the number of the image
        file = dir + "\(" + str(x) + ").jpg"
        image = cv2.imread(file)
        image = cv2.resize(image, fixed_size)
        
        fv_hu_moments = fd_hu_moments(image)
        fv_haralick   = fd_haralick(image)
        fv_histogram  = fd_histogram(image)

        #stacking these three features into a variable called global_feature
        global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

        labels.append(current_label)
        global_features.append(global_feature)

   


# In[ ]:


get_ipython().run_cell_magic('time', '', "targetNames = np.unique(labels)\nle = LabelEncoder()\ntarget = le.fit_transform(labels)\n\n\n# normalize the feature vector in the range (0-1)\nscaler = MinMaxScaler(feature_range=(0, 1))\nrescaled_features
                             = scaler.fit_transform(global_features)\n\n\n# save the feature vector using HDF5\nh5f_data 
                             = h5py.File('Desktop\\\\LIMG\\\\Output\\\\data.h5', 'w')\nh5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))\n\nh5f_label = h5py.File('Desktop\\\\LIMG\\\\Output\\\\labels.h5', 'w')\nh5f_label.create_dataset('dataset_1', data=np.array(target))\n\nh5f_data.close()\nh5f_label.close()")


# In[ ]:


#importing necessary packages for classification

import h5py
import numpy as np
import os
import glob
import cv2
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


results = []
names = []
scoring = "accuracy"


# In[ ]:


#Instead of using HDF5 file-format, we could use “.csv” file-format to store the features.
#But, as we will be working with large amounts of data in future, HDF5 format will be better.

h5f_data = h5py.File('Desktop\LIMG\Output\data.h5', 'r')
h5f_label = h5py.File('Desktop\LIMG\Output\labels.h5', 'r')

global_features_string = h5f_data['dataset_1']
global_labels_string = h5f_label['dataset_1']

global_features = np.array(global_features_string)
global_labels = np.array(global_labels_string)

h5f_data.close()
h5f_label.close()


# In[ ]:




#we will use train_test_split function provided by scikit-learn to split our training dataset into train_data and test_data. 
#By this way, we train the models with the train_data and
#test the trained model with the unseen test_data. The split size is decided by the test_size parameter.


(trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal)
                        = train_test_split(np.array(global_features), np.array(global_labels), test_size=test_size, random_state=seed)
                                                                                          


# In[ ]:



models = []
models.append(('RF', RandomForestClassifier(n_estimators=num_trees, random_state=9)))


# In[ ]:


get_ipython().run_cell_magic('time', '', '# filter all the warnings\nimport warnings\nwarnings.filterwarnings(\'ignore\')\n# go through random forest alggorithm and print accuracy\n#more models can be added in the models[] variable to compare accuracy\nfor name, model in models:\n    kfold = KFold(n_splits=10, random_state=7)\n    
 cv_results = cross_val_score(model, trainDataGlobal, trainLabelsGlobal, cv=kfold, scoring=scoring)\n    results.append(cv_results)\n    names.append(name)\n    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())\n    print(msg)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'import matplotlib.pyplot as plt\n\nclf 
= RandomForestClassifier(n_estimators=100, random_state=9)\n\n\nclf.fit(trainDataGlobal, trainLabelsGlobal)\n\n# path to validation data\ntest_path 
                             = "Desktop\\Test"\n\n#extract features of each validation image\nfor file in glob.glob(test_path + "\\*.jpg"):\n  \n    image = cv2.imread(file)\n\n\n    image = cv2.resize(image, fixed_size)\n\n\n    fv_hu_moments = fd_hu_moments(image)\n    fv_haralick   = fd_haralick(image)\n    fv_histogram  = fd_histogram(image)\n\n\n    global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])\n\n    #predict output using clf.fit()\n    prediction = clf.predict(global_feature.reshape(1,-1))[0]\n\n    print("File Name: "+file)\n    #output prediction\n    print(train_labels[prediction], ":")\n    image = cv2.resize(image, (2000,2000))\n    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))\n    plt.show()\n    print("\\n \\n \\n")')


# In[ ]:




