#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pydicom
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import keras 


# In[2]:


# This function reads in a .dcm file, checks the important fields for our device, and returns a numpy array
# of just the imaging data
def check_dicom(filename): 
    # todo
     dcm = pydicom.dcmread('test3.dcm')
     print('Load file {} ...'.format('test3.dcm'))
     ds = pydicom.dcmread('test3.dcm')       
     img = ds.pixel_array
        
     return img
    
    
# This function takes the numpy array output by check_dicom and 
# runs the appropriate pre-processing needed for our model input
def preprocess_image(img,img_mean,img_std,img_size): 
    img_mean = np.mean(dcm.pixel_array)
    img_std = np.std(dcm.pixel_array)
    img_size = np.size(img)
    proc_img = img_mean, img_std, img_size
    
    return proc_img

# This function loads in our trained model w/ weights and compiles it 
def load_model(model_path, weight_path):
    # todo
    
    
    return model

# This function uses our device's threshold parameters to predict whether or not
# the image shows the presence of pneumonia using our trained model
def predict_image(model, img, best_threshold, numericLabels=False): 
    
    result = my_model.predict(img)
    predict = result[0]
    if numericLabels:
        if(predict > best_threshold):
            prediction = 1
        else:
            prediction = 0
    else:
        if(predict > best_threshold):
            prediction = 'Prediction: Positive'
            
        else:
            prediction = 'Prediction Negative'

return prediction 


# In[ ]:


test_dicoms = ['test1.dcm','test2.dcm','test3.dcm','test4.dcm','test5.dcm','test6.dcm']

model_path = my_model.load_weights({}_my_model.best.hdf5)
weight_path = my_model.load_weights({}_my_model.best.hdf5)

IMG_SIZE=(1,224,224,3) # This might be different if you did not use vgg16
img_mean = # loads the mean image value they used during training preprocessing
img_std = # loads the std dev image value they used during training preprocessing

my_model = #loads model
thresh = #loads the threshold they chose for model classification 

# use the .dcm files to test your prediction
for i in test_dicoms:
    
    img = np.array([])
    img = check_dicom(i)
    img_reshape = img.reshape((img.shape[0], img.shape[1]))
    
    
    # Declare the mean and std here
    img_mean = np.mean(img)
    img_std = np.std(img)
    # To run prediction on single image, use numpy.expand_dims()
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    IMG_SIZE = (1,224, 224,3) # this is the shape for VGG16 model
    
    if img is None:
        continue
    print('Image shape: {}'.format(img.shape))

    img_proc = preprocess_image(img,img_mean,img_std,IMG_SIZE)
    pred = predict_image(my_model,img_proc,thresh)
    print(pred)


# In[ ]:





# In[ ]:





# In[ ]:




