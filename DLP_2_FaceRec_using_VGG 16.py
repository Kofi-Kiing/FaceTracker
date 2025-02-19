#!/usr/bin/env python
# coding: utf-8

# In[6]:


from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt


# In[7]:


# re-size all the images to this
IMAGE_SIZE = [224, 224]

train_path ="C:/Users/Asus/Desktop/DeepLearningProjects/Datasets/Train"
valid_path = "C:/Users/Asus/Desktop/DeepLearningProjects/Datasets/Test"

# add preprocessing layer to the front of VGG
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# don't train existing weights
for layer in vgg.layers:
  layer.trainable = False
  

  
  # useful for getting number of classes
folders = glob("C:/Users/Asus/Desktop/DeepLearningProjects/Datasets/Train/*")
  

# our layers - you can add more if you want
x = Flatten()(vgg.output)
# x = Dense(1000, activation='relu')(x)
prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=vgg.input, outputs=prediction)

# view the structure of the model
model.summary()

# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory("C:/Users/Asus/Desktop/DeepLearningProjects/Datasets/Train",
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory("C:/Users/Asus/Desktop/DeepLearningProjects/Datasets/Test",
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')


# fit the model
r = model.fit(
  training_set,
  validation_data=test_set,
  epochs=5,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)
# loss
plt.plot(r.history['loss'], label='train loss')
#plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('Loss_vals')

# accuracies
plt.plot(r.history['accuracy'], label='train acc')
#plt.plot(r.history['val_acc'], label='val acc')
plt.legend()
plt.show()
plt.savefig('Acc_val')

import tensorflow as tf

from keras.models import load_model

model.save('features_of_face.h5')


# In[8]:


# Importing the libraries
from PIL import Image
from keras.applications.vgg16 import preprocess_input
import base64
from io import BytesIO
import json
import random
import cv2
from keras.models import load_model
import numpy as np


# In[9]:


from keras.preprocessing import image
import tensorflow as tf

print(f"Tensorflow version {tf.__version__}")
print("Number of GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU is available and configured")
        
    except:
        print("Could not set memory growth for GPU")
model = load_model('features_of_face.h5')


# In[6]:


get_ipython().system('pip list')


# In[11]:


import cv2
print(cv2.__version__)
width=640
height=360
cam=cv2.VideoCapture(0,cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH,width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
cam.set(cv2.CAP_PROP_FPS,30)
cam.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))

faceCascade=cv2.CascadeClassifier('haar\haarcascade_frontalface_default.xml')

count=0

while True:
        ignore, frame = cam.read()    
        frameGray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(frame,1.3,5)
        
        all_faces=[]
        for face in faces:
            count+=1
            x,y,w,h=face
            print('x=',x,'y=',y,'w=',w,'h=',h)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
            cropped_face=frame[y:y + h +50, x:x + w +50]
            all_faces.append(cropped_face)
        cropped_faces = np.array(all_faces)           
        print(cropped_faces.shape)
        
        if len(cropped_faces)==0 or cropped_faces[0].size==0:
            cv2.putText(frame, "No face found", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),2)
            #cv2.imshow("video", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue    
        
        cropped_faces= cropped_faces.astype(np.uint8)
        print(cropped_faces.shape)
        real_image=cropped_faces[0]
        print("length of cropped faces is: ",len(cropped_faces))
        print(f"Shape of cropped_faces: {cropped_faces.shape}")
        print(f"Shape of real_image: {real_image.shape}")
        print(type(cropped_faces))
       
    
    
                        
        if real_image is not None and len(real_image) > 0:
            print("=====================0==========================")
            face = cv2.resize(real_image, (224, 224))
            print("=======================1==============================")
            print(face.shape)
            im = Image.fromarray(face, 'RGB')
           #Resizing into 128x128 because we trained the model with this image size.
            img_array = np.array(im)
                    #Our keras model used a 4D tensor, (images x height x width x channel)
                    #So changing dimension 128x128x3 into 1x128x128x3 
            img_array = np.expand_dims(img_array, axis=0)
            print("=======================2==============================")
            print(img_array.shape)
            pred = model.predict(img_array)
            print(pred)
                     
            name="None matching"
        
            if len(pred) > 0:  # Check if prediction has output
                predicted_class = np.argmax(pred[0])
                class_names = {0: "Kingsley", 1: "Lemuel"} 
                
                if predicted_class in class_names:  # Check if class index is valid
                    name = class_names[predicted_class]
                    cv2.putText(frame, name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame,"No face found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2) 
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
video_capture.release()
cv2.destroyAllWindows()    
            
             


# In[ ]:




