#!/usr/bin/env python
# coding: utf-8

# In[24]:


import tensorflow as tf
import keras as keras


# In[25]:


import cv2


# In[26]:


import numpy as np


# In[27]:


import face_recognition


# In[28]:


import keras


# In[1]:


import cv2
print(cv2.__version__)
import time
width=640
height=360
myFont=cv2.FONT_HERSHEY_COMPLEX
cam=cv2.VideoCapture(1,cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH,width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
cam.set(cv2.CAP_PROP_FPS,33)
cam.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))

tlast=time.time()
time.sleep(.1)
fpsFILT=33

faceCascade=cv2.CascadeClassifier('haar\haarcascade_frontalface_default.xml')
eyeCascade=cv2.CascadeClassifier('haar\haarcascade_eye.xml')

while True:
        dT=time.time()-tlast
        fps=(1/dT)
        fpsFILT=fpsFILT*.97+fps*.03 
       
        tlast=time.time()
        ignore, frame = cam.read()    
        frameGray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(frameGray,1.3,5)
    
        for face in faces:
                x,y,w,h=face
                print('x=',x,'y=',y,'w=',w,'h=',h)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
                frameROI=frame[y:y+h,x:x+w]
                frameROIGray=cv2.cvtColor(frameROI,cv2.COLOR_BGR2GRAY)
                eyes=eyeCascade.detectMultiScale(frameROIGray)
                for eye in eyes:
                        xeye,yeye,weye,heye=eye
                        cv2.rectangle(frame[y:y+h,x:x+w],(xeye,yeye),(xeye+weye,yeye+heye),(255,0,0),3)
        
    
        
        cv2.putText(frame, str(int(fpsFILT))+' FPS',(5,30),myFont,1,(0,255,255),2)
        cv2.imshow('my Webcam',frame)
        cv2.moveWindow('my Webcam',640,0)
        if cv2.waitKey(1) & 0xff ==ord('q'):
                break
cam.release()        


# In[1]:


import mediapipe as mp


# In[4]:


import cv2
print(cv2.__version__)

class mpHands:
        import mediapipe as mp
        def __init__(self,maxHands=2,tol1=1,tol2=1):
                self.hands=self.mp.solutions.hands.Hands(False,maxHands,)
        def Marks(self,frame):
                myHands=[]
                frameRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                results=self.hands.process(frameRGB)
                if results.multi_hand_landmarks != None:
                    for handLandmarks in results.multi_hand_landmarks:
                        myHand=[]
                        for landMark in handLandmarks.landmark:
                                myHand.append((int(landMark.x*width),int(landMark.y*height)))
                        myHands.append(myHand)
                return myHands                
                                
                                              

width=640
height=360
cam=cv2.VideoCapture(1,cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH,width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
cam.set(cv2.CAP_PROP_FPS,30)
cam.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))

findHands=mpHands(2,.5,.5)
paddleWidth=125
paddleHeight=25
paddleColor=(0,255,0)

ballRadius=15
ballColor=(0,0,255)
xPos=int(width/2)
yPos=int(height/2)

DeltaX=2
DeltaY=2
score=0
lives=5 
font=cv2.FONT_HERSHEY_SIMPLEX
fontSize=6
ind=8

while True:
        ignore, frame = cam.read()  
        cv2.circle(frame,(xPos,yPos),ballRadius,ballColor,-1)
        cv2.putText(frame,str(score),(25,int(6*paddleHeight)),font,fontSize,paddleColor,5)
        cv2.putText(frame,str(lives),(width-125,int(6*paddleHeight)),font,fontSize,paddleColor,5)
        handData=findHands.Marks(frame) 
        for hand in handData:
            cv2.rectangle(frame,(int(hand[8][0]-paddleWidth/2),0),(int(hand[8][0]+paddleWidth/2),paddleHeight),paddleColor,-1)
        topEdgeBall=yPos-ballRadius
        bottomEdgeBall=yPos+ballRadius
        leftEdgeBall=xPos-ballRadius
        rightEdgeBall=xPos+ballRadius 


        if leftEdgeBall<=0 or rightEdgeBall>=width:
               DeltaX=DeltaX*(-1)
        if bottomEdgeBall>=height:
               DeltaY=DeltaY*(-1)

        if topEdgeBall<=paddleHeight:       
                if xPos>=int(hand[ind][0]-paddleWidth/2) and xPos<int(hand[ind][0]+paddleWidth/2):
                      DeltaY=DeltaY*(-1)
                      score=score+1 
                      if score==5 or score==10 or score==15 or score==20 or score==25:
                             DeltaY=DeltaY*2
                             DeltaX=DeltaX*2
                             
       
                else:
                      xPos=int(width/2)
                      yPos=int(height/2)  
                      lives=lives-1 
        xPos=xPos+DeltaX
        yPos=yPos+DeltaY                 
                                    
        cv2.imshow('my Webcam',frame)
        cv2.moveWindow('my Webcam',640,0)
        if lives==0:
                break
        if cv2.waitKey(1) & 0xff ==ord('q'):
                break
        cam.release()        


# In[2]:


import cv2
print(cv2.__version__)

class mpHands:
        import mediapipe as mp
        def __init__(self,maxHands=2,tol1=1,tol2=1):
                self.hands=self.mp.solutions.hands.Hands(False,maxHands,)
        def Marks(self,frame):
                myHands=[]
                frameRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                results=self.hands.process(frameRGB)
                if results.multi_hand_landmarks != None:
                    for handLandmarks in results.multi_hand_landmarks:
                        myHand=[]
                        for landMark in handLandmarks.landmark:
                                myHand.append((int(landMark.x*width),int(landMark.y*height)))
                        myHands.append(myHand)
                return myHands                
                                
                                              

width=1280
height=720
cam=cv2.VideoCapture(0,cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH,width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
cam.set(cv2.CAP_PROP_FPS,30)
cam.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))

findHands=mpHands(2,.5,.5)
paddleWidth=125
paddleHeight=25
paddleColor=(0,255,0)


while True:
        ignore, frame = cam.read()  
        handData=findHands.Marks(frame) 
        for hand in handData:
            cv2.rectangle(frame,(int(hand[8][0]-paddleWidth/2),0),(int(hand[8][0]+paddleWidth/2),paddleHeight),paddleColor,-1)
                      
                       
        cv2.imshow('my Webcam',frame)
        cv2.moveWindow('my Webcam',640,0)
        if cv2.waitKey(1) & 0xff ==ord('q'):
                break
cam.release()        


# In[16]:


import cv2
print(cv2.__version__)
import mediapipe as mp

width=1280
height=720

hands=mp.solutions.hands.Hands(False,2)
mpDraw=mp.solutions.drawing_utils

def parseLandmarks(frame):
        myHands=[]
        frameRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results=hands.process(frameRGB)
        if results.multi_hand_landmarks != None:
                for handLandmarks in results.multi_hand_landmarks:
                        myHand=[]
                        for landMark in handLandmarks.landmark:
                                myHand.append((int(landMark.x*width),int(landMark.y*height)))
                        myHands.append(myHand)        
                                
        return myHands                
                        
        
        

cam=cv2.VideoCapture(0,cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH,width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
cam.set(cv2.CAP_PROP_FPS,30)
cam.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
while True:
        ignore, frame = cam.read()    
        myHands=parseLandmarks(frame)
        for hand in myHands:
                for dig in [4,8,12,16,20]:                  
                    cv2.circle(frame,hand[dig],15,(0,255,0),3)
        cv2.imshow('my Webcam',frame)
        cv2.moveWindow('my Webcam',640,0)
        if cv2.waitKey(1) & 0xff ==ord('q'):
                break
cam.release()        


# In[7]:


import cv2
print(cv2.__version__)
width=640
height=360
cam=cv2.VideoCapture(1,cv2.CAP_DSHOW)
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
        
        for face in faces:
            count+=1
            x,y,w,h=face
            print('x=',x,'y=',y,'w=',w,'h=',h)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
            cropped_face=frame[y:y + h +50, x:x + w +50]
            
            # Save file in specified directory with unique name
            file_name_path ="C:/Users/Asus/Desktop/DeepLearningProjects/Datasets/Train/Kingsley/" + str(count) + '.jpg'
            cv2.imwrite(file_name_path, cropped_face)
            
        
            
        # Put count on images and display live count
        cv2.putText(frame, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow('my Webcam',frame)
        cv2.moveWindow('my Webcam',640,0)
        if cv2.waitKey(1) & 0xff ==ord('q') or count == 500:
                break
cam.release()  
cv2.destroyAllWindows() 


# In[ ]:


import cv2
print(cv2.__version__)
width=640
height=360
cam=cv2.VideoCapture(1,cv2.CAP_DSHOW)
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
        
        for face in faces:
            count+=1
            x,y,w,h=face
            print('x=',x,'y=',y,'w=',w,'h=',h)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
            cropped_face=frame[y:y + h +50, x:x + w +50]
            
            # Save file in specified directory with unique name
            file_name_path ="C:/Users/Asus/Desktop/DeepLearningProjects/Datasets/Train/Kingsley/" + str(count) + '.jpg'
            cv2.imwrite(file_name_path, cropped_face)
            
        
            
        # Put count on images and display live count
        cv2.putText(frame, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow('my Webcam',frame)
        cv2.moveWindow('my Webcam',640,0)
        if cv2.waitKey(1) & 0xff ==ord('q') or count == 500:
                break
cam.release()  
cv2.destroyAllWindows() 

