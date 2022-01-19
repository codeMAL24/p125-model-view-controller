import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import PIL.ImageOps
import os,ssl

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image

X = np.load('image.npz')['arr_0']
y = pd.read_csv("labels.csv")["labels"]

print(pd.Series(y).value_counts())

classes = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z",]
nclasses = len(classes)
xtrain , xtest ,ytrain , ytest = train_test_split(X,y,random_state = 9 , train_size = 3500 , test_size = 500)
xtrainScaled = xtrain / 255.0
xtestScaled = xtest / 255.0
clf = LogisticRegression(solver = 'saga',multi_class = 'multinomial').fit(xtrainScaled , ytrain)

def get_prediction(image):

    impil = Image.open(image)
    imgbw = impil.convert('L')
    imgbwresize = imgbw.resize((28,28),Image.ANTIALIAS)

    pixelfilter = 20
    minpixel = np.percentile(imgbwresize , pixelfilter)
    imginvertedscale = np.clip(imgbwresize - minpixel ,0 ,255)
    maxpixel = np.max(imgbwresize)
    imginvertedscale = np.asarray(imginvertedscale)/maxpixel
    testsample = np.array(imginvertedscale).reshape(1,784)
    testpred = clf.predict(testsample)
    
    return testpred[0]

    