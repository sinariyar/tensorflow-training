#引入套件
import numpy as np
import cv2
import os 
import random
import matplotlib.pyplot as plt
from sklearn import preprocessing



#%%當前的檔案路徑

DIRECTORY = r'C:\Users\user\Desktop\tensorflow\Data_set' #訓練資料集目錄

CATEGORIES = ['cats','dogs'] #類別資料夾

#%%處理圖片資料

IMG_SIZE = 100 #圖片的邊長，用來統一長和寬

data = [] #放置圖片和標籤的data

for category in CATEGORIES:
    
    label = CATEGORIES.index(category)  #類別的順序標籤，從0開始， 0:cat , 1:dogs
    
    folder = os.path.join( DIRECTORY , category )    
    #print(folder)
    # 結果: C:\Users\user\Desktop\tensorflow\Dataset\training_set\cats
    # 結果: C:\Users\user\Desktop\tensorflow\Dataset\training_set\dogs
    
    for img in os.listdir( folder ):
        
        img_path = os.path.join( folder , img )  # 結果: C:\Users\user\Desktop\tensorflow\Dataset\training_set\cats\cat.1.jpg
        
        img_arr = cv2.imread( img_path )  #讀取圖片
        
        img_arr = cv2.resize( img_arr , ( IMG_SIZE , IMG_SIZE ))  #設定長和寬都在100
        
        data.append([ img_arr , label ])  #將整理好的圖片跟標籤等和後放入data陣列

        

random.shuffle(data)  #隨機排序

X = []

Y = []

for features,labels in data:
    
    X.append(features)
    
    Y.append(labels)
    
X = np.array(X)

lb = preprocessing.LabelBinarizer()

Y = lb.fit_transform(Y)

X = X.astype('float32') / 255.0
#X = X/255

X.shape

from keras.utils import np_utils
Y = np_utils.to_categorical(Y)

#print(X.shape)

#print(len(X),len(Y))

#%% 建置訓練模型
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
    
model = Sequential([
    Conv2D(64, (3, 3), input_shape = X.shape[1:] , padding = 'same', activation = 'relu'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    
    Dense(1024 , activation='relu'),
    
    Dense(2, activation='softmax')
])

model.summary()    
    
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])    
    
mhistory = model.fit(X , Y ,epochs=5, validation_split=0.2,batch_size=32 )

def show_train_history(train_acc,test_acc):
    plt.plot(mhistory.history[train_acc])
    plt.plot(mhistory.history[test_acc])
    plt.title('Train History')
    plt.ylabel(train_acc)
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

show_train_history('accuracy','val_accuracy')

show_train_history('loss','val_loss')    

from datetime import datetime

now = datetime.now()

dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")

model.save('./model/%s.h5' %(dt_string)) #儲存訓練模型
    
#%% 圖像辨識
# test_data =[]
# new_model = tf.keras.models.load_model('./model/topPolics4.h5')
# image = cv2.imread(r'C:\Users\user\Desktop\tensorflow\dogscats\sample\valid\dogs\dog.4090.jpg')
# image = cv2.resize(image, (100, 100))
# image = np.array(image)
# image = image.astype('float32')
# test_data.append(image)
# test_data = np.array(test_data, dtype="float") / 255.0
# result = new_model(test_data)
# print('result:', result)
#%%

    
    
    
    
    
    
    
    
    






















































