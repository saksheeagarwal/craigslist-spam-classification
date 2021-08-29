
#importing Packages and Library
import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Flatten,GlobalAveragePooling2D,Conv2D,MaxPooling2D,Activation,Dropout
from sklearn.metrics import classification_report, log_loss, accuracy_score
from sklearn.model_selection import train_test_split


#importing the directory of all the files

dir_all = glob.glob('/Users/harsh/Desktop/Purdue/Mod4/archive/*')


#Saving the address for each boat type

dir0 = '/Users/harsh/Desktop/Purdue/Mod4/archive/Not Boat'
dir1 = '/Users/harsh/Desktop/Purdue/Mod4/archive/inflatable boat'
dir2 = '/Users/harsh/Desktop/Purdue/Mod4/archive/gondola'
dir3 = '/Users/harsh/Desktop/Purdue/Mod4/archive/paper boat'
dir4 = '/Users/harsh/Desktop/Purdue/Mod4/archive/kayak'
dir5 = '/Users/harsh/Desktop/Purdue/Mod4/archive/sailboat'
dir6 = '/Users/harsh/Desktop/Purdue/Mod4/archive/ferry boat'
dir7 = '/Users/harsh/Desktop/Purdue/Mod4/archive/freight boat'
dir8 = '/Users/harsh/Desktop/Purdue/Mod4/archive/cruise ship'



#createing list of different classes of baot
classes = ['Not Boat', 'cruise ship', 'ferry boat', 'freight boat', 'gondola', 'inflatable boat', 'kayak', 'paper boat', 'sailboat']


#creating a single file for address of all the images

images_dir= []
for dire in dir_all:
    temp = glob.glob(dire + '/*.jpg')
    for t in temp:
        images_dir.append(t)


# creating a directory list and mapping the class with 0 to 8 int values

directory=[dir0,dir1,dir2,dir3,dir4,dir5,dir6,dir7,dir8]
class_map={ classes[0]:0,classes[1]:1,classes[2]:2,classes[3]:3,classes[4]:4,classes[5]:5,classes[6]:6,classes[7]:7,classes[8]:8 }


#loading all the images with there class as label in tupel

images=[]
count=0
for dirs in classes:
    for file in images_dir:
        if dirs in file:
            image=load_img(file, grayscale=False, target_size=(224,224))
            image=img_to_array(image)
            image=image/255
            images.append([image,count])
    count=count+1


#extracting images and label in different variables

data_0,labels_0=zip(*images)


#ploting to check that image and label is correctly ziped

plt.imshow(data[555]), labels_0[555]


#checking shape

labels_1=to_categorical(labels_0)

data=np.array(data_0)
labels=np.array(labels_1)

print(data.shape)
print(labels.shape)


#spliting training and test dataset.

train_x,test_x,train_y,test_y=train_test_split(data,labels,test_size=0.2,random_state=44)


#checking shape
print(train_x.shape)
print(test_x.shape)
print(train_y.shape)
print(test_y.shape)



#data agumentation using ImageDataGenerator
datagen = ImageDataGenerator(horizontal_flip=True,vertical_flip=True,rotation_range=20,zoom_range=0.2,
                        width_shift_range=0.2,height_shift_range=0.2,shear_range=0.1,fill_mode="nearest")




#loading transferlearning models
model_0 = tf.keras.applications.MobileNetV2(input_shape=(224,224,3),include_top=False,weights='imagenet',pooling='avg')
model_1 = tf.keras.applications.ResNet50(input_shape=(224,224,3),include_top=False,weights='imagenet',pooling='avg')
model_2 = tf.keras.applications.VGG16(input_shape=(224,224,3),include_top=False,weights='imagenet',pooling='avg')
model_3 = tf.keras.applications.DenseNet121(input_shape=(224,224,3),include_top=False,weights='imagenet',pooling='avg')

model_0.trainable = False
model_1.trainable = False
model_2.trainable = False
model_3.trainable = False


#saving inout shape
inputs_0 = model_0.input
inputs_1 = model_1.input
inputs_2 = model_2.input
inputs_3 = model_3.input


# defining a dense layer of 128 neurons and 9 neurons to predict the probabilites of the 9 categories we have in each model.
l_0 = tf.keras.layers.Dense(128, activation='relu')(model_0.output)
o_0 = tf.keras.layers.Dense(9, activation='softmax')(l_0)
l_1 = tf.keras.layers.Dense(128, activation='relu')(model_1.output)
o_1 = tf.keras.layers.Dense(9, activation='softmax')(l_1)
l_2 = tf.keras.layers.Dense(128, activation='relu')(model_2.output)
o_2 = tf.keras.layers.Dense(9, activation='softmax')(l_2)
l_3 = tf.keras.layers.Dense(128, activation='relu')(model_3.output)
o_3 = tf.keras.layers.Dense(9, activation='softmax')(l_3)


#adding layers
model_0 = tf.keras.Model(inputs=inputs_0, outputs=o_0)
model_1 = tf.keras.Model(inputs=inputs_1, outputs=o_1)
model_2 = tf.keras.Model(inputs=inputs_2, outputs=o_2)
model_3 = tf.keras.Model(inputs=inputs_3, outputs=o_3)




#CNN model 
model_5 = Sequential()
model_5.add(Conv2D(128,(5,5), input_shape=(224,224,3), activation='relu'))
model_5.add(MaxPooling2D(2,2))
model_5.add(Conv2D(64,(5,5), activation='relu'))
model_5.add(MaxPooling2D(2,2))
model_5.add(Conv2D(32,(5,5), activation='relu'))
model_5.add(MaxPooling2D(2,2))
model_5.add(Flatten())
model_5.add(Dense(units=512, activation='relu'))
model_5.add(Dense(units=128, activation='relu'))
model_5.add(Dense(units=9, activation='softmax'))


#compiling model with adam as optimizer and metrics as accuracy
model_0.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_1.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_2.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_3.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_5.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



#training the model with 20 epochs
res_0=model_0.fit(datagen.flow(train_x,train_y,batch_size=32),validation_data=(test_x,test_y),epochs=20)
res_1=model_1.fit(datagen.flow(train_x,train_y,batch_size=32),validation_data=(test_x,test_y),epochs=20)
res_2=model_2.fit(datagen.flow(train_x,train_y,batch_size=32),validation_data=(test_x,test_y),epochs=20)
res_3=model_3.fit(datagen.flow(train_x,train_y,batch_size=32),validation_data=(test_x,test_y),epochs=20)
res_5=model_5.fit(datagen.flow(train_x,train_y,batch_size=32),validation_data=(test_x,test_y),epochs=20)



#creaating prediction on test dataset
y_pred_0 = model_0.predict(test_x)
y_pred_1 = model_1.predict(test_x)
y_pred_2 = model_2.predict(test_x)
y_pred_3 = model_3.predict(test_x)
y_pred_5 = model_4.predict(test_x)


#choosing the max probability as the final category
pred_0 = np.argmax(y_pred_0,axis=1)
pred_1 = np.argmax(y_pred_1,axis=1)
pred_2 = np.argmax(y_pred_2,axis=1)
pred_3 = np.argmax(y_pred_3,axis=1)
pred_5 = np.argmax(y_pred_5,axis=1)

lab = np.argmax(test_y,axis=1)

#printing clasiffication reports
from sklearn.metrics import classification_report

print(classification_report(lab,pred_0))
print(classification_report(lab,pred_1))
print(classification_report(lab,pred_2))
print(classification_report(lab,pred_3))
print(classification_report(lab,pred_5))



#saving accuracy in variable and ploting train accuracy
acc0 = res_0.history['accuracy']
acc1 = res_1.history['accuracy']
acc2 = res_2.history['accuracy']
acc3 = res_3.history['accuracy']
acc4 = res_4.history['accuracy']
acc5 = res_5.history['accuracy']

epochs = range(len(acc0))

plt.plot(epochs, acc0, 'r', label='MobileNetV2')
plt.plot(epochs, acc1, 'b', label='ResNet50')
plt.plot(epochs, acc2, 'g', label='VGG16')
plt.plot(epochs, acc3, 'y', label='DenseNet121')
plt.plot(epochs, acc5, 'm', label='Conv2D')

plt.title('Accuracy comparison')
plt.legend(loc=0)
plt.figure()

plt.show()


#saving train loss and ploting a graph.
loss0 = res_0.history['loss']
loss1 = res_1.history['loss']
loss2 = res_2.history['loss']
loss3 = res_3.history['loss']
loss4 = res_4.history['loss']
loss5 = res_5.history['loss']

epochs = range(len(loss0))

plt.plot(epochs, loss0, 'r', label='MobileNetV2')
plt.plot(epochs, loss1, 'b', label='ResNet50')
plt.plot(epochs, loss2, 'g', label='VGG16')
plt.plot(epochs, loss3, 'y', label='DenseNet121')
plt.plot(epochs, loss5, 'm', label='Conv2D')

plt.title('Loss comparison')
plt.legend(loc=0)
plt.figure()

plt.show()




#saving validation accuracy and ploting a grpah
value_acc0 = res_0.history['val_accuracy']
value_acc1 = res_1.history['val_accuracy']
value_acc2 = res_2.history['val_accuracy']
value_acc3 = res_3.history['val_accuracy']
value_acc4 = res_4.history['val_accuracy']
value_acc5 = res_5.history['val_accuracy']

epochs = range(len(acc0))

plt.plot(epochs, value_acc0, 'r', label='MobileNetV2')
plt.plot(epochs, value_acc1, 'b', label='ResNet50')
plt.plot(epochs, value_acc2, 'g', label='VGG16')
plt.plot(epochs, value_acc3, 'y', label='DenseNet121')
plt.plot(epochs, value_acc5, 'm', label='Conv2D')

plt.title('Validation accuracy comparison')
plt.legend(loc=0)
plt.figure()

plt.show()





#saving validation loss and ploting a grpah

val_loss0 = res_0.history['val_loss']
val_loss1 = res_1.history['val_loss']
val_loss2 = res_2.history['val_loss']
val_loss3 = res_3.history['val_loss']
val_loss4 = res_4.history['val_loss']
val_loss5 = res_5.history['val_loss']
epochs = range(len(loss0))

plt.plot(epochs, val_loss0, 'r', label='MobileNetV2')
plt.plot(epochs, val_loss1, 'b', label='ResNet50')
plt.plot(epochs, val_loss2, 'g', label='VGG16')
plt.plot(epochs, val_loss3, 'y', label='DenseNet121')
plt.plot(epochs, val_loss4, 'c', label='EfficientNetB7')
plt.plot(epochs, val_loss5, 'm', label='Conv2D')

plt.title('Validation loss comparison')
plt.legend(loc=0)
plt.figure()

plt.show()



#loading scraped image from craiglist which is type of boat
load_img("/Users/harsh/Desktop/Purdue/Mod4/New folder/boat/z_08I05i_300x300.jpg",target_size=(224,224))




# preprocessing the image
image=load_img("/Users/harsh/Desktop/Purdue/Mod4/New folder/boat/z_08I05i_300x300.jpg",target_size=(224,224))

image=img_to_array(image) 
image=image/255
prediction_image=np.array(image)
prediction_image=np.expand_dims(image, axis=0)


# predicting the class of the image
reverse_class={ 0:classes[0],1:classes[1],2:classes[2],3:classes[3],4:classes[4],5:classes[5],6:classes[6],7:classes[7],8:classes[8] }

def mapper(value):
    return reverse_class[value]

pred_0=model_0.predict(prediction_image)
value_0=np.argmax(pred_0)
move_name=mapper(value_0)
if pred_0.max() > 0.3:
    print("Prediction is {}.".format(move_name))
else:
    print("not a boat")
load_img("/Users/harsh/Desktop/Purdue/Mod4/New folder/boat/z_08I05i_300x300.jpg",target_size=(224,224))




#loading scraped image from craiglist which is type of not boat
load_img('/Users/harsh/Desktop/Purdue/Mod4/New folder/no_boat/y_0t20CI_300x300.jpg',target_size=(224,224))




# preprocessing the image
image=load_img("/Users/harsh/Desktop/Purdue/Mod4/New folder/no_boat/y_0t20CI_300x300.jpg",target_size=(224,224))

image=img_to_array(image) 
image=image/255
prediction_image=np.array(image)
prediction_image=np.expand_dims(image, axis=0)


# predicting the class of the image
reverse_class={ 0:classes[0],1:classes[1],2:classes[2],3:classes[3],4:classes[4],5:classes[5],6:classes[6],7:classes[7],8:classes[8] }

def mapper(value):
    return reverse_class[value]

pred0=model_0.predict(prediction_image)
value0=np.argmax(pred0)
move_name=mapper(value0)
if pred0.max() > 0.3:
    print("Prediction is {}.".format(move_name))
else:
    print("not a boat")


