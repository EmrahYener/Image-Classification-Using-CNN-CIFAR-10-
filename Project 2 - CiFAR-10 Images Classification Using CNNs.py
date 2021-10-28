#!/usr/bin/env python
# coding: utf-8

# # STEP 0: PROBLEM STATEMENT

# - CIFAR-10 is a dataset that consists of several images divided into the following 10 classes: 
#     - Airplanes
#     - Cars 
#     - Birds
#     - Cats
#     - Deer
#     - Dogs
#     - Frogs
#     - Horses
#     - Ships
#     - Trucks
# 
# - The dataset stands for the Canadian Institute For Advanced Research (CIFAR)
# - CIFAR-10 is widely used for machine learning and computer vision applications. 
# - The dataset consists of 60,000 32x32 color images and 6,000 images of each class.
# - Images have low resolution (32x32). 
# - Data Source: https://www.cs.toronto.edu/~kriz/cifar.html
# 

# # STEP #1: IMPORT LIBRARIES/DATASETS

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn 


# In[ ]:


from keras.datasets import cifar10


# In[ ]:



(X_train, y_train) , (X_test, y_test) = cifar10.load_data()


# In[ ]:


X_train.shape


# In[ ]:


X_test.shape


# In[ ]:


y_train.shape


# In[ ]:


y_test.shape


# # STEP #2: VISUALIZE DATA

# In[ ]:


i = 30009
plt.imshow(X_train[i])
print(y_train[i])


# In[ ]:


W_grid = 4
L_grid = 4

fig, axes = plt.subplots(L_grid, W_grid, figsize = (25, 25))
axes = axes.ravel()

n_training = len(X_train)

for i in np.arange(0, L_grid * W_grid):
    index = np.random.randint(0, n_training) # pick a random number
    axes[i].imshow(X_train[index])
    axes[i].set_title(y_train[index])
    axes[i].axis('off')
    
plt.subplots_adjust(hspace = 0.4)


# In[ ]:


n_training


# In[ ]:





# # STEP #3: DATA PREPARATION

# In[ ]:


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


# In[ ]:


number_cat = 10


# In[ ]:


y_train


# In[ ]:


import keras
y_train = keras.utils.to_categorical(y_train, number_cat)


# In[ ]:


y_train


# In[ ]:


y_test = keras.utils.to_categorical(y_test, number_cat)


# In[ ]:


y_test


# In[ ]:


X_train = X_train/255
X_test = X_test/255


# In[ ]:


X_train


# In[ ]:


X_train.shape


# In[ ]:


Input_shape = X_train.shape[1:]


# In[ ]:


Input_shape


# # STEP #4: TRAIN THE MODEL

# In[ ]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard


# In[ ]:


cnn_model = Sequential()
cnn_model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu', input_shape = Input_shape))
cnn_model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))
cnn_model.add(MaxPooling2D(2,2))
cnn_model.add(Dropout(0.4))


cnn_model.add(Conv2D(filters = 128, kernel_size = (3,3), activation = 'relu'))
cnn_model.add(Conv2D(filters = 128, kernel_size = (3,3), activation = 'relu'))
cnn_model.add(MaxPooling2D(2,2))
cnn_model.add(Dropout(0.4))

cnn_model.add(Flatten())

cnn_model.add(Dense(units = 1024, activation = 'relu'))

cnn_model.add(Dense(units = 1024, activation = 'relu'))

cnn_model.add(Dense(units = 10, activation = 'softmax'))


# In[ ]:


cnn_model.compile(loss = 'categorical_crossentropy', optimizer = keras.optimizers.rmsprop(lr = 0.001), metrics = ['accuracy'])


# In[ ]:


history = cnn_model.fit(X_train, y_train, batch_size = 32, epochs = 1, shuffle = True)


# # STEP #5: EVALUATE THE MODEL

# In[ ]:


evaluation = cnn_model.evaluate(X_test, y_test)
print('Test Accuracy: {}'.format(evaluation[1]))


# In[ ]:


predicted_classes = cnn_model.predict_classes(X_test) 
predicted_classes


# In[ ]:


y_test


# In[ ]:


y_test = y_test.argmax(1)


# In[ ]:


y_test


# In[ ]:


L = 7
W = 7
fig, axes = plt.subplots(L, W, figsize = (12, 12))
axes = axes.ravel()

for i in np.arange(0, L*W):
    axes[i].imshow(X_test[i])
    axes[i].set_title('Prediction = {}\n True = {}'.format(predicted_classes[i], y_test[i]))
    axes[i].axis('off')

plt.subplots_adjust(wspace = 1)    


# In[ ]:


from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, predicted_classes)
cm
plt.figure(figsize = (10, 10))
sns.heatmap(cm, annot = True)


# # STEP #6: SAVING THE MODEL

# In[ ]:


import os 
directory = os.path.join(os.getcwd(), 'saved_models')

if not os.path.isdir(directory):
    os.makedirs(directory)
model_path = os.path.join(directory, 'keras_cifar10_trained_model.h5')
cnn_model.save(model_path)


# # STEP #7: IMPROVING THE MODEL WITH DATA AUGMENTATION

# - Image Augmentation is the process of artificially increasing the variations of the images in the datasets by flipping, enlarging, rotating the original images. 
# - Augmentations also include shifting and changing the brightness of the images.

# # STEP 7.1 DATA AUGMENTATION FOR THE CIFAR-10 DATASET

# In[ ]:


import keras
from keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()


# In[ ]:


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


# In[ ]:


X_train.shape


# In[ ]:


n = 8 
X_train_sample = X_train[:n]


# In[ ]:


X_train_sample.shape


# In[ ]:


from keras.preprocessing.image import ImageDataGenerator

# dataget_train = ImageDataGenerator(rotation_range = 90)
# dataget_train = ImageDataGenerator(vertical_flip=True)
# dataget_train = ImageDataGenerator(height_shift_range=0.5)
dataget_train = ImageDataGenerator(brightness_range=(1,3))


dataget_train.fit(X_train_sample)


# In[ ]:


from scipy.misc import toimage

fig = plt.figure(figsize = (20,2))
for x_batch in dataget_train.flow(X_train_sample, batch_size = n):
     for i in range(0,n):
            ax = fig.add_subplot(1, n, i+1)
            ax.imshow(toimage(x_batch[i]))
     fig.suptitle('Augmented images (rotated 90 degrees)')
     plt.show()
     break;


# # STEP 7.2 MODEL TRAINING USING AUGEMENTED DATASET

# In[ ]:


from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
                            rotation_range = 90,
                            width_shift_range = 0.1,
                            horizontal_flip = True,
                            vertical_flip = True
                             )


# In[ ]:


datagen.fit(X_train)


# In[ ]:


cnn_model.fit_generator(datagen.flow(X_train, y_train, batch_size = 32), epochs = 2)


# In[ ]:


score = cnn_model.evaluate(X_test, y_test)
print('Test accuracy', score[1])


# In[ ]:


# save the model
directory = os.path.join(os.getcwd(), 'saved_models')

if not os.path.isdir(directory):
    os.makedirs(directory)
model_path = os.path.join(directory, 'keras_cifar10_trained_model_Augmentation.h5')
cnn_model.save(model_path)

