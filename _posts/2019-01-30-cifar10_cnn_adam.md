---
layout: post
title:  "Convolutional Neural Nets Introduction"
date:  2019-01-30
desc: "A tutorial to build a CNN to learn CIFAR 10 Images"
keywords: "Keras, Deep Learning"
categories: [Ml]
tags: [blog, Keras, Deep Learning]
icon: fa-pencil
---
# Artificial Intelligence Nanodegree

## Convolutional Neural Networks

---

In this notebook, we train a CNN to classify images from the CIFAR-10 database.

### 1. Load CIFAR-10 Database


```python
import keras
from keras.datasets import cifar10

# load the pre-shuffled train and test data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
```How

    Using TensorFlow backend.


### 2. Visualize the First 24 Training Images


```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

fig = plt.figure(figsize=(20,5))
for i in range(36):
    ax = fig.add_subplot(3, 12, i + 1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(x_train[i]))
```


![png](static/images/cifar10/output_3_0.png)


### 3. Rescale the Images by Dividing Every Pixel in Every Image by 255


```python
# rescale [0,255] --> [0,1]
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
```

### 4.  Break Dataset into Training, Testing, and Validation Sets


```python
from keras.utils import np_utils

# one-hot encode the labels
num_classes = len(np.unique(y_train))
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# break training set into training and validation sets
(x_train, x_valid) = x_train[5000:], x_train[:5000]
(y_train, y_valid) = y_train[5000:], y_train[:5000]

# print shape of training set
print('x_train shape:', x_train.shape)

# print number of training, validation, and test images
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print(x_valid.shape[0], 'validation samples')
```

    x_train shape: (45000, 32, 32, 3)
    45000 train samples
    10000 test samples
    5000 validation samples


### 5. Define the Model Architecture


```python
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=3, padding='same', activation='relu',
                        input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax'))

model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_13 (Conv2D)           (None, 32, 32, 16)        448       
    _________________________________________________________________
    max_pooling2d_13 (MaxPooling (None, 16, 16, 16)        0         
    _________________________________________________________________
    conv2d_14 (Conv2D)           (None, 16, 16, 32)        4640      
    _________________________________________________________________
    max_pooling2d_14 (MaxPooling (None, 8, 8, 32)          0         
    _________________________________________________________________
    conv2d_15 (Conv2D)           (None, 8, 8, 64)          18496     
    _________________________________________________________________
    max_pooling2d_15 (MaxPooling (None, 4, 4, 64)          0         
    _________________________________________________________________
    conv2d_16 (Conv2D)           (None, 4, 4, 128)         73856     
    _________________________________________________________________
    max_pooling2d_16 (MaxPooling (None, 2, 2, 128)         0         
    _________________________________________________________________
    dropout_7 (Dropout)          (None, 2, 2, 128)         0         
    _________________________________________________________________
    flatten_4 (Flatten)          (None, 512)               0         
    _________________________________________________________________
    dense_7 (Dense)              (None, 256)               131328    
    _________________________________________________________________
    dropout_8 (Dropout)          (None, 256)               0         
    _________________________________________________________________
    dense_8 (Dense)              (None, 10)                2570      
    =================================================================
    Total params: 231,338.0
    Trainable params: 231,338.0
    Non-trainable params: 0.0
    _________________________________________________________________


### 6. Compile the Model


```python
# compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
```

### 7. Train the Model

How
```python
from keras.callbacks import ModelCheckpoint   

# train the model
checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose=1,
                               save_best_only=True)
hist = model.fit(x_train, y_train, batch_size=32, epochs=15,
          validation_data=(x_valid, y_valid), callbacks=[checkpointer],
          verbose=2, shuffle=True)
```

    Train on 45000 samples, validate on 5000 samples
    Epoch 1/15
    Epoch 00000: val_loss improved from inf to 0.86286, saving model to model.weights.best.hdf5
    67s - loss: 0.8959 - acc: 0.6962 - val_loss: 0.8629 - val_acc: 0.7140
    Epoch 2/15
    Epoch 00001: val_loss did not improve
    67s - loss: 0.8065 - acc: 0.7232 - val_loss: 0.9271 - val_acc: 0.7000
    Epoch 3/15
    Epoch 00002: val_loss improved from 0.86286 to 0.84090, saving model to model.weights.best.hdf5
    66s - loss: 0.7559 - acc: 0.7383 - val_loss: 0.8409 - val_acc: 0.7116
    Epoch 4/15
    Epoch 00003: val_loss improved from 0.84090 to 0.78299, saving model to model.weights.best.hdf5
    66s - loss: 0.7184 - acc: 0.7494 - val_loss: 0.7830 - val_acc: 0.7384
    Epoch 5/15
    Epoch 00004: val_loss did not improve
    66s - loss: 0.6893 - acc: 0.7604 - val_loss: 0.7959 - val_acc: 0.7274
    Epoch 6/15
    Epoch 00005: val_loss did not improve
    66s - loss: 0.6586 - acc: 0.7705 - val_loss: 0.8175 - val_acc: 0.7264
    Epoch 7/15
    Epoch 00006: val_loss did not improve
    67s - loss: 0.6346 - acc: 0.7774 - val_loss: 0.7837 - val_acc: 0.7416
    Epoch 8/15
    Epoch 00007: val_loss did not improve
    67s - loss: 0.6126 - acc: 0.7846 - val_loss: 0.8139 - val_acc: 0.7322
    Epoch 9/15
    Epoch 00008: val_loss did not improve
    66s - loss: 0.5896 - acc: 0.7937 - val_loss: 0.8309 - val_acc: 0.7312
    Epoch 10/15
    Epoch 00009: val_loss improved from 0.78299 to 0.75832, saving model to model.weights.best.hdf5
    67s - loss: 0.5599 - acc: 0.8003 - val_loss: 0.7583 - val_acc: 0.7498
    Epoch 11/15
    Epoch 00010: val_loss did not improve
    67s - loss: 0.5517 - acc: 0.8061 - val_loss: 0.7844 - val_acc: 0.7472
    Epoch 12/15
    Epoch 00011: val_loss did not improve
    67s - loss: 0.5363 - acc: 0.8096 - val_loss: 0.8014 - val_acc: 0.7500
    Epoch 13/15
    Epoch 00012: val_loss did not improvestatic/images/cifar10/output_3_0.png

    67s - loss: 0.5154 - acc: 0.8167 - val_loss: 0.8286 - val_acc: 0.7378
    Epoch 14/15
    Epoch 00013: val_loss did not improve
    66s - loss: 0.5058 - acc: 0.8185 - val_loss: 0.7958 - val_acc: 0.7554
    Epoch 15/15
    Epoch 00014: val_loss did not improve
    66s - loss: 0.4890 - acc: 0.8264 - val_loss: 0.7996 - val_acc: 0.7462


### 8. Load the Model with the Best Validation Accuracy


```python
# load the weights that yielded the best validation accuracy
model.load_weights('model.weights.best.hdf5')
```

### 9. Calculate Classification Accuracy on Test Set


```python
# evaluate and print test accuracy
score = model.evaluate(x_test, y_test, verbose=0)
print('\n', 'Test accuracy:', score[1])
```


     Test accuracy: 0.7361


### 10. Visualize Some Predictions

This may give you some insight into why the network is misclassifying certain objects.


```python
# get predictions on the test set
y_hat = model.predict(x_test)

# define text labels (source: https://www.cs.toronto.edu/~kriz/cifar.html)
cifar10_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
```


```python
# plot a random sample of test images, their predicted labels, and ground truth
fig = plt.figure(figsize=(64, 64))
for i, idx in enumerate(np.random.choice(x_test.shape[0], size=16, replace=False)):
    ax = fig.add_subplot(16, 1, i + 1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(x_test[idx]))
    pred_idx = np.argmax(y_hat[idx])
    true_idx = np.argmax(y_test[idx])
    ax.set_title("{} ({})".format(cifar10_labels[pred_idx], cifar10_labels[true_idx]),
                 color=("green" if pred_idx == true_idx else "red"))
```


![png](static/images/cifar10/output_20_0.png)



```python

```
