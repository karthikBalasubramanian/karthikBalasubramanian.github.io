---
layout: post
title:  "First Kaggle Submission - Deep learning"
date:  2017-01-07
desc: "Introduction to Convolutional Neural Networks"
keywords: "Machine Learning,Deep Learning"
categories: [Ml]
tags: [blog, Machine learning, Deep Learning]
icon: fa-pencil
---





# Audience

  This tutorial is intended to deep learning aspirants. CLI coding, Python and an understanding about Artificial neural networks is necessary for the reader to comprehend the tutorial. 

 [Competition link](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/)
 
 Task: Use pre-trained VGG16 (Visual Geometry Group) model to classify between Cats and Dogs.
 
### To-Do

 * Install Kaggle-CLI
 * configure username, password, competiton-id
 * download data
 * segregate data in to train, valid, test sets. Also have a small sample set and test algo with sample set.
 * Sample set contains sample train, sample valid
 * Fine tune VGG 16 to classify the data set's requirements.
 * submit to Kaggle.
 

### Kaggle CLI Installation and configuration

        sudo pip install kaggle-cli
    
   Configuring cli to a username, password and competition. This config file is available in 
   
        ~/.kaggle-cli/config 

   you can edit this file as you start your new competition.
   
        kg config -g -u 'username' -p 'password' -c 'dogs-vs-cats-redux-kernels-edition'
   
   finally create directory called **redux** under data as **data/redux** and download the competition data. Unzip the files. Before downloading accept to competition rules in the website 
   
        kg download
    


```python
%matplotlib inline
import utils; reload(utils)
from utils import *
```

    Using gpu device 0: Tesla K80 (CNMeM is disabled, cuDNN 5103)
    /home/ubuntu/anaconda2/lib/python2.7/site-packages/theano/sandbox/cuda/__init__.py:600: UserWarning: Your cuDNN version is more recent than the one Theano officially supports. If you see any problems, try updating Theano or downgrading cuDNN to version 5.
      warnings.warn(warn)
    Using Theano backend.



```python
batch_size=64
```

## Segregate Data


```python
current_dir = os.getcwd()
LESSON_HOME_DIR = current_dir
DATA_HOME_DIR = current_dir+'/data/redux'
```


```python
%cd $DATA_HOME_DIR
```

    /home/ubuntu/courses/deeplearning1/nbs/data/redux



```python
%cd train
```

    /home/ubuntu/courses/deeplearning1/nbs/data/redux/train



```python
%mkdir $DATA_HOME_DIR/valid
# also create sample train and sample test
%mkdir $DATA_HOME_DIR/sample
%mkdir $DATA_HOME_DIR/sample/train
%mkdir $DATA_HOME_DIR/sample/valid
```


```python
%ls -l | wc -l
```

    25001



```python
# select all the file paths ending with *jpg
g = glob('*.jpg')
# shuffle file paths
shuf = np.random.permutation(g)
shuf[:5]
```




    array(['dog.7046.jpg', 'cat.11353.jpg', 'cat.9465.jpg', 'dog.4110.jpg', 'dog.1306.jpg'], 
          dtype='|S13')




```python
# rename 2000 random files
for i in range(2000):
    os.rename(shuf[i],DATA_HOME_DIR+'/valid/'+shuf[i])
```


```python
# check number of files in train now
%ls -l | wc -l
```

    23001



```python
# lets try to make our model run for a sample. copy files from train and valid to sample/train, sample/valid

from shutil import copyfile
```


```python
# we are in /data/redux/train
g = glob('*.jpg')
shuf = np.random.permutation(g)
for i in range(200):
    copyfile(shuf[i],DATA_HOME_DIR+'/sample/train/'+shuf[i])
```


```python
%cd $DATA_HOME_DIR/valid
```

    /home/ubuntu/courses/deeplearning1/nbs/data/redux/valid



```python
%ls -l | wc -l
```

    2001



```python
g = glob('*.jpg')
shuf = np.random.permutation(g)
for i in range(50):
    copyfile(shuf[i],DATA_HOME_DIR+'/sample/valid/'+shuf[i])
```


```python
%cd $DATA_HOME_DIR/sample/train
%ls -l | wc -l
```

    /home/ubuntu/courses/deeplearning1/nbs/data/redux/sample/train
    201



```python
%cd $DATA_HOME_DIR/sample/valid
%ls -l | wc -l
```

    /home/ubuntu/courses/deeplearning1/nbs/data/redux/sample/valid
    51



```python
%cd $DATA_HOME_DIR
```

    /home/ubuntu/courses/deeplearning1/nbs/data/redux


Keras require that the images of same classification label should be in the same folder. So lets put all cat images and dog images in one folder.


```python
%mkdir $DATA_HOME_DIR/train/cats
%mkdir $DATA_HOME_DIR/train/dogs
%mkdir $DATA_HOME_DIR/valid/cats
%mkdir $DATA_HOME_DIR/valid/dogs
%mkdir $DATA_HOME_DIR/sample/train/cats
%mkdir $DATA_HOME_DIR/sample/train/dogs
%mkdir $DATA_HOME_DIR/sample/valid/cats
%mkdir $DATA_HOME_DIR/sample/valid/dogs
```


```python
%mv $DATA_HOME_DIR/train/cat*.jpg $DATA_HOME_DIR/train/cats
%mv $DATA_HOME_DIR/train/dog*.jpg $DATA_HOME_DIR/train/dogs
%mv $DATA_HOME_DIR/valid/cat*.jpg $DATA_HOME_DIR/valid/cats
%mv $DATA_HOME_DIR/valid/dog*.jpg $DATA_HOME_DIR/valid/dogs
%mv $DATA_HOME_DIR/sample/train/cat*.jpg $DATA_HOME_DIR/sample/train/cats
%mv $DATA_HOME_DIR/sample/train/dog*.jpg $DATA_HOME_DIR/sample/train/dogs
%mv $DATA_HOME_DIR/sample/valid/cat*.jpg $DATA_HOME_DIR/sample/valid/cats
%mv $DATA_HOME_DIR/sample/valid/dog*.jpg $DATA_HOME_DIR/sample/valid/dogs
```


```python
%ls $DATA_HOME_DIR/train
```

    [0m[01;34mcats[0m/  [01;34mdogs[0m/


##  The Deep Learning! Lets see how deep it is..

 For this submission, My objective is to just run a Deep learning program. All I am trying to know are the wrappers that solves the classification problem via Deep learning.
 
Before coding, I wanted to understand the problem that I am trying to solve. Here is the gist I came up with to understand the problem better.


## Convolutional Neural Networks

  [![nn_example-624x218.png](https://s6.postimg.org/x0gzu8twh/nn_example_624x218.png)](https://postimg.org/image/dvdqkhf8d/)

   The above is a working heuristic of Convolutional neural network. This [article](https://devblogs.nvidia.com/parallelforall/accelerate-machine-learning-cudnn-deep-neural-network-library/) is a brilliant starting point to know about evolution from fully-connected to Convolutional networks. It also justfies the need for GPUs in Machine learning and gives a brief introduction to CuDNN.

   In nutshell, we are trying to simulate the way our own neural network works. To put in perspective, A billion neurons works in parallel to understand an image like Cat which has been previously identified and stored in our memory. This needs lot of computational power. GPUs give this computational power to process lot of data and adjust tons of weight to fit the data.

## CuDNN

  NVIDIA cuDNN is a GPU-accelerated library of primitives for DNNs.  It provides tuned implementations of routines that arise frequently in DNN applications, such as:

    * Convolution
    * Pooling
    * SoftMax
    * Neural activation functions

The overall machine learning framework looks something of this sort. 

[![DeepLearingArchitecture.png](https://s6.postimg.org/nq0acytsx/Deep_Learing_Architecture.png)](https://postimg.org/image/gmsexcod9/)
 
 
 We won't be delving into the deeper layers of the framework. In this tutorial we will try to figure out how to make Keras communicate with the below layers.
 
 Important functions and Data Structures of Keras:
 
     * Model - Data structure for modelling the CNN
     
     * Sequential - A type of data structure
     
     * input_dim - the dimension of the input tensor/vector.
     
     * input_batch_dim - the dimension of bacth of inputs to be sent to Keras
     
     * compile - A function which accepts objective function, optimization function and metrics to be performed in the neural network. We should specify compile function before training the model. The compile function defines the learning process for the model.
     
     * merge - Merge two Sequential models into one.There are many ways two sequential model output tensors can be merged. Please refer Keras documentation. The merged data can again modeled as a sequential model.
     
     * Objective functions, Optimizers and metrics are of different types. Please see the documentation.
     
     * fit - Once you specify the model, learning process we can train the model for known labels.
     
     * train_on_batch - We can also train on a particular batch
     
     * Evaluate - function to evaluate the performance
     
     * nb_epoch - number of iterations in the training process.
     
 If we know the working of all these functions, we have covered pretty much of keras. Beauty of keras is that it hids all the mathematical complexities involved in developing a deep convolutional neural network.
 

I am using a VGG16 python class. I have documented to an extent of what every function does.
 

**Build and train a VGG Model. Finetune it after building**


```python
from vgg16 import Vgg16
```


```python
vgg = Vgg16()
```


```python
batches = vgg.get_batches(batch_size=batch_size,path=DATA_HOME_DIR+"/train")
val_batches = vgg.get_batches(batch_size=batch_size*2,path=DATA_HOME_DIR+"/valid")
vgg.finetune(batches)
```

    Found 23000 images belonging to 2 classes.
    Found 2000 images belonging to 2 classes.



```python
vgg.fit(batches,val_batches,nb_epoch=1)
```

    Epoch 1/1
    23000/23000 [==============================] - 615s - loss: 0.0972 - acc: 0.9757 - val_loss: 0.0698 - val_acc: 0.9810



```python
%mkdir $DATA_HOME_DIR/results
vgg.model.save_weights(DATA_HOME_DIR+"/results/ft1.h5")
```

**predict the unlabled test images and assign a class**


```python
%mkdir $DATA_HOME_DIR/test/unknown
%mv $DATA_HOME_DIR/test/*.jpg $DATA_HOME_DIR/test/unknown
# We have an namesake folder introduced because thats the folder format of keras test method.
%ls $DATA_HOME_DIR/test/unknown
```

```python
%ls $DATA_HOME_DIR/test
```

    [0m[01;34munknown[0m/



```python
batches,pred = vgg.test(path=DATA_HOME_DIR+"/test",batch_size=batch_size*2)
```

    Found 12500 images belonging to 1 classes.


# Submission to Kaggle

 Now we have found the classes for the test set, we can submit the solution to Kaggle. But Kaggle expects the solution in a format. Lets see how it looks.


```python
%ls
```

    [0m[01;34mresults[0m/  sample_submission.csv  [01;31mtest.zip[0m  [01;31mtrain.zip[0m
    [01;34msample[0m/   [01;34mtest[0m/                  [01;34mtrain[0m/    [01;34mvalid[0m/



```python
!head sample_submission.csv
```

    id,label
    1,0.5
    2,0.5
    3,0.5
    4,0.5
    5,0.5
    6,0.5
    7,0.5
    8,0.5
    9,0.5


Here Id is the name of the unknown class image file and label gives the probability of the image being in a class. This is our submission format.


```python
type(batches),type(pred)
```




    (keras.preprocessing.image.DirectoryIterator, numpy.ndarray)




```python
# get filenames from directory iterator

filenames = batches.filenames
type(filenames)
```




    list




```python
filenames[:5]
```




    ['unknown/9292.jpg',
     'unknown/12026.jpg',
     'unknown/9688.jpg',
     'unknown/4392.jpg',
     'unknown/779.jpg']




```python
save_array(fname=DATA_HOME_DIR+"/results/filenames.dat",arr=filenames)
save_array(fname=DATA_HOME_DIR+"/results/test_preds.dat",arr=pred)
```


```python
# load from local here after
filenames=load_array(fname=DATA_HOME_DIR+"/results/filenames.dat")
pred = load_array(fname=DATA_HOME_DIR+"/results/test_preds.dat")
```


```python
# use same probabilities but penalize super confident predictions
is_dog = np.clip(a=pred[:,1],a_min=0.025,a_max=0.975)
```


```python
ids = [int(f[8:f.find(".")]) for f in filenames]
```


```python
ids[:5]
```




    [9292, 12026, 9688, 4392, 779]




```python
len(ids),len(is_dog)
```




    (12500, 12500)




```python
sub = np.column_stack((ids,is_dog))
```


```python
sub[:5]
```




    array([[  9.2920e+03,   2.5000e-02],
           [  1.2026e+04,   4.4612e-01],
           [  9.6880e+03,   2.5000e-02],
           [  4.3920e+03,   2.5000e-02],
           [  7.7900e+02,   9.7500e-01]])




```python
np.savetxt(comments='',fmt='%d,%.5f',header='id,label',X=sub,fname="sub_karthik.csv")
```


```python
%ls
```

    [0m[01;34mresults[0m/  sample_submission.csv  [01;34mtest[0m/     [01;34mtrain[0m/     [01;34mvalid[0m/
    [01;34msample[0m/   sub_karthik.csv        [01;31mtest.zip[0m  [01;31mtrain.zip[0m



```python
from IPython.display import FileLink
FileLink('sub_karthik.csv')
```




<a href='sub_karthik.csv' target='_blank'>sub_karthik.csv</a><br>




```python
!head "sub_karthik.csv"
```

    id,label
    9292,0.02500
    12026,0.44612
    9688,0.02500
    4392,0.02500
    779,0.97500
    2768,0.97500
    2399,0.02500
    12225,0.75026
    10947,0.02500



```python
!kg submit 'sub_karthik.csv'
```

    Starting new HTTPS connection (1): www.kaggle.com
    


# Fantastic We are done. 

We can delve deep into Categorical cross entropy loss and internals of Keras and theano in the coming lessons!
