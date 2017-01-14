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
    

## Visualising results

 This part of the experiment is very important. We want to understand what makes the model predict the images correctly and what makes it predict wrong. In any image classification task, Jeremy Howard advises to follow the underlying steps to make yourself better understand the model and finetune its parameters accordingly. This is a standard template

 * look into few examples of images in random that we got right.
 * Few examples of images in random that we got wrong.
 * Most correct labels in each class classified right. (Images with models assuring highest possibility of belonging to say Cat class when they are actually Cat.)
 * Most incorrect labels in each class classified wrong.(Images with models assuring highest possibility of belonging to say Cat class when they are actually Dog.)
 * Most uncertain labels. (Labels with confusion to incline towards a class or other)

 This time we will be working with validation set. Because the validation set has the correct answers.


```python
# We have already loaded weights ft3.h5
batches_valid,prob_valid = vgg.test(path=DATA_HOME_DIR+"/valid",batch_size=batch_size*2)
```

    Found 2000 images belonging to 2 classes.



```python
type(batches_valid),type(prob_valid)
```




    (keras.preprocessing.image.DirectoryIterator, numpy.ndarray)




```python
filenames_valid = batches_valid.filenames
```

> We already have 98% of accuracy in validation set.


```python
# we are predicting is_dog in competition.
expected_labels = val_batches.classes #0 or 1

#Round our predictions to 0/1 to generate labels
# our_predictions are for probability
our_predictions = prob_valid[:,0]
# our_labels determines the class label.
our_labels = np.round(1-our_predictions)
```


```python
# number of images to view at once
n_view = 4
```


```python
#Helper function to plot images by index in the validation set
#Plots is a helper function in utils.py
def plots_idx(idx, titles=None):
    plots([image.load_img(DATA_HOME_DIR +"/valid/"+ filenames_valid[i]) for i in idx], titles=titles)
```

1. look into few examples of images in random that we got right.


```python
# we are getting zeroth column because its a tuple.
random_correct_labels  = np.where(expected_labels==our_labels)[0]
random_correct_labels.shape
```




    (1971,)




```python
# check_accuracy
len(random_correct_labels)/float(len(expected_labels))
# adhering to our model accuracy!
```




    0.9855




```python
idx = permutation(random_correct_labels[:n_view])
# titles are nothing but probabilities of being a cat.
plots_idx(idx=idx,titles=our_predictions[idx])
```


[![output_90_0.png](https://s6.postimg.org/3w69ko3tt/output_90_0.png)](https://postimg.org/image/9kckbk865/)


#2 Few Incorrect labels


```python
random_incorrect_labels  = np.where(expected_labels!=our_labels)[0]
random_incorrect_labels.shape
```




    (29,)




```python
idx = permutation(random_incorrect_labels[:n_view])
# titles are nothing but probabilities of being a cat.
plots_idx(idx=idx,titles=our_predictions[idx])
```


[![output_93_0.png](https://s6.postimg.org/tgd54ueld/output_93_0.png)](https://postimg.org/image/squcshe1p/)


we have to interpret these images. These images illustrates how trivial the task of classification to humans, but are complex for computers to learn.

1. the given image is labeled as cat. But probability of being a cat is as close to 6X10^-8. This attributes to the concept of Occlusion.
2. the second image is a cat. But labled as dog.This corresponds to the concept of Deformation.
3. Third image has the same problem as first. it is labeled as Dog. But its actually a cat.
4. Fourth image corresponds to the concept of Background clutter. To get better Idea, see the image below.

[![challenges.jpg](https://s6.postimg.org/71m9x4x29/challenges.jpg)](https://postimg.org/image/a8gtgrhi5/)


#3a The images we most confident were cats, and are actually cats


```python
# when our_lables are 0, we have predicted cat.
# this statement only get the indexes of labels that are correctly predicted as cat.
correct_cats = np.where((our_labels==0) & (our_labels==expected_labels))[0]
print "Found %d confident correct cats labels" % len(correct_cats)
# argsort sorts the indexes based on the value in ascending order.
# example np.argsort([3,1,2]) = [1,2,0]
# we then reverse it.
# most_correct_cats gives indexes of most correct cat lables. [::-1] is used to get the indexes in descending order
# put our prediction indexes in correct cats. It gives the indexes of most correct cats.
# get our predictions of correct cat and get the most correct cat
most_correct_cats = np.argsort(our_predictions[correct_cats])[::-1][:n_view]
plots_idx(correct_cats[most_correct_cats], our_predictions[correct_cats][most_correct_cats])
```

    Found 979 confident correct cats labels



[![output_96_1.png](https://s6.postimg.org/u75vamgyp/output_96_1.png)](https://postimg.org/image/8xi8zs0nx/)


#3b. The images we most confident were dogs, and are actually dogs



```python
# same as above
correct_dogs = np.where((our_labels==1) & (our_labels==expected_labels))[0]
print "Found %d confident correct dogs labels" % len(correct_dogs)
most_correct_dogs = np.argsort(our_predictions[correct_dogs])[:n_view]
plots_idx(correct_dogs[most_correct_dogs], our_predictions[correct_dogs][most_correct_dogs])
```

    Found 992 confident correct dogs labels



[![output_98_1.png](https://s6.postimg.org/iw37m9a3l/output_98_1.png)](https://postimg.org/image/bsvc6n4nx/)


```python
#4a. The images we were most confident were cats, but are actually dogs
# label 0 is cat. get labels which we predicted as cat but are actually dogs.
incorrect_cats = np.where((our_labels==0) & (our_labels!=expected_labels))[0]
print "Found %d incorrect cats" % len(incorrect_cats)
# if there are any, follow same steps as above cells
if len(incorrect_cats):
    most_incorrect_cats = np.argsort(our_predictions[incorrect_cats])[::-1][:n_view]
    plots_idx(incorrect_cats[most_incorrect_cats], our_predictions[incorrect_cats][most_incorrect_cats])
```

    Found 19 incorrect cats



[![output_99_1.png](https://s6.postimg.org/bu5a026ht/output_99_1.png)](https://postimg.org/image/7xry42li5/)



```python
#4b. The images we were most confident were dogs, but are actually cats
# same as above
incorrect_dogs = np.where((our_labels==1) & (our_labels!=expected_labels))[0]
print "Found %d incorrect dogs" % len(incorrect_dogs)
if len(incorrect_dogs):
    most_incorrect_dogs = np.argsort(our_predictions[incorrect_dogs])[:n_view]
    plots_idx(incorrect_dogs[most_incorrect_dogs], our_predictions[incorrect_dogs][most_incorrect_dogs])
```

    Found 10 incorrect dogs



[![output_100_1.png](https://s6.postimg.org/pp3kij0wx/output_100_1.png)](https://postimg.org/image/l37ga6fdp/)


```python
#5. The most uncertain labels (ie those with probability closest to 0.5).
# 0.5 -0.5 becomes 0. 1-0.5 becomes 0.5. 0-0.5 becomes -0.5 absolute of these values becomes positive.
# argsort of these values brings in most uncertain lables first.
most_uncertain = np.argsort(np.abs(our_predictions-0.5))
plots_idx(most_uncertain[:n_view], our_predictions[most_uncertain])
```


[![output_101_0.png](https://s6.postimg.org/xvvka3qzl/output_101_0.png)](https://postimg.org/image/4gpw13mfx/)



```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(expected_labels, our_labels)
cm
```




    array([[979,  10],
           [ 19, 992]])




```python
plot_confusion_matrix(cm, batches_valid.class_indices)
```

    [[979  10]
     [ 19 992]]



[![output_103_1.png](https://s6.postimg.org/bl7pa4tpd/output_103_1.png)](https://postimg.org/image/h9e010y1p/)


# Fantastic We are done. 

We can delve deep into Categorical cross entropy loss and internals of Keras and theano in the coming lessons!
