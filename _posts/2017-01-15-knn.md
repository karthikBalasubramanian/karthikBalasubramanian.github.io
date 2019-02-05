---
layout: post
title:  "Image Classification"
date:  2017-01-15
desc: "Road to Convolutional Neural Networks"
keywords: "Machine Learning,Deep Learning"
categories: [Ml]
tags: [blog, Machine learning, Deep Learning]
icon: fa-pencil
---


# Audience

  This tutorial is intended to Machine/deep learning aspirants. CLI coding, Python and an understanding about Classification as a problem is necessary for the reader to comprehend the tutorial.

  The Tutorial introduces the reader to the problem of Image classification and explains why simple baseline techniques like K Nearest Neighbor fails for high dimensional classification problems. Tutorial also develops foundational knowledge to understand complex techniques which will be dealt in the coming weeks.



# Road to Convolutional Neural Networks

### Take - 1 Classification explained using Nearest Neighbor

> Note : These tutorials are created from Stanford's course on [Convolutional Neural network](http://cs231n.github.io/). Please read through the material for better understanding. Some text are borrowed from the same as it is better.

Task Example:
    
  [![classify.png](https://s6.postimg.org/v433s0dpd/classify.png)](https://postimg.org/image/3thsk3asd/)


   As shown in the image, keep in mind that to a computer an image is represented as one large 3-dimensional array of numbers. In this example, the cat image is 248 pixels wide, 400 pixels tall, and has three color channels Red,Green,Blue (or RGB for short). Therefore, the image consists of **248 X 400 X 3** numbers, or a total of 297,600 numbers. Each number is an integer that ranges from 0 (black) to 255 (white). Our task is to turn this quarter of a million numbers into a single label, such as “cat”. We will be using [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html) dataset which consists of 60000 **32X32** images of 10 different classes. 50000 images, 5000 from each class would be in the training set and 10000 images 1000 from each class would be in test set.
   

As our first approach, we will develop what we call a **Nearest Neighbor Classifier**. This classifier has nothing to do with Convolutional Neural Networks and it is very rarely used in practice, but it will allow us to get an idea about the basic approach to an image classification problem.

How do we compare images?

   The nearest neighbor classifier will take a test image, compare it to every single one of the training images, and predict the label of the closest training image. While will effectively compare two **32X32X3** images, pixel by pixel and add up all the differences.In other words, given two images and representing them as vectors **I1**, **I2**, a reasonable choice for comparing them might be the **L1** distance:
   
   [![l1distance.png](https://s6.postimg.org/pk4v5yzxd/l1distance.png)](https://postimg.org/image/bqgigx7bx/)
   

 If we try to find the distance between **4X4** images, assuming it is from single color code (Red,Blue or green) it looks something like this. First Subtract element wise and then add all differences.
 
 [![nneg.jpg](https://s6.postimg.org/usuu6qc0x/nneg.jpg)](https://postimg.org/image/i1go0828t/)
   

### Downloading and pre-processing the dataset

Steps:

```
~ mkdir data 
~ cd data
~ wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
~ mkdir cifar10
~ tar -xvf cifar-10-python.tar.gz -C cifar10
```

There are **7** files in the cifar10 folder. All are python pickled object files. We have to unpickle them and use it. **5** batch file indicates **5X10000** images or 50000 training images. These are labeled batches. Each batch is a dictionary of Data (key) and label (value)

1. Data : **10000X3072** matrix. There are 10000 images. Each row in the matrix is an image. Each image is of size **32X32**. Also remember that they are *colored images*. We have three color schemes *R G B*. So **32X32X3**. Therefore each image is a vector of size **3072**. The first **1024** entries contain the red channel values, the next **1024** the green, and the final **1024** the blue. The image is stored in row-major order, so that the first **32** entries of the array are the red channel values of the first row of the image.

2. Lables : A list of 10000 numbers with a range of 0-9.

Lets convert these objects into numbers!


   







```python
import os
current_dir = os.getcwd()
DATA_HOME_DIR = current_dir+'/data/cifar10'
```


```python
%ls $DATA_HOME_DIR
```

    batches.meta  data_batch_2  data_batch_4  readme.html
    data_batch_1  data_batch_3  data_batch_5  test_batch



```python
# writing a routine to unpickle dictionary object.
import cPickle
def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict
```


```python
batch_1_dict = unpickle(DATA_HOME_DIR+"/data_batch_1")
```


```python
batch_1_dict.keys()
```




    ['data', 'labels', 'batch_label', 'filenames']




```python
batch_1_dict['filenames'][:5]
```




    ['leptodactylus_pentadactylus_s_000004.png',
     'camion_s_000148.png',
     'tipper_truck_s_001250.png',
     'american_elk_s_001521.png',
     'station_wagon_s_000293.png']




```python
batch_1_dict['data'].shape
```




    (10000, 3072)




```python
type(batch_1_dict),len(batch_1_dict['labels']), batch_1_dict['labels'][:5]
```




    (dict, 10000, [6, 9, 9, 4, 1])




```python
batch_1_dict['batch_label']
```




    'training batch 1 of 5'




```python
# likewise getting all training batches
batch_2_dict = unpickle(DATA_HOME_DIR+"/data_batch_2")
batch_3_dict = unpickle(DATA_HOME_DIR+"/data_batch_3")
batch_4_dict = unpickle(DATA_HOME_DIR+"/data_batch_4")
batch_5_dict = unpickle(DATA_HOME_DIR+"/data_batch_5")
```


```python
# building training set matrix. 50000X3072

Y_tr = [batch_1_dict['labels'],batch_2_dict['labels'],batch_3_dict['labels'],batch_4_dict['labels'],batch_5_dict['labels']]
# converting list of lists to list
Y_tr = [item for sublist in Y_tr for item in sublist]
len(Y_tr)
```




    50000




```python
import numpy as np

X_tr = np.concatenate((batch_1_dict['data'],batch_2_dict['data']),axis=0)
X_tr = np.concatenate((X_tr,batch_3_dict['data']),axis=0)
X_tr = np.concatenate((X_tr,batch_4_dict['data']),axis=0)
X_tr = np.concatenate((X_tr,batch_5_dict['data']),axis=0)
X_tr.shape
```




    (50000, 3072)




```python
# getting test data
batch_test =  unpickle(DATA_HOME_DIR+"/test_batch")
batch_test.keys()
```




    ['data', 'labels', 'batch_label', 'filenames']




```python
len(batch_test['labels']),batch_test['labels'][:5],batch_test['data'].shape,batch_test['batch_label']
```




    (10000, [3, 8, 8, 0, 6], (10000, 3072), 'testing batch 1 of 1')




```python
X_te =  batch_test['data']
Y_te = batch_test['labels']
```


```python
# lets unpickle batches.meta
meta_file = unpickle(DATA_HOME_DIR+"/batches.meta")
meta_file.keys()
```




    ['num_cases_per_batch', 'label_names', 'num_vis']




```python
meta_file['num_cases_per_batch'],meta_file['label_names'],meta_file['num_vis']
# label_names denotes the classes. 0 = airplane, 1 = automobile. etc. till 9
```




    (10000,
     ['airplane',
      'automobile',
      'bird',
      'cat',
      'deer',
      'dog',
      'frog',
      'horse',
      'ship',
      'truck'],
     3072)




```python
# Summarizing
X_tr.shape,len(Y_tr),X_te.shape,len(Y_te)
```




    ((50000, 3072), 50000, (10000, 3072), 10000)



Now Data is ready. Lets understand How nearest neighbor works.Our tutorial will cover following aspects.

* The Nearest Neighbor approach
* K- Nearest Neighbor approach
* Generalisation, Instance based algorithms
* Basic problems
* Solutions : Validation, Cross-Validation
* summary


 

### The Nearest Neighbor Approach

   As I have discussed earlier, Nearest neighbor approach decides the label of an instance (in our case a Picture) and decides whether It belongs to a specific class based on the shortest distance between itself and an image from that class. Our metric for evaluation is accuracy. Any learning algorithm (supervised) will have two phases A train phase. A model is developed from the training examples and its corresponding labels. After training task, We can predict the effectiveness of the model using a small set of hold out labels. This is called a validation set and effectiveness of the model can be interpreted as the functions and that a model can represent. We then predict labels of the test set which does not have labels usually. We can finally decide that a model is so and so percent accurate based on the accuracy metric. This is a standard API of any Machine learning task. Lets start with a very simple Nearest neighbor model which does not have a validation set.


```python
# Nearesr neighbor algorithm with L1 distance.

class NearestNeighbor(object):
    
    def __init__(self):
        pass
    
    def train(self,X_tr, Y_tr):
        """
           X_tr : A matrix of N images of D pixels each. N X D
           Y_tr : An array or list of N labels.
        """
        
        self.X_tr = X_tr
        self.Y_tr = Y_tr
    
    def predict(self,X_te):
        """
          X_te :  A matrix of N images of D pixels each. NXD.
          returns a list or numpy array of predicted labels of length N.
        """
        
        predicted_labels =  np.zeros(X_te.shape[0],dtype=self.Y_tr.dtype)
        
        for each_image_index in xrange(X_te.shape[0]):
            distance_from_each_image = np.sum(np.abs(self.X_tr-X_te[each_image_index]))
            minimum_distance_index = np.argmin(distance_from_each_image)
            predicted_labels[each_image_index] = self.Y_tr[minimum_distance_index]
        
        return predicted_labels
            
            
```


```python
Y_tr = np.asarray(Y_tr)
Y_te = np.asarray(Y_te)
nn = NearestNeighbor() # create a Nearest Neighbor classifier class
nn.train(X_tr, np.array(Y_tr)) # train the classifier on the training images and labels
Y_te_pred = nn.predict(X_te[:1000]) # predict labels on the test images
# and now print the classification accuracy, which is the average number
# of examples that are correctly predicted (i.e. label matches)
print 'accuracy: %f' % ( np.mean(Y_te_pred == Y_te[:1000]) )
```

    accuracy: 0.112000


What did we just witness?

A very slow model to test and a very fast model to train. Each image in the test set has to compare with all the 50000 images. There are 10000 training images. So a total of more than 500000 calculation steps at a time. At this point I want to emphasize why we chose to use nearest neighbor algorithm for a classification problem? It is because it sets up a baseline. We have several augumentation in place for this simple algorithm. But it will not exceed the near human classification performance or near state of art Deep learning model which is around 95%.

Some augumentations we can try are:

* Using $L_2$ distance or Euclidean distance instead of $L_1$ distance.
* We have only considered nearest neighbor. We can consider K nearest neighbor before deciding on the label. Find the minimum distance between K nearest neighbors and take a vote between these neighbors to decide on the label of the instance.

But, There are many more such parameters that can improve this model. Infact, We can use different considerations like different types of distances and different number of neighbors to tune the model better and bring in better accuracy. These choices are called **hyperparameters** and they come up very often in the design of many Machine Learning algorithms that learn from data. It’s often not obvious what values/settings one should choose.

## Hyperparameter Tuning

 We cannot use the test set for the purpose of tweaking hyperparameters. Whenever you’re designing Machine Learning algorithms, you should think of the test set as a very precious resource that should ideally never be touched until one time at the very end. Otherwise, the very real danger is that you may tune your hyperparameters to work well on the test set, but if you were to deploy your model you could see a significantly reduced performance. In practice, we would say that you overfit to the test set. Another way of looking at it is that if you tune your hyperparameters on the test set, you are effectively using the test set as the training set, and therefore the performance you achieve on it will be too optimistic with respect to what you might actually observe when you deploy your model. But if you only use the test set once at end, it remains a good proxy for measuring the generalization of your classifier (we will see much more discussion surrounding generalization later in the class).
 
 So how to tune hyperparameters? The idea is to split our training set in two: a slightly smaller training set, and what we call a validation set. Using CIFAR-10 as an example, we could for example use 49,000 of the training images for training, and leave 1,000 aside for validation. This validation set is essentially used as a fake test set to tune the hyper-parameters.


```python
X_val = X_tr[:1000,:]
Y_val = Y_tr[:1000]
X_tr = X_tr[1000:,:]
Y_tr =  Y_tr[1000:]
```

We have code to find only the nearest neighbor. So writing the code to augument. K nearest neighbor.


```python
class KNearestNeighbor(object):
    
    def __init__(self):
        pass
    
    def train(self,X_tr, Y_tr):
        """
           X_tr : A matrix of N images of D pixels each. N X D
           Y_tr : An array or list of N labels.
        """
        
        self.X_tr = X_tr
        self.Y_tr = Y_tr
    
    
    
    def predict(self,X_te,k):
        """
          X_te :  A matrix of N images of D pixels each. NXD.
          returns a list or numpy array of predicted labels of length N.
        """
        
        predicted_labels =  np.zeros(X_te.shape[0],dtype=self.Y_tr.dtype)
        
        for each_image_index in xrange(X_te.shape[0]):
            # using L2 distance this time.
            distance_from_each_image = np.sqrt(np.sum(np.square(self.X_tr - X_te[each_image_index,:]), axis = 1)) 
            # 49000 X 1
            # a very useful function to know. 
            # http://stackoverflow.com/questions/34226400/find-the-k-smallest-values-of-a-numpy-array
            k_minimum_distance_index = np.argpartition(distance_from_each_image,k) 
            index_of_labels_to_vote =  k_minimum_distance_index[:k]
            k_predicted_labels = self.Y_tr[index_of_labels_to_vote]
            # another useful function to know.
            # http://stackoverflow.com/questions/6252280/find-the-most-frequent-number-in-a-numpy-vector
            counts_of_labels = np.bincount(k_predicted_labels)
            best_label = np.argmax(counts_of_labels)
            predicted_labels[each_image_index] = best_label
        
        return predicted_labels
```


```python
validation_accuracies = []
X_val = X_val[:400]
Y_val = Y_val[:400]
for k in [1,3,5,10,20,50,100]:
    knn = KNearestNeighbor()
    knn.train(X_tr,Y_tr)
    Y_predic_val = knn.predict(X_val, k = k)
    acc = np.mean(Y_predic_val == Y_val)
    print 'K-value: {0} accuracy :{1}'.format(k,acc)
    validation_accuracies.append((k, acc))

print validation_accuracies
```

    K-value: 1 accuracy :0.26
    K-value: 3 accuracy :0.2125
    K-value: 5 accuracy :0.2275
    K-value: 10 accuracy :0.22
    K-value: 20 accuracy :0.2225
    K-value: 50 accuracy :0.21
    K-value: 100 accuracy :0.2225
    [(1, 0.26000000000000001), (3, 0.21249999999999999), (5, 0.22750000000000001), (10, 0.22), (20, 0.2225), (50, 0.20999999999999999), (100, 0.2225)]


You can see that, i ran only for $400$ validation samples out of $1000$. This is again because of the computational performance of the algorithm. You are most-welcome to run for all the 1000 validation samples.By the end of this procedure, we could get an intution on which values of k work best. We would then stick with this value and evaluate once on the actual test set.

If you want to fine-tune better, we can use another technique called **Cross-validation** in typical scenarios where the size of your training data (and therefore also the validation data) might be small.




```python
# lets try a five fold cross validation
# train set contains 1000 elements
# test set contains 200 elements.
# all taken from train set
X_cv_tr = X_tr[:1000,:]
Y_cv_tr =  Y_tr[:1000]
X_cv_te =  X_tr[1000:1200,:]
Y_cv_te = Y_tr[1000:1200]
X_cv_tr.shape,Y_cv_tr.shape,X_cv_te.shape,Y_cv_te.shape
```




    ((1000, 3072), (1000,), (200, 3072), (200,))




```python
# do not use test set for hyper parameter tuning.

val_set_start =0
val_set_end =200
counter = 1

cv_accuracy_list = []

```


```python
while(val_set_start<val_set_end) and (val_set_end <= len(X_cv_tr)):
    print "iteration {0}, between {1} and {2} in train set are set to validation".format(counter,val_set_start,val_set_end)
    # validation set
    X_cv_valid = X_cv_tr[val_set_start:val_set_end]
    Y_cv_valid = Y_cv_tr[val_set_start:val_set_end]
    
    # train set
    X_train_cv = np.concatenate((X_cv_tr[:val_set_start],X_cv_tr[val_set_end:]),axis=0)
    Y_train_cv = np.concatenate((Y_cv_tr[:val_set_start],Y_cv_tr[val_set_end:]),axis=0)
    
    # validation accuracy for this set
    validation_accuracy = []
    
    for k in [1,3,5,10,20,50,100]:
        
        knn = KNearestNeighbor()
        knn.train(X_train_cv,Y_train_cv)
        
        Y_predic_val = knn.predict(X_cv_valid, k = k)
        acc = np.mean(Y_predic_val == Y_cv_valid)
        
        print 'validation set {0} K-value: {1} accuracy :{2}'.format(counter,k,acc)
        validation_accuracy.append((k, acc))
    cv_accuracy_list.extend(validation_accuracy)
    print
    
    val_set_start +=200
    val_set_end +=200
    counter +=1
```

    iteration 1, between 0 and 200 in train set are set to validation
    validation set 1 K-value: 1 accuracy :0.185
    validation set 1 K-value: 3 accuracy :0.16
    validation set 1 K-value: 5 accuracy :0.145
    validation set 1 K-value: 10 accuracy :0.135
    validation set 1 K-value: 20 accuracy :0.195
    validation set 1 K-value: 50 accuracy :0.215
    validation set 1 K-value: 100 accuracy :0.205
    
    iteration 2, between 200 and 400 in train set are set to validation
    validation set 2 K-value: 1 accuracy :0.195
    validation set 2 K-value: 3 accuracy :0.155
    validation set 2 K-value: 5 accuracy :0.18
    validation set 2 K-value: 10 accuracy :0.215
    validation set 2 K-value: 20 accuracy :0.195
    validation set 2 K-value: 50 accuracy :0.21
    validation set 2 K-value: 100 accuracy :0.215
    
    iteration 3, between 400 and 600 in train set are set to validation
    validation set 3 K-value: 1 accuracy :0.185
    validation set 3 K-value: 3 accuracy :0.12
    validation set 3 K-value: 5 accuracy :0.155
    validation set 3 K-value: 10 accuracy :0.145
    validation set 3 K-value: 20 accuracy :0.225
    validation set 3 K-value: 50 accuracy :0.245
    validation set 3 K-value: 100 accuracy :0.21
    
    iteration 4, between 600 and 800 in train set are set to validation
    validation set 4 K-value: 1 accuracy :0.18
    validation set 4 K-value: 3 accuracy :0.18
    validation set 4 K-value: 5 accuracy :0.15
    validation set 4 K-value: 10 accuracy :0.18
    validation set 4 K-value: 20 accuracy :0.185
    validation set 4 K-value: 50 accuracy :0.17
    validation set 4 K-value: 100 accuracy :0.18
    
    iteration 5, between 800 and 1000 in train set are set to validation
    validation set 5 K-value: 1 accuracy :0.155
    validation set 5 K-value: 3 accuracy :0.175
    validation set 5 K-value: 5 accuracy :0.185
    validation set 5 K-value: 10 accuracy :0.165
    validation set 5 K-value: 20 accuracy :0.165
    validation set 5 K-value: 50 accuracy :0.16
    validation set 5 K-value: 100 accuracy :0.155
    


What actually happend above is called a n-fold cross validation. where n here is 5. 

* We have 1000 training data points. 
* Divide them into 5 equal chunks which is 200 points
* use each of the chuck as validation set and tune hyper parameters.
* So for each fold in cross validation, we have calculated the accuracy for different values of K

### How to identify which is the best K in n-fold cross validation.

We have an array called cv accuracy list. Mean accuracy for a specify K over all folds gives the ultimate accuracy of K. We can then use the best K based on the accuracy and use it in test set.   


```python
accuracy_dict = {}
for (key,acc) in cv_accuracy_list:
    
    if key in accuracy_dict:
        accuracy_dict[key].append(acc)
    else:
        accuracy_dict[key] = [acc]

for key, value in accuracy_dict.iteritems():
    accuracy_dict[key] = round(sum(value)/float(len(value)),3)

print accuracy_dict
```

    {1: 0.18, 3: 0.158, 100: 0.193, 5: 0.163, 10: 0.168, 50: 0.2, 20: 0.193}


You can see for 50 there is maximum accuracy of **0.2**. Use it to predict the the test labels.


```python
predict_test_Knn = KNearestNeighbor()
predict_test_Knn.train(X_cv_tr,Y_cv_tr)
Y_te_pred = predict_test_Knn.predict(X_cv_te,k=50)
print 'accuracy: %f' % ( np.mean(Y_te_pred == Y_cv_te))
```

    accuracy: 0.210000


In practice, people prefer to avoid cross-validation in favor of having a single validation split, since cross-validation can be computationally expensive. The splits people tend to use is between 50%-90% of the training data for training and rest for validation. We used 80-20 rule. 

However, this depends on multiple factors: 
* If the number of hyperparameters is large you may prefer to use bigger validation splits. 
* If the number of examples in the validation set is small (perhaps only a few hundred or so), it is safer to use cross-validation. 
* Typical number of folds you can see in practice would be 3-fold, 5-fold or 10-fold cross-validation.

### Pros and Cons of Nearest Neighbor

**Pros**

* It is very simple to implement and understand
* Takes no time to train, since all that is required is to store and possibly index the training data

**Cons**

* Fails over high dimensions. High dimensional distance identification is always a problem. Look into an example given in the resource I have mentioned above to understand how particulary image classification, which is a very good example of High dimensional classification is poorly done by kNN. Also we got an accuracy of around 20 percent which itself proves that the algo does poorly for high dimensional classification.

* Computationally hard. Again we witnessed this for simple NN and KNN. There are new advancements in resolving this problem. Algorithms like [FLANN](http://www.cs.ubc.ca/research/flann/), [KDtrees](https://en.wikipedia.org/wiki/K-d_tree) are trying to solve this problems. 

* Takes a lot of time to test even though we have no training time. We witnessed the same in our code.

## Recommeneded Readings

1. [CS231n] (http://cs231n.github.io/classification/). Particularly the last few parts. I have not briefed much here.

## Conclusion

 Clearly, we came to know that a baseline classification method like KNN failed miserably with data in high dimensions. But in this tutorial, we came to know about the Machine learning template. That is
 
 * Try a simple algorithm
 * improvise it. Understand Hyperparameters that affects performance of the algorithm
 * fine tune it. Regular validation approach
 * Cross validation approach
 * infer what we learnt.

Learning is all about introspecting and retrospecting what we have already learnt!
