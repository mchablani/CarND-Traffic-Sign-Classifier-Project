#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/mchablani/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

I randomly plotted few images from train-validation split and test set.  Also used the validation1 set from the pickel file for validation that was provided.   After this plot it was obvious that images especially in the valid.p file were very dark and later on when I did prediction on it gave poor results so I decided to do image scaling.

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

Initially I trianed on architecture similar to LeNet but I increased the depth in all convolution layer by factor of 3 to account for more channels in images, there was no data augmentation and only data transform I applied was what was taught in lessons earlier: (x - 128) / 255
I did this because I felt color is important and conveys critical information for street signs (stop is usually red, etc).
Using this approach I was able to achieve following performance after 100 epochs: 
Validation Accuracy = 0.934
Validation Accuracy1 = 0.780
Train Accuracy = 0.993

Then I tried converting to rgb2gray() and I got similar perf results but with smaller network (no need to increased the depth in all convolution layer by factor of 3) and training was much faster.  So concluded that color does not add much value on top of shape for the given train/test/valiation data.  Maybe when using network on new data it might come useful to train with color info.  For purpose of this lab it made sense to train and iterate faster.

I was still getting bad results on Validation Accuracy1 and after looking at these images realised these images in valid.p were much darker. so I applied follwoing normalisation;  Note that I removed dividing by std() as it gave me divide by 0 in some cases and without it I was getting >99% train and validation accuracy so did not seem necessary.  
```
def normalize(x):
    # Stretch the image 
    max = np.max(x)
    x = (x/max) * 255
    # zero mean
    mu = np.mean(x)
    x = x - mu
#     # Normalize
#     std = np.std(x, axis = 0)
#     if std.all() > 0:
#         x /= std
    return x

X_train = np.array([normalize(rgb2gray(x)) for x in X_train]).reshape(-1, 32, 32, 1)
X_test = np.array([normalize(rgb2gray(x)) for x in X_test]).reshape(-1, 32, 32, 1)
X_valid = np.array([normalize(rgb2gray(x)) for x in X_valid]).reshape(-1, 32, 32, 1)
X_valid1 = np.array([normalize(rgb2gray(x)) for x in X_valid1]).reshape(-1, 32, 32, 1)

```



####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the second code cell of the IPython notebook.  

Initially I used validation data from valid.p and training from train.p.  However I found the dataset was quite defferent and this made it hard to evaluate my model.  So I have 2 validation set.  I randomly shuffled the train data and split it into train-validation in a 80-20 split.  validation1 was the data from valid.p. 

My final training set had 27839 number of images. My validation set and test set had 6960 and 12630 number of images.
Validation1 set had 4410 images.

```
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2)

print("Number of training examples =", n_train)
print("Number of validation examples =", n_valid)
print("Number of validation1 examples =", n_valid1)
print("Number of testing examples =", n_test)

print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
```
Number of training examples = 27839
Number of validation examples = 6960
Number of validation1 examples = 4410
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43


The seventh code cell of the IPython notebook contains the code for augmenting the data set. I decided to generate additional data because my train accuracy would be much better than my validation accuracy and both of them plateued at 94% without data augmentation.

To add more data to the the data set, I used the helper function in Keras. 
```
from keras.preprocessing.image import ImageDataGenerator
# define data preparation
datagen = ImageDataGenerator(
#     featurewise_center=True,
#     featurewise_std_normalization=True,
#     zca_whitening=True, 
    rotation_range=40,
    width_shift_range=0.1,
    height_shift_range=0.1,
)
# fit parameters from data
datagen.fit(X_train)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    num_examples = len(X_train)

    print("Training...")
    print()

    for i in range(EPOCHS):
        if i > EPOCHS/2:
            rate = 0.001
        batches = 0
        X_train, y_train = shuffle(X_train, y_train)

        for batch_x, batch_y in datagen.flow(X_train, y_train, batch_size=BATCH_SIZE):
#         for offset in range(0, num_examples, BATCH_SIZE):
#             end = offset + BATCH_SIZE
#             batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            batches += 1
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            if batches > (len(X_train) / BATCH_SIZE):
                break

```


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the fifth cell of the ipython notebook. 

My final model consisted of the following layers: (Very similar to LeNet and added dropouts)

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16      									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				    |
| Flatten       		| outputs 400       							|
| Fully connected		| Input = 400. Output = 120                     |
| RELU					|												|
| Dropout				| keep_prob = 0.65		                        |
| Fully connected		| Input = 120. Output = 84.                     |
| RELU					|												|
| Dropout				| keep_prob = 0.65		                        |
| Fully connected		| Input = 84. Output = 43.                      |
| Softmax				|           									|
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 

To train the model, I used an ....

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the sixth cell of the Ipython notebook.

My final model results were:
* training set accuracy of 0.993
* validation set accuracy of 0.988 
* test set accuracy of 


A well known architecture was chosen:
* What architecture was chosen?
LeNet with input of 32x32x1 and dropout was added after dense layers to avoid overfitting as dataset was smaller. 
* Why did you believe it would be relevant to the traffic sign application?
LeNet is simple to understand and debug, gave mostly good results right away.  But I could see train set do better than validation set until I added drpout and data augmentation.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 
