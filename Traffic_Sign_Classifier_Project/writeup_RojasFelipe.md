# **Traffic Sign Recognition** 

## Writeup
## Traffic Sign Recognition Project
### By Felipe Rojas
---



The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./Writeup_images/dataset_plot.png "Data Representation"
[image2]: ./Writeup_images/image_example.png "Color Image Example"
[image3]: ./Writeup_images/grayscale_image_example.png "Grayscale Image Example"
[image4]: ./Writeup_images/normalized_image_example.png "Normalized Image"
[image5]: ./test_images/test1.png "Traffic Sign 1"
[image6]: ./test_images/test2.png "Traffic Sign 2"
[image7]: ./test_images/test3.png "Traffic Sign 3"
[image8]: ./test_images/test4.png "Traffic Sign 4"
[image9]: ./test_images/test5.png "Traffic Sign 5"
[image10]: ./Writeup_images/test1_norm.png "Traffic Sign 1 Normalized"
[image11]: ./Writeup_images/test2_norm.png "Traffic Sign 2 Normalized"
[image12]: ./Writeup_images/test3_norm.png "Traffic Sign 3 Normalized"
[image13]: ./Writeup_images/test4_norm.png "Traffic Sign 4 Normalized"
[image14]: ./Writeup_images/test5_norm.png "Traffic Sign 5 Normalized"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

### Data Set Summary & Exploration

#### 1. Summary of the Dataset

The dataset has 4 dimensiones for unique values. Features, labels, sizes and coords. 
For the training, I am interested in the features (X) and the labels (y). So, I take those dimensiones and assigned them to new variables.
X_train, y_train, X_valid, y_valid, X_test, y_test.

Both X_train and y_train have the same amount of elements. To know how many I have two ways:
* First, with the len() method
* Or with the shape method > X_train.shape[0]
 The same process for the Validation and Test datasets.
 
 To know the size of each image in the data set, I just need to get the shape of one of the elements in the array.
 * With the shape method, I can get the shape of the element 0 > X_train[0].shape
  It is the same for all the datasets
  
  Finally, to get the number of unique classes or labels, I used the set method because this returns only one of each item in an array. With one of each item, I just need to count
  the total number of items
  The labels are on the y variables. > n_classes = len(set(y_train))

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43 

#### 2.Exploratory visualization
As explained in the class, it is good practice to separate the Training set and test set 80/20. Here, I plotted the amount of data on each dataset. 

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Firstly, I plotted the image to see with what I was dealing.

![alt text][image2]

Just like stated in the data summary, I have 34799 three channel color images for the training set. I decided to convert these images to grayscale because the objective of the convolution and pooling is to highlight certain features in the images and make it "smaller" and I think working with 1 channel instead of 3 can help the CovNet to identify these features. Here is the image after grayscaling

![alt text][image3]

Like in the class, I normalized the image to ensure a similar data distribution among all the pixels. It was recommended to have a mean of 0, so I normalized the data between -1 and 1. Here is the image

![alt text][image4]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 Grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 10x10x16		|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| flatten	| Output = 400        									|
| Fully Connected		| Output = 120 				|
|		RELU			|												|
|			Fully Connected			|						Output = 84						|
|		RELU		|									|
|			Fully Connected			|						Output = 43	 (Total number of classes)					|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model I used the LeNet architecture as described above. 
My first aproach was to use low epochs and change the learning rate. After that, I added 5 epochs and tried again.

Best results for each +5 epochs 
10 epochs rate = 0.0013 accuracy = 0.89 / 
10 epochs rate = 0.0015 accuracy = 0.878 / 
10 epochs rate = 0.0017 accuracy = 0.902/ 
10 epochs rate = 0.002 accuracy = 0.916 / 
10 epochs rate = 0.0022 accuracy = 0.910 / 
10 epochs rate = 0.0025 accuracy = 0.935 / 0.917
best accuracy = 0.935

15 epochs rate = 0.0013 accuracy = 0.897 / 
15 epochs rate = 0.0015 accuracy = 0.912 / 
15 epochs rate = 0.0017 accuracy = 0.919/ 0.909
15 epochs rate = 0.002 accuracy = 0.913 / 
15 epochs rate = 0.0022 accuracy = 0.910 / 
15 epochs rate = 0.0025 accuracy = 0.91
best accuracy = 0.919

20 epochs rate = 0.0013 accuracy = 0.893 / 
20 epochs rate = 0.0015 accuracy = 0.917 / 
20 epochs rate = 0.0017 accuracy = 0.901/ 0.895
20 epochs rate = 0.002 accuracy = 0.920 / 0.902
20 epochs rate = 0.0022 accuracy = 0.92 / 0.92
20 epochs rate = 0.0025 accuracy = 0.907 / 
 best accuracy = 0.92

and so on...

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

As mentioned before, I was testing different rates and adding +5 epochs to see which gave better results. The more epochs I added, the better the accuracy and the best learning rate was 0.0025.
Above you can see some of my tries to get the 0.93. I did a lot more of tries than those, but included some here to show my approach.

My final model results were:
* training set accuracy of 1
* validation set accuracy of 0.94 
* test set accuracy of 0.91


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7] 
![alt text][image8] ![alt text][image9]

I choose these images because I wanted traffic sign images close to the ones we may find in the streets. ]I think the more difficult images for the algorithm to correctly classify are the second, fourth and fifth. This is because there are other things inside the image that can "confuse" the classification proccess. 

First, I preprocess the images and this is what I got.

![alt text][image10] ![alt text][image11] ![alt text][image12] 
![alt text][image13] ![alt text][image14]



#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No entry      		| No passing   									| 
| Road Work     			| Road Work  										|
| Speed limit (50km/h)		| Road Work							|
| Yield	      		| Priority Road					 				|
| Stop			| No passing for vehicles over 3.5 metric tons	|

The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 50%. I expected a better accuracy but as I predicted, the model had trouble with the images. Viewing the top 5 probability for each sign, the correct prediction where pretty close

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


For the first image, the No Entry, the model has pretty sure it was the No Passing traffic sign. This is pretty bad, and I can say the model will predict wrong the No entry signs.
The No entry probability is still in the Top 5, but compared to the first probability it is far.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 20.396048         			| No passing   									| 
| 11.721661     				| No passing for vehicles over 3.5 metric tons 	|
| 10.22104					| Ahead only	|
| 9.639948	      			| Turn right ahead				|
| 3.1942415				    | No entry	|


For the second image, the Road Work, the model correctly predicted this road sign, but it was not like really really sure. So I cannot say that the model will certainly predict this road sign everytime. 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 6.856482       			| Road work   									| 
| 3.852938     				| Bumpy road 	|
| 3.0637822				| Dangerous curve to the right	|
| 0.7296789	      			| Bicycles crossing		|
| 0.25214455		    | General caution	|

For the third image, the speed limit (50km/h), the model couldn't correctly predict the traffic sign. However, the prediction was not certain. Among the top 5 prediction, there is the Speed limit (50km/h) and also Speed limit (80km/h), which is close, so there are some features that are correctly predicted.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 5.4225535       			| Road work   									| 
| 3.918419    				| Speed limit (80km/h) 	|
| 3.8647203			| Wild animals crossing	|
| 2.0528603      			| Speed limit (50km/h)		|
| 1.9770452	    | No passing for vehicles over 3.5 metric tons	|

For the fourth image, the Yield sign, the model was incorrect. Again, the model was not certain but this time the correct answer was not among the top 5 predictions.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 4.4828563      			| Priority road 									| 
| 1.7780874   				| Speed limit (60km/h) 	|
| 1.5796206			| Speed limit (50km/h)	|
| 1.504427     			| End of speed limit (80km/h)		|
| 0.9585714    | Speed limit (100km/h)	|

For the fifth image, the Stop sign, it was pretty sure on totally incorrect signs so the model completely missed the correct features for the Stop Sign. In my opinion, this was the hardest traffic sign to predict, and I did it this way because Stop is a really important sign to correctly predict, but the low accuracy is disappointing. 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 16.038725       			| No passing for vehicles over 3.5 metric tons									| 
| 11.541281    				| Priority road 	|
| 8.385678			| Keep right	|
| 5.7303505      			| Double curve		|
| 4.5727777	    | Road work)	|

I tried a lot of times to get correct prediction of the images from internet, but at most I got a 60% accuracy (3 out of 5). However, this was not consistent when I run the same block many times. It would vary between 20% and 40% most of the times. 
Maybe a new approach, or other data pre processing techniques, could improve this accuracy, but here are the best results I could get in the end.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
