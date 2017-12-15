## Building a Traffic Sign Classifier
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, I will use what I've learned about deep neural networks and convolutional neural networks to classify traffic signs. I will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, I will try out my model on images of German traffic signs found on the web.

The Project
---
The goals/steps of this project:
* Load the data set 
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Images/Visualization.jpg "Visualization"
[image2]: ./Images/beforePreprocess.jpg "before Preprocess"
[image3]: ./Images/afterPreprocess.jpg "after Preprocess"
[image4]: ./New_German_Traffic_Signs/01_Speed_limit_30.jpg "Traffic Sign 1"
[image5]: ./New_German_Traffic_Signs/13_Yield.jpg "Traffic Sign 2"
[image6]: ./New_German_Traffic_Signs/14_Stop.jpg "Traffic Sign 3"
[image7]: ./New_German_Traffic_Signs/17_No_entry.jpg "Traffic Sign 4"
[image8]: ./New_German_Traffic_Signs/22_Bumpy_road.jpg "Traffic Sign 5"
[image9]: ./New_German_Traffic_Signs/25_Road_work.jpg "Traffic Sign 6"
[image10]: ./New_German_Traffic_Signs/28_Children_crossing.jpg "Traffic Sign 7"


### Project Code

Here is a link to [Traffic_Sign_Classifier.ipynb](https://github.com/zhoujh30/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Load the data set

Here is a link to the [data set](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip) used for training.

### Explore, Summarize and Visualize the Data Set

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34,799
* The size of the validation set is 4,410
* The size of test set is 12,630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

The bar chart below shows the data distribution of the training set. The x axis represents each class ID of traffic sign images and the y axis represents the number of training sets. Here is the metadata for traffic sign names: [signnames.csv](./signnames.csv)

![Visualization][image1]

### Design, Train, and Test a Model Architecture

#### Preprocess the image

I converted the images to grayscale because color is not a significant factor that will influence the classification and in this way the same number of training set also should make training faster. I then normalized/standardized the image data because it can make training faster and reduce the chances of getting stuck in local optima.

Here is an comparison of a traffic sign image before and after preprocessing.

![before preprocess][image2]  ![after preprocess][image3]


#### Build model architecture

The model used to claasify traffic signs is a convolutional neuronal network based on [LeNet](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) architecture. Here is a summary of the final model by layers:
 
| Layer         		|     Description	        					| Input Shape|Output Shape| 
|:---------------------:|:---------------------------------------------:| :----:|:-----:|
| Convolution 5x5     	| 1x1 stride, valid padding, RELU activation 	|32x32x1|28x28x48|
| Max Pooling			| 2x2 stride, 2x2 window						|28x28x48|14x14x48|
| Convolution 5x5 	    | 1x1 stride, valid padding, RELU activation 	|14x14x48|10x10x96|
| Max Pooling			| 2x2 stride, 2x2 window	   					|10x10x96|5x5x96|
| Convolution 3x3 		| 1x1 stride, valid padding, RELU activation    |5x5x96|3x3x172|
| Max Pooling			| 1x1 stride, 2x2 window        				|3x3x172|2x2x172|
| Flatten				| resize the input from 3 dimensions to 1 dimension	|2x2x172| 688|
| Fully Connected | Connecting the layer, RELU activation|688|120|
| Fully Connected | Connecting the layer, RELU activation|120|84|
| Fully Connected | Connecting the layer, output 43 classes	|84|43|

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

My training parameters:
* EPOCHS = 40
* BATCH_SIZE = 128
* SIGMA = 0.1
* OPIMIZER: AdamOptimizer (learning rate = 0.0008)

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My model results were:
* Validation set accuracy: 0.961
* Test set accuracy of: 0.941

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] ![alt text][image7]
![alt text][image8] ![alt text][image9] ![alt text][image10] 

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 5 of the 7 traffic signs, which gives an accuracy of 71.4%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

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




