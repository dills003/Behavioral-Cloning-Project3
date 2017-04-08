**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center1.jpg "Center"
[image2]: ./examples/left1.jpg "Left"
[image3]: ./examples/right1.jpg "Right"
[image4]: ./examples/trouble.jpg "Trouble"



**Rubric Points**
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
Files Submitted & Code Quality

1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

Model Architecture and Training Strategy

1. An appropriate model architecture has been employed

My model consists of a two convolution neural networks with 5x5 filter sizes and depths between 12 and 36 (model.py code lines 80 & 82). I also used two max pooling layers and four dense layers(model.py code lines 80 & 82 ).

The model includes RELU (model.py code lines 80 & 82) and a sigmoid (model.py code line 85)  to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (model.py code line 81-88). 

2. Attempts to reduce overfitting in the model

To reduce overfitting, I tried to use the smallest network with less convulution filters as possible to achieve good autonomous driving. I orignally tried to use a large Nvidia/googlent type architecture, but realized it wasn't necessary. I also used augmented my data by using all three cameras and by flipping every imagage. More data is a great way to help with overfitting I believe.  

Also, the model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. I reserved 20% of my collected data to validate with.

3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 89).

4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used only center lane driving, but used all three cameras to fake like the car was heading off the road. I figured if I was building an off road driver, I would drive off road, but since the goal was to stay on the road, I dind't bother driving off the road.

For details about how I created the training data, see the next section. 

Model Architecture and Training Strategy

1. Solution Design Approach

My first solution to the solution went horribly wrong. I proabably spent four sleepless night trying to figure out where I went wrong. I decided to implement a Nvidia/googlenet type strategy with tons of data. At one point I had 400,000 data points. I drove on both tracks forward and reverse several times. I had layers and layers of convolutions and dropouts. I kept messing with filter depths and sizes, to the point I was driving myself mad. I then decided to take a step back and use a more traditionsl LeNet structure, it worked for me before, so I figured that I would give it a shot. After one hour, I was driving perfectly. The good thing that came out of this, is that I was forced to use a data genertor, so that was fun to learn how to use.

My first step after my first failed attempt, was to use a convolution neural network model similar to the one I built in project two (basic LeNet). I thought this model might be appropriate because it was simple and had fewer knobs for me to mess with.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training and on the validation set. This implied that I was on the right track 

To combat the overfitting, I modified the model so that the images were normalized and included more data.

The final step was to run the simulator to see how well the car was driving around track one. There were a one spot on the track where the vehicle gave me trouble. To improve the driving behavior in these cases, I recorded more data points only at those spots while driving very slowly (to gain even more data)

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

2. Final Model Architecture

The final model architecture (model.py code lines 77-90) consisted of a convolution neural network with the following layers and layer sizes:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 image   							| 
| Crop Data     	| Took 80 from the top and 25 from the bottom	|
| Convolution #1     	| 2x2 Stride, 5x5 Filter, 12 Deep 	|
| RELU					|												|
| Max pooling	      	| 2x2 Stride 				|
| Convolution #2     	| 2x2 Stride, 5x5 Filter, 36 Deep 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride 				|
| Flatten	      	| 		|
| Fully connected #1		| Outputs 200x1        									|
| Sigmoid					|												|
| Fully connected #2	| Outputs 100x1        									|
| Linear					|												|
| Fully connected #3	| Outputs 50x1        									|
| Linear					|												|
| Fully connected #4	| Outputs 1x1        									|
| Linear					|												|
| Loss Operation			|  Mean Squared Error       									|
| Optimizer			| Adam Optimizer       									|



3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

With the center lane driving, the video also records impages from left and right cameras. I used these to fake a recovery, rather then driving off the road. Below are the three cameras all at the same time:

![alt text][image1]
![alt text][image2]
![alt text][image3]

Then I repeated this process on my trouble areas in order to get more data points. Here is an image of my trouble area:

![alt text][image4]

To augment the data sat, I also flipped images and angles and added a steering factor, thinking that this would help my model train better.


After the collection process, I had  number of 3,738 data points from the center camera. After using the other cameras I had 11,214 total images. Finally, after flipping the data, I had 22,428 images. I then preprocessed this data by cropping and normalizing it.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 7 as evidenced by athe validation loss bouncing back up. I used an adam optimizer so that manually training the learning rate wasn't necessary.

| Epoch    		| Training Loss        		| Validation Loss					| 
|:---------------------:|:---------------------:|:---------------------------------------------:| 
| #1         		| 0.4294						| 0.1915 							| 
| #2         		| 0.1523						| 0.1317 							|  
| #3         		| 0.1219						| 0.1195 							|  
| #4         		| 0.1127						| 0.1015 							|  
| #5         		| 0.1082					  | 0.1179 							|  
| #6         		| 0.1046						| 0.1093 							|  
| #7         		| 0.1020						| 0.0922 							|  
| #8         		| 0.0998						| 0.1162 							|  
| #9         		| 0.0984						| 0.1066 							|  
| #10         	| 0.0989						| 0.0887 							|  


**Takeaways**
The biggest thing that I learned from this project is that the model and data must have a happy marriage to work well. You can have the greatest model in the world, but train it with terrible data and the results will be bad. The same goes for great data and a terrible model. This project made me think of something I always heard in construction, "Jack of all trades, but master of none." This is what I was trying to do with my first attempt, I was trying to make the car drive great at both tracks, but was only getting average results. But when I focused solely on the test track, good results were easy to realize. I hope autonomous cars of the future can great everywher e and not just average everywhere.
