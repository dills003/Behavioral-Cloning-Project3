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

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

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

My model consists of a two convolution neural networks with 5x5 filter sizes and depths between 12 and 36 (model.py code lines 80 & 82). I also used two max pooling layers and four dense layers(model.py code lines 80 & 82 

The model includes RELU (model.py code lines 80 & 82) and a sigmoid (model.py code line 85)  to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (model.py code line 81-88). 

2. Attempts to reduce overfitting in the model

To reduce overfitting, I tried to use the smallest network with less convulution filters as possible to achieve good autonomous driving. I orignally tried to use a large Nvidia/googlent type architecture, but realized it wasn't necessary.I also used augmented my data by using all three cameras and by flipping every imagage. More data is a great way to help with overfitting I believe.  

Also, the model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. I reserved 20% of my collected data to validate with.

3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 89).

4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used only center lane driving, but used all three cameras to fake like the car was heading off the road. I figured if I was building an off road driver, I would drive off road, but since the goal was to stay on the road, I dind't bother driving off the road.

For details about how I created the training data, see the next section. 

Model Architecture and Training Strategy

1. Solution Design Approach

My first solution to the solution went horribly wrong. I proabably spent four sleepless night trying to figure out where I went wrong. I decided to implement a Nvidia/googlenet type strategy with tons of data. At one point I had 400,000 data points. I drove on both tracks forward and reverse several times. I had layers and layers of convolutions and dropouts. I kept messing with filter depths and sizes, to the point I was driving myself mad. I then decided to take a step back and use a more traditionsl LeNet structure, it worked for me before, so I figured that I would give it a shot. After one hour, I was driving perfectly.

My first step was to use a convolution neural network model similar to the one I built in project two. I thought this model might be appropriate because it was simple and has fewer knobs for me to tweak.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that the images were normalized.

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had  number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
