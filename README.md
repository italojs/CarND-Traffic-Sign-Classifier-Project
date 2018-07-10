## Project: Traffic Sign Recognition
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
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

[image1]: ./images/image1.png "Visualization 1"
[image2]: ./images/image2.png "Visualization 2"
[image3]: ./images/image3.png "Traffic Sign 1"
[image4]: ./images/image4.png "Traffic Sign 2"
[image5]: ./images/image5.png "CNN Visualization"

## Rubric Points
### All [rubric points](https://review.udacity.com/#!/rubrics/481/view) are in this [jupyter notebook](https://github.com/italojs/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)
---

## Conclusion:

1 - I think the bad prediction of the max speed and others sign was due the small quantity of examples for this kind of images on the data sample.

2 - I would reduce the number of epochs to prevent it from going up and down on the prediction accuracy.