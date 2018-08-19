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

## Reviwer questions:
<br>

#### Question
 Your last layer is described as "Fully connected input = 84, output = 10",which is incorrect. As you stated, the output is 43 labels instead of 10.<br>
#### Answer
 Sorry, i use the output= 10 on the initial times that I run it, i used the LeNet code from "lenet for trafic signs" (lesson 4 - LeNet Implementation ), but i wanna a better CNN, so i was trying many options, as I was using the google colab, i don't care in train many differents values because in colab I have free GPU, so i really used it, in the finish i had the values that are in the jupyter notebook.
<br>
---
#### Question
 You may also add a visualization of the final model showing the connections between the different layers. Have your tried visualizing this architecture using TensorBoard?
<br>
#### Answer
 I'm trying, but i'm having dificults and i can't contact the teacher ;-; initially had a bug taht i cant send message to my teacher, after the teacher's chat option go away, please, where can i say qith a teacher? you can respond me at my e-mail italo.i@live.com
<br>
---
#### Question
 Please add some words on why you choose the hyperparameters that you have chosen. Why did you choose a batch size of 512 instead of 256 or 128?<br>
#### Answer
 I used 128 every time '-'
<br>
---
#### Question 
Please discuss and justify your choice ( and include in your write-up) the following.

batch size
number of epochs
values for hyperparameters.
what optimizer was used
<br>
#### Answer
batch size: i used the values that are used in udacity lessons
number of epochs: i try 5, 10, 20, 30, 40 and 50, after 30 it dont have more effects
values for hyperparameters: aswer is the same of first question
what optimizer was used: Adam Optimizer
<br>
---
#### Question 
How would I choose the optimizer? What is its Pros & Cons and how would I evaluate it?
#### Answer
I used the same of LeNet code from "lenet for trafic signs" (lesson 5 - LeNet Training Pipeline ), but i researched the why use the adamOptimizer and i liked this answer https://stats.stackexchange.com/questions/232719/what-is-the-reason-that-the-adam-optimizer-is-considered-robust-to-the-value-of
<br>
---
#### Question 
Overfitting and Underfitting?
#### Answer
I guess had Underfitting, because the training acuracy is 97% and my test accuracy is 86%, buuuut i guess it's not a problem in this case, the acuracy values make sense.
<br>
---
#### Question 
How would I decide the number and type of layers?
#### Answer
i used the default LeNet layers numbers and types that i found on lenet lessons
<br>
---
#### Question 
How would I tune the hyperparameter? How many values should I test and how to decide the values?
#### Answer
I dont know a formule to do it, but if the dataset is big, i generally use 20% of data to test, when i have a small dataset, i use 30% of dta to test.
<br>
---
#### Question 
How would I preprocess my data? Why do I need to apply a certain technique?
#### Answer
Some algorithms require or perform better when the data be in a specific format.
<br>
---
#### Question 
How would I train the model?
#### Answer
I don't understanded the motive of this question, if i need explain how can i train a model, i will write so much, this question don't make sense to this project.






