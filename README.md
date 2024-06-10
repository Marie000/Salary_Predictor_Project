# Salary Predictor

I am working here on organizing my code for my salary predictor in a more modular fashion, and gradually integrating mlops principles.

This is still very much a work in progress, so a lot of the files are not done and/or have not been tested yet. 

It should improve in the next few days/weeks (as of May 29, 2024)

The original notebook for this project can be found here:
https://github.com/Marie000/Linkedin-predictor-model/blob/main/Copy_of_LinkedIn_Salary_Predictor.ipynb

Link to the working demo of this project can be found here:
https://huggingface.co/spaces/marie000/salary-predictor

## Some of what I did with this project

### Reorganizing the code

I made the code more modular and set up the predictor_model as a python package.
The organization of the code continues to evolve. 
I have set up the requirements.txt for the whole project, a Makefile, etc.

### Experiment tracking with MLFlow

experiment_tracking is set up to run experiments with different values of various hyperparameters. 
Eventually I would like to set it up to be able to run multiple experiments by defining a range of values for hyperparameters. 

### Docker

I started by creating containers for the predictor and the training pipeline. They seem to work, but they are missing some important files, like a trained model and vocabulary. 

I then created a dockerfile to wrap the whole project, including a fastAPI application. It worked, but I didn't deploy it or go any further with it because with the size it had, I couldn't use any free hosting services. 

### WhyLabs

I used whylabs to record logs of changes in prediction metrics over time. I used 5 very small subsets of the data and changed their timestamp to make it seem like they were dated from the last 5 days. 

![whylabs results](/images/whylabs.png?raw=true "WhyLabs Demo Results")

