# next_order_prediction
Predict the user's next order for SberMarket Kaggle Competition

## Introduction
An online shop collected a history of user purchases and would like to have a prediction for every user for the next order.

## Data
Train.csv is a DataFrame that holds user id, time of order and product id in each row.
Original DataFrame is transformed by pivoting into MultiIndex DataFrame to get every user carts in history order (the last is more recent). Every user has unique cart products which are collected to form a full cart possible for a user based on users history. This full cart vector is used to transform user history over time  to one_hot vectors according to what products at a time were in a cart. Stacked together these vectors form a history matrix for each user. As full cart vectors are different and history lengths vary in a wide range (min is 3, max is over 1500), history matrices were transformed to one length. The length was calculated automatically to get 70% of user's histories under the length. 

## Usage
run.py hiiden_size batch_size learning_rate num_epochs verbose

positional arguments:
  hidden_size    Number of sentences in a summary
  batch_size     The size of the batch for training
  learning_rate  Learning rate for optimization
  num_epochs     Number of epochs for training
  verbose        Print state messages


## Architecture
The solution implemented is Recurrent Neural Network (GRU) that expects (cart one-hot sequence, history) tensors for training and predicts next cart one-hot vector. 

## Training
For training the history matrices without the last time were used as training data, and the last cart vector as a target to campare to. Model prediction vector was turned into a one-hot vector by threshold comparison. THe target vector and prediction were campared by f1 score. 

The full run on args=(64, 64, 0.01, 30, True) is around 10 mins on GPU or 20 mins on M1 with 1Gb load on GPU. Best num_epoch is in (20, 35) based on more than 10 runs. Over 40 overfitting is observed.

## Score
Best f1-beta score is 0.3 =(
