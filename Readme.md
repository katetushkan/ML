# ML labs

This repo has a code for the ML labs.
## Image classification
In this work 2 ways of the image classification was created: _Linear Regression_ with the help of Sklearn lib, and _Neuron Network_ with the help of Tensorflow&Keras resource.
Classifier can be chosen by passing an arg to `classify("nn") | classify("")` function. Linear regression will be chosen by default. 

Dupes removing is happening by calculating data hashes.

File `finalized_model.sav` has saved `LinearRegression` model.
### Task:
1. Download data and show a few examples with the help of the python.
2. Check if data is balanced.
3. Divide data into 3 datasets: train, test, validation
4. Get rid of duplicates
5. Build a classifier
6. Display learning curve

## Learning curves:
#### Linear regression
<img width="409" alt="Screenshot 2022-11-03 at 17 47 30" src="https://user-images.githubusercontent.com/43992068/199783049-0aba818f-f75a-4257-be47-0045039c96d5.png">


#### NN
<img width="627" alt="Screenshot 2022-11-03 at 16 40 24" src="https://user-images.githubusercontent.com/43992068/199783007-9b851e65-75a7-43ec-b1b0-515f5a2ccb72.png">
