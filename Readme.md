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

![](../../../../var/folders/6w/rfwlz_6d763cr2n0vkv1w1vh0000gn/T/TemporaryItems/NSIRD_screencaptureui_oMhQqV/Screenshot 2022-11-03 at 17.41.25.png)

#### NN

![](../../../../var/folders/6w/rfwlz_6d763cr2n0vkv1w1vh0000gn/T/TemporaryItems/NSIRD_screencaptureui_U0KPqW/Screenshot 2022-11-03 at 17.42.15.png)