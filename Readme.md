# ML labs

In this repository you can find a realization for diffrent ML problems which is a practice part of the Machine Learning class.

## Lab1: Image classification
2 approaches of the image classification problem' solving were chosen to take a hands on this work: _Linear Regression_ was implemnted with the help of Sklearn lib, and _Neural Network_ with the help of Tensorflow&Keras resource.

Classifier type can be chosen by passing an arg to `classify("nn") | classify("")` function. Linear regression will be chosen by default. 

Images are displayed with the help of PIL library. 

The big amount of the execution time is taking by the `compute_hashes and remove_hashes` functions, they offer us an ability to check the dataset for the dupes by calculating hash number for the content of each file, get rid of the empty files, and remove them from the dataset.

Training set size is about ~200000 examples, all data are flattened and converted to the grayscale range of shades.

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
