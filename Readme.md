Katsiaryna Tushynskaya, 256241

# ML labs

In this repository you can find a realization for different ML problems which is a practice part of the Machine Learning class.

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

### Learning curves:
<img width="409" alt="Screenshot 2022-11-03 at 17 47 30" src="https://user-images.githubusercontent.com/43992068/199783049-0aba818f-f75a-4257-be47-0045039c96d5.png">


## Lab2: Image classification (DNN)
In this stage a deep neural network was designed and implemented to solve image classification task with increased accuracy. Designed netowrk consists of 3 inner layers alongside with some utilitary steps. A different layer configurations were used to achieve the highest possible classification precision. Inner layers utilize **ReLU** (rectified linear unit) activation function.

### Learning curves:
<img width="627" alt="Screenshot 2022-11-03 at 16 40 24" src="https://user-images.githubusercontent.com/43992068/199783007-9b851e65-75a7-43ec-b1b0-515f5a2ccb72.png">

**Result:** average validation precision score with training dataset of 200.000 items is approx 10% higher than in task #1.

On the next stage a dropout layer was introduced to mitigate risks of network overfitting. Network was trained and validated with different combination of 1 to 3 dropout layers using various parameters.

**Result:** neural network with introduced dropout layer did not show significant increase in classification precision – the result was improved by 1-1.5% of validation accuracy. This is possible because training dataset is not too big to introduce overfitting of the neural network.

On the last stage of the task a variable learning rate was introduced. This change did not show any significant impact on the overall classification precision. During this stage we could notice, that accuracy value wasn't increasing so fast, the range was about 0.0015 - 0.003 for an epoch.

**Result:** the highest achieved classification precision with training dataset of 200.000 items is 91% and traingng accuracy is near is 95%.

### Learning curves:
![Screenshot 2022-11-09 at 23 18 47](https://user-images.githubusercontent.com/43992068/201393660-262548fc-7cc4-4bd1-9095-7e795bf84102.png)


# Semester 2

## Lab3: Image classification (CNN)

For the 2nd semester it was chosen to proceed education process with CNN – Сonvolutional Neural Networks.

### Task:
1. Download dataset.
2. Get rid of duplicates
3. Build a classifier
   > a. 2 Con layers and one Dense
   > 
   > b. Replace conv layers with Pooling
   > 
   > c. LeNet-5 arch
4. Make a model's report

### Task 1

For the first task a CNN with two convolutional layers and one Dense layer to classify images of the first 10 letters from the alphabet (A to J) was build. The partial-linear activator was required for this task.


**Result:** average validation precision score with training dataset of 200.000 items is 94%. 

Applying the partial-linear activation function to the convolutional layers may improve the performance of the model compared to using the standard ReLU activation function, but the performance will depend on the specific dataset and the complexity of the classification task.

### Task 2
The second task was to replace conv layers with Pooling.

**Result:** average validation precision score with training dataset of 200.000 items is approx 10% lower than in task #1. 

This model has significantly fewer parameters than the previous model with convolutional layers, so it may not perform as well. 
However, the performance will depend on the specific dataset and the complexity of the classification task.

If we replace pooling layers with the maximum function with pooling layers with the average function, the accuracy of the model on test data may decrease delta will be around 3%. 
In general, replacing convolutional layers with pooling layers can affect the accuracy of the model, and for a specific task, it may be necessary to find the optimal combination of convolutional and pooling layers, as well as other model parameters, to achieve the best accuracy.

### Task 3
The 3rd task was to implement LeNet-5 arch.

**Result:** average validation precision score is in range ±3% than task #1 depending on the size of the dataset and epochs amount. 

LeNet-5 uses a series of convolutional and pooling layers to extract relevant features from input images, which makes it an efficient architecture for image classification.

### Conclusion

1. CNN with 2 conv layers and one dense: This architecture is a simple CNN with two convolutional layers followed by a dense layer for classification. It is a basic architecture that can be used for simple image classification tasks but may not perform well on more complex tasks or larger datasets.
2. CNN with pooling layers: Replacing convolutional layers with pooling layers can result in a simpler architecture that can be faster to train, but it may also result in a reduction in accuracy. This architecture may be suitable for simple image classification tasks or when there are memory constraints.
3. LeNet-5: The LeNet-5 architecture includes both convolutional and pooling layers, with multiple layers of each. It also includes two fully connected layers and an output layer with a softmax activation function. LeNet-5 has been shown to perform well on simple image classification tasks, such as recognizing handwritten digits, but may not be as effective for more complex tasks or larger datasets.

Overall, the choice of architecture will depend on the specific requirements of the image classification task, including the complexity of the task, the size of the dataset, and the available computational resources. It may be necessary to experiment with multiple architectures and hyperparameters to determine the best approach.