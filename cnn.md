## Convolutional Neural Networks (CNNs / ConvNets)

Convolutional neural networks as very similar to the ordinary [feed-forward neural networks](05_neural_network.ipynb). They differ in the sense that CNNs assume explicitly that the inputs are images, which enables us to encode specific properties in the architecture to recognize certain patterns in the images. The CNNs make use of *spatial* nature of the data. It means, CNNs perceive the objects similar to our perception of different objects in nature. For example, we recognize various objects by their shapes, size and colors. These objects are combinations of edges, corners, color patches, etc. CNNs can use a variety of detectors (such as edge detectors, corner detectors) to interpret images. These detectors are called **filters** or **kernels**. The mathematical operator that takes an image and a filter as input and produces a filtered output (e.g. edges, corners, etc. ) is called **convolution**.

![](images/cnn_filter.jpg)
*Learned features in a CNN. [[Image Source](https://medium.com/diaryofawannapreneur/deep-learning-for-computer-vision-for-the-average-person-861661d8aa61)]*

## CNNs Architecture

Convolutional Neural Networks have a different architecture than regular Neural Networks. CNNs are organized in 3 dimensions (width, height and depth). Also,
Unlike ordinary neural networks that each neuron in one layer is connected to all the neurons in the next layer, in a CNN, only a small number of the neurons in the current layer connects to neurons in the next layer.

![](images/cnn_architecture.png)
*Architecture of a CNN.[[Image Source](https://www.mathworks.com/videos/introduction-to-deep-learning-what-are-convolutional-neural-networks--1489512765771.html)]*

ConvNets have three types of layers: **Convolutional Layer**, **Pooling Layer** and **Fully-Connected Layer**. By stacking these layers we can construct a convolutional neural network.

### Convolutional Layer
Convolutional layer applies a convolution operator on the input data using a filter and produces an output that is called **feature map**. The purpose of the convolution operation is to extract the high-level features such as edges, from the input image. The first ConvLayer is captures the Low-Level features such as edges, color, orientation, etc. Adding more layers enables the architecture to adapt to the high-level features as well, giving us a network which has the wholesome understanding of images in the dataset.

We execute a convolution by sliding the filter over the input. At every location, an element-wise matrix multiplication is performed and sums the result onto the feature map.

![](images/cnn_convolution.gif)
*Left: the filter slides over the input. Right: the result is summed and added to the feature map. [[Image Source](https://towardsdatascience.com/applied-deep-learning-part-4-convolutional-neural-networks-584bc134c1e2)]*

The example above was a convolution operation shown in 2D using a 3x3 filter. But in reality these convolutions are performed in 3D because an image is represented as a 3D matrix with dimensions of width, height and depth, where depth corresponds to color channels (RGB). Therefore, a convolution filter covers the entire depth of its input so it must be 3D as well.

![](images/cnn_convolution_2.png)
*The filter of size 5x5x3 slides over the volume of input. [[Image Source](https://towardsdatascience.com/applied-deep-learning-part-4-convolutional-neural-networks-584bc134c1e2)]*


We perform many convolutions on our input, where each convolution operation uses a different filter. This results in different feature maps. At the end, we stack all of these feature maps together and form the final output of the convolution layer.

![](images/cnn_convolution_3.png)
*Example of two filters (green and red) over the volume of input. [[Image Source](https://towardsdatascience.com/applied-deep-learning-part-4-convolutional-neural-networks-584bc134c1e2)]*

In order to make our output non-linear, we pass the result of the convolution operation through an activation function (usually ReLU). Thus, the values in the final feature maps are not actually the sums, but the ReLU function applied to them.

### Stride and Padding

**Stride** is the size of the step we move the convolution filter at each step. The default value of the stride is 1.

![](images/cnn_stride1.gif)
*Stride with value of 1. [[Image Source](https://towardsdatascience.com/applied-deep-learning-part-4-convolutional-neural-networks-584bc134c1e2)]*

If we increase the size of stride the feature map will get smaller. The figure below demonstrates a stride of 2.

![](images/cnn_stride2.gif)
*Stride with value of 2. [[Image Source](https://towardsdatascience.com/applied-deep-learning-part-4-convolutional-neural-networks-584bc134c1e2)]*

We can see that the size of the feature map feature is reduced in dimensionality as compared to the input. If we want to prevent the feature map from shrinking, we apply **padding** to surround the input with zeros.

![](images/cnn_padding1.gif)
*Stride = 1 with padding = 1. [[Image Source](https://www.cntk.ai/pythondocs/CNTK_103D_MNIST_ConvolutionalNeuralNetwork.html)]*

### Pooling Layer

After a convolution layer we usually perform *pooling* to reduce the dimensionality. This allows us to reduce the number of parameters, which both shortens the training time and prevents overfitting. Pooling layers downsample each feature map independently, reducing the width and height and keeping the depth intact. *max pooling* is the most common types of pooling, which takes the maximum value in each window. Pooling does not have any parameters. It just decreases the size of the feature map while at the same time keeping the important information (i.e. dominant features).

![](images/cnn_maxpooling.png)
*Max pooling takes the largest value. [[Image Source](https://www.cntk.ai/pythondocs/CNTK_103D_MNIST_ConvolutionalNeuralNetwork.html)]*


### Hyperparameters

When using ConvNets, there are certain *hyperparameters* that we need to determine.

1. Filter size (kernel size): 3x3 filter are very common, but 5x5 and 7x7 are also used depending on the application.
2. Filter count: How many filters do we want to use. It’s a power of two anywhere between 32 and 1024. The more filters, the more powerful model. However, there is a possibility of overfitting due to large amount of parameters. Therefore, we usually start off with a small number of filters at the initial layers, and gradually increase the count as we go deeper into the network.
3. Stride: The common stride value is 1
4. Padding:

### Fully Connected Layer (FC Layer)
We often have a couple of fully connected layers after convolution and pooling layers. Fully connected layers work as a classifier on top of these learned features. The last fully connected layer outputs a N dimensional vector where N is the number of classes. For example, for a digit classification CNN, N would be 10 since we have 10 digits.
 Please note that the output of both convolution and pooling layers are 3D volumes, but a fully connected layer only accepts a 1D vector of numbers. Therefore, we ***flatten*** the 3D volume, meaning we convert the 3D volume into 1D vector.

![](images/cnn_fc.jpeg)
*A CNN to classify handwritten digits. [[Image Source](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)]*


## Training ConvNets

Training CNNs is the same as ordinary neural networks. We apply backpropagation with gradient descent. For reading about training neural networks please see [here](05_neural_network.ipynb).

## ConvNets Architectures
This section is adopted from Stanford course [here](http://cs231n.github.io/convolutional-networks/). Convolutional Networks are often made up of only three layer types: CONV, POOL (i.e. Max Pooling), FC. Therefore, the most common architecture pattern is as follows:

`INPUT -> [[CONV -> RELU]*N -> POOL?]*M -> [FC -> RELU]*K -> FC`

where the `*` indicates repetition, and the `POOL?` indicates an optional pooling layer. Moreover, `N >= 0` (and usually `N <= 3`), `M >= 0`, `K >= 0` (and usually `K < 3`).

There are several architectures of CNNs available  that are very popular:

- LeNet
- AlexNet
- ZF Net
- GoogLeNet
- VGGNet
- ResNet


## Implementation



## Extra Rsources
[1] [Stanford course on Convolutional Neural networks](http://cs231n.github.io/convolutional-networks/)

[2] [A Beginner's Guide To Understanding Convolutional Neural Networks](https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks/)

[3] [Convolutional Neural Networks](https://towardsdatascience.com/applied-deep-learning-part-4-convolutional-neural-networks-584bc134c1e2)




