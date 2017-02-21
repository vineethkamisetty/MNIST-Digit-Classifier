# MNIST-Digit-Classifier
A Convolution Neural Network for Classifying Digits in MNIST Dataset

The Goal of the project is to make a Deep Neural Network for classifying Numerical Digits. The data is taken from MNIST dataset. 

The program is in python and the library/platform of choice is tensflow by Google. 

DATASET ANALYSIS:

NOTE : The dataset (train.csv and test.csv) provided in the project folder is not the compelete MNIST dataset, but a subset of it.

First, lets take a look at the dataset. The data files train.csv and test.csv contain gray-scale images of hand-drawn digits, ranging from 0 -> 9.

Each image of the digit is 28 x 28 pixels, for a total of 784 pixels. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. The pixel-value is an integer between 0 and 255, inclusive.

The training dataset(train.csv), has 785 columns. The first column, called "label", is the digit label. The rest of the 784 columns(from column index no:1 to Column index no:784) contain the pixel-values of the associated image. The structure of the image w.r.t pixels is shown as below :

000 001 002 003 ... 026 027
028 029 030 031 ... 054 055
 |   |   |   |  ...  |   |
728 729 730 731 ... 754 755
756 757 758 759 ... 782 783 

The test dataset(test.csv), is the same as the above set, except that it does not contain the "label" column only pixel values.

SETUP & TRAINING:

As mentioned above, we are using tensorflow to do the heavylifting. First, we read provided data (both train and test files). To do that, we use "pandas" python library for reading .csv files. Then we split the pixel values and the corresponding Image Lables into two seperate variables usign 'iloc'.

We set the value type as 'float' and then normalize them ([0,255] to [0.0,1.0]) in both train and test dataset.

We apply 'One-hot' vector method used in classification problems for labels mostly.

We create placeholders for input and output, to be used later during training the NN.

Next step is to initialize the weights with +ve bias and small amount of noise as we are using ReLU activation functions. 

The first layer is a convolution, followed by max pooling. The convolution computes 32 features for each 5x5 patch. Its weight tensor has a shape of [5, 5, 1, 32]. The first two dimensions are the patch size, the next is the number of input channels (1 means that images are grayscale), and the last is the number of output channels. After the convolution, pooling reduces the size of the output from 28x28 to 14x14.

The second layer has 64 features for each 5x5 patch. Its weight tensor has a shape of [5, 5, 32, 64]. Similar to previous layer, the first two dimensions are the patch size, third one is the number of input channels (32 correspond to 32 featured that we got from previous layer), and the last is the number of output channels. After max pooling, the size goes down to 7x7.

Now we add FC for the processing of the entire image, followed by dropout to reduce any overfitting. After this, the final read out/FC gives the predicted/final output.

To evaluate the NN we use cost function and to minimise it, ADAM optimiser (gradient based optimization algorithm, based on adaptive estimates) is used. Finally, the value with the highest probability is selected.

We train the NN using mini-batches instead of using all the data which might be expensive. While training we set the droput probability to be 0.5 meaning there is a 0.5 probability of keeping the connection. 

But while testing we keep the value to 1.0 (rendering it inactive).

OBSERVATIONS & ANALYSIS : 

Training the above Model on the 'train' MNIST dataset(42000 examples) and testing on test dataset (28000 examples) resulted in an accuracy of 99.08.

But, when trained and tested on the dataset (32000 training examples and 10000 testing examples) provided, an accuracy of 98.90
The Offset in accuracy is mainly due to the lack of training examples in the later dataset compared to original dataset.

The NN has been trained with different models like GradientDescentOptimizer and RMSPropOptimizer on complete dataset resulting in performance of 97.2 and 99.02 respectively.

Changing the learning rate from 0.0001 to 0.001, did not result in any significant change in time, partialy due to the GPU hardware. [training and testing time combined is around 4 minutes]

Changing the training iterations to 30,000 from 20,000, an increase of 10,000, suprisingly resulted in decrease in accuracy to 98.74. The possible reason might be that as we are feeding same images multiple times because iterations being more than number of images, there might be the a problem of overfitting resulting loss of generalization.


Hardware : NVIDIA GTX 1060 
           RAM 6GB GDDR5
           BASE CLOCK : 1506 MHz
           FLOPS : 4.4 TERAFLOPS
           Memory BW : 192Gb/s
           CUDA Cores : 1920
