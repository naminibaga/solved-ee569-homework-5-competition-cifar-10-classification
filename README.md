Download Link: https://assignmentchef.com/product/solved-ee569-homework-5-competition-cifar-10-classification
<br>
CIFAR-10 dataset has been used to implement LeNet5 model architecture. Convolutional Neural Networks has wide variety of applications in Computer Vision, Image classification and recognition, etc. CNN are necessary to extract the features from the images which are key elements in those applications. The HW5 model gives the accuracy which can be improved and made the model better.

Many State of Art models have been published and I tried to implement some of those to create new model and achieve higher accuracy.

In most cases, modifying the parameters in the given architecture gives us the better result.

Also, Data Augmentation is also used as reading about it fetched me that it trains the model better. More details about using data augmentation is further discussed.

MoDel size is also one factor which is been considered while designing the same along with the model training time.

The LeNet architecture is same as the one displayed in the previous section. The main functionality of architecture is to extract the key features from the data and using it to train the model. LeNet model consists of 7 layers. The layers are as follows:

<ol>

 <li>C1- Convolution Layer</li>

 <li>S2- Sub-sampling layer</li>

 <li>C3- Convolution layer</li>

 <li>S4- Sub-sampling layer</li>

 <li>C5- Fully Connected layer</li>

</ol>

<strong><u>Flowchart</u> </strong>

Input Image(32*32*3)

Convolution 2D layer: 32 filters (3*3)

Max Pooling Layer (Pool Size: 3*3)

Convolution 2D layer: 64 filters (3*3)

Max Pooling Layer (Pool Size: 3*3)

Fully Connected layer (256 filters)

Fully Connected Layer (128 filters)

Softmax Layer




<strong><u>Parameters</u></strong>: Algorithm run on CIFAR10 dataset

Input image size= 32*32, color image

<table width="0">

 <tbody>

  <tr>

   <td width="48">C1:</td>

   <td width="341">Kernel filter used in convolution: 3*3, 32 filters are used.Padding used.Activation- ReLUOutput image size= 3*3Output- 32*32*32 </td>

  </tr>

  <tr>

   <td width="48">S2:</td>

   <td width="341">Max PoolingPadding used.</td>

  </tr>

  <tr>

   <td width="48"> </td>

   <td width="341">Pool size= 3*3</td>

  </tr>

  <tr>

   <td width="48">  </td>

   <td width="341">Output= 11*11*32</td>

  </tr>

  <tr>

   <td width="48">C3:</td>

   <td width="341">Kernel filter used in convolution: 3*3, 64 filters are used.Padding used.Activation- ReLUOutput- 11*11*64 </td>

  </tr>

  <tr>

   <td width="48">S4:</td>

   <td width="341">Max PoolingPadding used.</td>

  </tr>

  <tr>

   <td width="48"> </td>

   <td width="341">Pool size= 3*3</td>

  </tr>

  <tr>

   <td width="48">  </td>

   <td width="341">Output= 4*4*64</td>

  </tr>

 </tbody>

</table>

<table width="0">

 <tbody>

  <tr>

   <td width="96"> </td>

   <td width="102"> </td>

  </tr>

  <tr>

   <td width="96">F5: </td>

   <td width="102">256 filters Activation- ReLU</td>

  </tr>

  <tr>

   <td width="96">F6:</td>

   <td width="102">128 filters Activation- ReLU</td>

  </tr>

 </tbody>

</table>




Output layer: Number of classes in the dataset= 10.

Activation- Softmax







Fig. 3 Parameters in the architecture




<strong><u>Reason choosing these parameters:</u> </strong>

Having done HW5 problem 1, I had a fair idea what each parameter does and the functionality of the same.

Firstly, I knew data augmentation was necessary which was not done by me in homework 5 thus increasing my accuracy here.

Adding, choosing the number of filters was trial and error as I knew more filters will make more trainable parameters which have potential to increase the accuracy but downfall of high model size. I have tried to keep the best number of filter and the kernel size.

Max Pooling was used in padding was set to avoid data loss around boundary and kernel of 3*3

Choosing the second convolutional layer, number of filters to be higher than 1<sup>st</sup> convolutional which is usually chosen, I did the same as well.

The second pooling layer parameter was chosen same as that of previous pooling layer.

The dense layer parameter was chosen in the power of 2 and decreasing thereby.

Dropout also helps in achieving higher performance which is why I have added the same. It is usually between 0 and 1, normally chosen as 0.1 To 0.4.

Finally, the dense layer using softmax with parameter 10 which is same as that of labels in the data.




<strong> </strong>

<strong> </strong>

<strong> </strong>

<strong> </strong>

<strong><u>Training Mechanism:</u> </strong>

<ol>

 <li><u>Data Augmentation:</u> Convolutional Neural Network (CNN) is invariant to shift, scale, rotate, illumination, etc. This property is used in data augmentation. Data augmentation is used to randomly choose the images and make multiple copies of it using the original image by shifting, scaling, rotating, shearing, zooming, whitening, etc.</li>

</ol>

In real word, we have very limited amount of data to access. So, whatever the data is, we need to utilize it fully. The given data can be made into a large amount of data by using data augmentation. Each image can have multiple copies with image looking different for the each one.

The main reason to use data augmentation is to train the model better. Let say an inverted test image appeared in the model. This image should be correctly identified. To accommodate all the different test images, we make a model such that it correctly classifies all the varieties.




Parameters used in Data Augmentation:

<ol>

 <li>Width Shift range= 0.1</li>

 <li>Height Shift range= 0.1</li>

 <li>Fill Mode= Nearest</li>

 <li>Horizontal Flip= True</li>

</ol>

Many more parameters can be altered. I found these parameters to be best and got more performance accuracy using those.




<ol start="2">

 <li><u>Loss function:</u> Cross Entropy Loss which is also called log loss is used as it measures the performance when the output is a probability value between 0 and 1. As we are using softmax function in the last layer which gives the output in 0 and 1, I have used cross entropy as the loss function.</li>

</ol>

Moreover, Cross entropy is a good loss function for the classification problem as it works on the principle minimizing the distance between two probability function, which are the predicted function and actual function.

Fig. 6 Cross Entropy

<ol start="3">

 <li><u>Optimizer</u>: I have used ‘Adam’ optimizer which is supposed to give better results in most cases. SGD is also one which is used, but my accuracy was better in using Adam. The parameters in Adam are invariant to the rescaling of the gradient unlike SGD optimizer.</li>

</ol>







<ol start="4">

 <li>Batch Size:</li>

</ol>

Batch Size is the number of training samples in one pass. Batch Size becomes important as the number specified are trained in one epoch. If it is not mentioned, all the training data would be trained in all the epochs. This will create a huge load on the memory. Generally, the smaller the batch size is better which is chosen in the powers of 2 majorly 32,64,128,256. The networks trained better using the mini batch sizes as the update of the parameters takes after every batch size. I have chosen batch size of 128 here as it gives more accuracy then the batch size of generally use size of 32.




<ol start="5">

 <li>Epochs:</li>

</ol>

It is defined as the one forward and one backward pass for all the training data. It is a hyperparameter meaning the number of times the learning algorithm would work over the entire training data.




<strong><u>Algorithm:</u> </strong>

<strong> </strong>

<ol>

 <li>Loading training and testing images from CIFAR10 dataset.</li>

 <li>Training and testing labels converted to categorical type.</li>

 <li>Pre-Processing the training data to get 0 mean and unit variance. This is necessary step as the data present maybe out of range.</li>

 <li>Using sequential model and adding layers onto it.</li>

 <li>Model includes 7 layers which are as follows: convolution, max pooling, convolution, max pooling, fully connected, fully connected, softmax layer. (Parameters given as per discussed above.)</li>

 <li>Model compilation using categorical loss function.</li>

 <li>Choosing best optimizer for the model. (Here- Adam)</li>

 <li>Data Augmentation is used to make the model train better.</li>

 <li>Training data passed through the model and fitting it into model.</li>

 <li>Passing testing data through the model over specifying number of epochs.</li>

 <li>Calculating test loss, training and testing accuracy.</li>

 <li>Plotting train and test accuracy graphs.</li>

</ol>




<h1>3. RESULTS</h1>

<strong><u>Best Choice of Parameter</u> </strong>

Epochs= 50

Batch Size= 128

Dropout=0.2

Optimizer= Adam




Model Parameter: C1- 32 filters, kernel 3*3

S1- Max Pooling, 3*3

C2- 64 filters, kernel 3*3

S2- Max Pooling, 3*3

F1: 256 filter

F2: 128 filters

F3: 10 filters softmax




<strong>Training Accuracy= 0.8239 </strong>

<strong>Training Loss= 0.4987 </strong>

<strong> </strong>

<strong>Testing Accuracy= 0.8041 </strong>

<strong>Testing Loss= 0.6002 </strong>




<strong><u>Model Size:</u> </strong>

The model size should be small as possible. While trying various parameters, I realized the model size can go pretty high very quickly. To monitor it, the parameters while training the model become super important. I have selected those which I feel the total learnable parameters are enough. As seen from the below figure, it does give the total parameters after each layer in the model and also the total model size.

Total model size: 315,978




Fig. 9 Highlighted parameter size




<strong><u>Model Time:</u> </strong>

The model time can be divided into 2 forms: training time and the inference time. The training time is the time, which is used for the training data, while the inference time is the time to infer the predictions on test data using the pretrained model. Ideally, the training time should also be smaller, and a lot of factors depend on it like the CPU, GPU or the memory one system has.

As per my model run on GPU, I get each epoch runtime as 25 seconds for each epoch. I am running for 50 epochs which means 50*25= 1250 seconds= 20.83 mins.

So, my model training time = 20.83 mins

About inference time, the testing data here is of 10K images which hardly takes around 1-2 seconds to predict once model is trained.

The total time which includes the model training time, inference time, plotting, saving the model, etc. has been reported which is 1264.1145 = 21.06 mins. (Fig. 10)




<strong><u>Graph:</u> </strong>

The model accuracy vs no of epochs and model loss vs no of epochs graph for the modified model is as shown below. It can be inferred that the model is neither underfitting nor overfitting.

<strong><u>Dropping random train samples:</u> </strong>

Fig. 13 Size of new train data after dropping 5K images




Training Accuracy: 82.19

Testing Accuracy: 79.42







Graph:







<h2>Case 2: 40K training images (10K images dropped)</h2>

Fig. 16 Size of new train data after dropping 10K images




Training Accuracy: 82.00

Graph:

Fig. 18 Accuracy vs epoch and Loss vs Epoch graph after dropping 10K images




<h1>4. DISCUSSION</h1>

<ol>

 <li>Performance Improvement from the result of homework 5 problem 1B:</li>

</ol>

The parameters and hyperparameters in the homework 5 problem 1B were given which weren’t giving the high accuracy as of this one.




<table width="0">

 <tbody>

  <tr>

   <td width="181"><strong>Parameters </strong></td>

   <td width="174"><strong>Problem 1B (HW5) </strong></td>

   <td width="174"><strong>Current Model Updated </strong></td>

  </tr>

  <tr>

   <td width="181"><strong>Batch Size </strong></td>

   <td width="174">16</td>

   <td width="174">128</td>

  </tr>

  <tr>

   <td width="181"><strong>Pre-Processing </strong></td>

   <td width="174">Yes</td>

   <td width="174">Yes</td>

  </tr>

  <tr>

   <td width="181"><strong>Data Augmentation </strong></td>

   <td width="174">No</td>

   <td width="174">Yes</td>

  </tr>

  <tr>

   <td width="181"><strong>Optimizer </strong></td>

   <td width="174">SGD</td>

   <td width="174">Adam</td>

  </tr>

  <tr>

   <td width="181"><strong>1<sup>st</sup> Conv layer </strong></td>

   <td width="174">6 filters, 5*5 kernel</td>

   <td width="174">32 filters, 3*3 kernel</td>

  </tr>

  <tr>

   <td width="181"><strong>2<sup>nd</sup> Conv layer </strong></td>

   <td width="174">16 filters, 5*5 kernel</td>

   <td width="174">64 filters, 3*3 kernel</td>

  </tr>

  <tr>

   <td width="181"><strong>1<sup>st</sup> Max Pooling </strong></td>

   <td width="174">Pool size: 2*2</td>

   <td width="174">Pool size: 3*3</td>

  </tr>

  <tr>

   <td width="181"><strong>2<sup>nd</sup> Max Pooling </strong></td>

   <td width="174">Pool size: 2*2</td>

   <td width="174">Pool size: 3*3</td>

  </tr>

  <tr>

   <td width="181"><strong>Padding </strong></td>

   <td width="174">Valid (No Pad)</td>

   <td width="174">Same (Pad)</td>

  </tr>

  <tr>

   <td width="181"><strong>1<sup>st</sup> Dense layer </strong></td>

   <td width="174">128</td>

   <td width="174">256</td>

  </tr>

  <tr>

   <td width="181"><strong>2<sup>nd</sup> Dense layer </strong></td>

   <td width="174">84</td>

   <td width="174">128</td>

  </tr>

  <tr>

   <td width="181"><strong>Dropout </strong></td>

   <td width="174">0.3</td>

   <td width="174">0.2</td>

  </tr>

  <tr>

   <td width="181"><strong>Last Dense layer </strong></td>

   <td width="174">10</td>

   <td width="174">10</td>

  </tr>

  <tr>

   <td width="181"><strong>Epochs </strong></td>

   <td width="174">20</td>

   <td width="174">50</td>

  </tr>

 </tbody>

</table>

Table 1. Comparison of models




Problem 1B Result:

Training Accuracy: 77.13

Testing Accuracy: 65.16




Current Model:

Training Accuracy: 82.39

Testing Accuracy: 80.41




As already discussed, the current model has more trainable parameters which results in higher accuracy then the previous one. This is done by increasing the filter size in both convolution layers.

Data Augmentation was added to the current model which helps in better performance as well.

The dense layer filters size has been increased to accommodate more features, same with the padding around the layers.

Batch Size has been increased to train more data in one pass.

All these collectively effect into the better training and testing accuracy for the CIFAR10 dataset.




<ol>

 <li>Degradation when randomly drop train data:</li>

</ol>

We are training all 50K images in the training data while making the model. But as we drop some random images from the training data, we see the degradation of the training and testing accuracy.

This is because we don’t have much data present in the network. It is always better to train the model with as much data one can provide. The model needs to be trained perfectly which is not possible if we give less training data as the input.

I have used 45K and 40K images as my trial for randomly dropping images which means I have dropped 5K and 10K respectively and trained the model and noted the results.

It can be seen in the result section that the training and testing accuracy in both cases have been decreased. The decrease has been 1-2% from the original model. If more samples are reduced the accuracy would drop down more. (Fig. 14 and 17) The graphs for Accuracy vs Epoch and Loss vs Epoch for both cases have also been plotted in which over entire epochs the accuracy can be visualized. (Fig. 15 and 18)