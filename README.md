# PNEUMO VISION

# PROBLEM STATEMENT 
Pneumonia is a serious and often life-threatening respiratory illness that affects millions of people 
worldwide. It is caused by a variety of pathogens, including bacteria, viruses, and fungi, which can 
lead to inflammation and fluid buildup in the lungs.

However, traditional methods of pneumonia 
detection, such as manual interpretation of chest X-rays by radiologists, can be labor-intensive and 
subjective. Early and accurate detection of pneumonia is crucial for effective treatment and 
management, as delayed diagnosis can result in severe complications and even death.  

# SOLUTION 
Pneumonia is most common in underdeveloped and developing countries, where overpopulation, 
pollution, and unhygienic environmental conditions exacerbate the situation, and medical 
resources are scanty. Therefore, early diagnosis and management can play a pivotal role in 
preventing the disease from becoming fatal. Radiological examination of the lungs using computed 
tomography (CT), magnetic resonance imaging (MRI), or radiography (X-rays) is frequently used 
for diagnosis. 

Deep Learning is one of the scientific fields in Computer Science that actually helps the application 
to learn from pre-defined experiences. The Deep Learning domain is very useful for the prediction 
of diseases in an individual by datasets that were learned by the application using Deep Learning. 
Pneumonia detection using Deep Learning actually predicts the using the dataset and algorithms 
for better accuracy and precise results. 

Since this project solely depends upon two domains i.e., Deep Learning and Web Page 
Development, there would be a lot of challenging tasks will be present in the project.  
Our Project is built strategically by collecting various datasets of unique valued attributes for 
accurate results. This project, PneumoVision seeks to address these challenges by harnessing the 
power of deep learning to automate the detection of pneumonia from medical images.  


# AIMS & OBJECTIVES 
 AIM  
The ultimate aim of this project, ‘PneumoVision’ is to develop a system or a model which can 
actually detect pneumonia with chest X-ray with higher accuracy results by combining different 
Deep Learning Algorithms such as  - CNN Algorithm  - DenseNet  - VGG - ResNet - Inception Net (GoogLeNet) 
Each Algorithm is undertaken for accuracy purposes, the one with the best accuracy results and 
precise prediction will be taken into account as a model. 
 
 OBJECTIVES 
 The main Objectives of this project ‘PneumoVision’ are: 
1. Fast and Reliable Results using Deep Learning. 
2. Web Page for User Convenience. 
3. Time-Saving and Cost Reliable.

# SOFTWARE DESCRIPTION
To successfully develop the Convolutional Neural Network (CNN) model for pneumonia detection 
PneumoVision project, the project team will need to fulfill several key requirements. First and 
foremost, they will require access to a large, high-quality dataset of chest X-ray images, both from 
patients with confirmed pneumonia and those with healthy lungs. This dataset will serve as the 
foundation for training and validating the CNN model, ensuring its accuracy and robustness. 

In addition to the image dataset, the project will require a powerful computing infrastructure 
capable of handling the intensive training process of the deep learning model. This may involve 
the use of specialized hardware, such as high-performance GPUs, to accelerate the training and 
inference stages of the CNN. Furthermore, the team will need to have access to advanced deep 
learning libraries and frameworks, such as TensorFlow or PyTorch, which provide the necessary 
tools and functions for building, training, and deploying the CNN model. 

# TOOLS 
• ANACONDA – JUPYTER NOTEBOOK: Anaconda is a popular platform for scientific 
computing that includes Python and various data science libraries. Jupyter Notebook allows 
researchers to create interactive documents that combine code, explanations, and visualizations, 
making it ideal for developing and experimenting with the CNN model. 

• VS CODE: While Jupyter Notebook excels in interactive coding, VS Code is a versatile code 
editor known for its customization options, debugging tools, and support for various programming 
languages, including Python. This makes it a valuable tool for writing and editing the Python code 
used to build and train the CNN model.   

# FRONT END & BACK END 
• PYTHON: As the primary programming language for the project, Python offers powerful 
libraries for data manipulation, machine learning, and scientific computing. It allows the team to 
develop all aspects of the model, from data preprocessing to building and training the CNN. 

• HTML, CSS, JAVA SCRIPT: While the core functionalities reside in Python, these web 
development technologies are used to create a user interface (UI) for the PneumoVision project. 
The UI could allow users to upload chest X-ray images and receive predictions from the trained 
CNN model. 

# ARCHITECTURE  
PneumoVision can be implemented in many algorithms of Deep Learning. So, we have chosen 
some of the algorithms and started to implement each one in a separate environment to test their 
efficiency, time consumption, and accuracy results. There are 5 Algorithms we initially used here: 
CNN, ResNet , DenseNet , GoogLeNet , VGG.

CNN ALGORITHM  
CNNs have been popular due to their improved performance in image classification. The 
convolutional layers in the network along with filters help in extracting the spatial and temporal 
features in an image. The layers have a weight-sharing technique, which helps in reducing 
computation efforts. Architecture-wise, CNNs are simply feedforwarded artificial neural networks 
(ANNs) with two constraints: neurons in the same filter are only connected to local patches of the 
image to preserve the spatial structure and their weights are shared to reduce the total number of 
the model’s parameters. A CNN consists of three building blocks: (i) a convolution layer to learn 
features, (ii) a max-pooling (subsampling) layer is to down sample the image and reduce the 
dimensionality and thereby reduction in computational efforts, and (iii) a fully connected layer to 
equip the network with classification capabilities

RESNET ALGORITHM  
ResNet, which is a short form of Residual Network, was originally developed for two problems, 
such as the vanishing gradient and degradation problem. Residual learning tries to solve both these 
problems. ResNet has three different variants: ResNet18, ResNet50, and ResNet101 based on the 
number of layers in the residual network. ResNet was successfully used in biomedical image 
classification for transfer learning. In this paper, we have used ResNet18 for pneumonia detection. 
Typically, deep neural network layers learn low- or high-level features during training, while 
ResNet learns residuals instead of features.

DENSENET ALGORITHM  
DenseNet, which is a short form of the Dense Convolutional Network, needs fewer numbers of 
parameters than a conventional CNN, as it does not learn redundant feature maps. The layers in 
DenseNet are very narrow, i.e., 12 filters, which add a lesser set of new feature maps. DenseNet 
has four different variants: DenseNet121, DenseNet169, DenseNet201, and DenseNet264. In this 
paper, we have used DenseNet121 for pneumonia detection. Each layer in DenseNet has direct 
access to the original input image and gradients from the loss function. Therefore, the 
computational cost is significantly reduced, which makes DenseNet a better choice for image 
classification.

GOOGLENET ALGORITHM  
The GoogLeNet architecture is a 22-layer deep network consisting of “inception modules,” instead 
of uniformly progressive layers. An inception block accommodates a large number of units at each 
stage by hosting parallel convolution and pooling layers, resulting in an uncontrolled 
computational complexity because of the increased number of parameters. To control the 
computational complexity, the GoogLeNet model uses inception blocks with dimension reduction, 
rather than the naive inception. The performance of GoogLeNet, in which the inception block was 
introduced, proves that an optimal sparse architecture built from the available dense building 
blocks improves the performance of artificial neural networks for computer vision tasks.

VGG ALGORITHM  
Presented in 2014, VGG16 has a very simple and classical architecture, with blocks of 2 or 3 
convolutional layers followed by a pooling layer, plus a final dense network composed of 2 hidden 
layers (of 4096 nodes each) and one output layer (of 1000 nodes). Only 3x3 filters are used.

# LIBRARIES USED 
NUMPY: A fundamental library for scientific computing in Python. It provides efficient arrays 
and mathematical operations crucial for manipulating image data and performing calculations 
within the CNN model. 

OPENCV (CV2): A library for computer vision tasks, including image processing and 
analysis. It might be used for image pre-processing steps like resizing and normalization before 
feeding them into the CNN model. 

KERAS: A high-level deep learning API built on top of libraries like TensorFlow. Keras 
simplifies the process of building, training, and evaluating deep learning models. 

MATPLOTLIB: A library for creating static, publication-quality visualizations. It might be 
used to visualize the training process of the CNN model, such as plotting loss curves and 
accuracy metrics.

TENSORFLOW: A powerful open-source library for numerical computation and large-scale 
machine learning. It provides the core building blocks for creating and training deep learning 
models like the CNN for pneumonia detection.

# FLOWCHART

<img width="248" alt="Screenshot 2024-07-18 110527" src="https://github.com/user-attachments/assets/f663cfc4-7cf4-4d7d-b8fc-c378d7a4ef06" />

# RESULTS

The Ultimate aim and motto of this project ‘PneumoVision’ is to bring awareness among the people that pneumonia can be predicted earlier and can be cured with certain preventive measures. So that we have used different algorithms to test their efficiency, throughput, and their accuracy percentage. CNN, ResNet, DenseNet, VGG and Inception Net Algorithms are widely used in Deep Learning Domain to predict numerous numbers of diseases. We have trained the dataset using 5 different types of algorithms with 5 different accuracy results.
 
CNN Algorithm: In this Algorithm, the prediction efficiency was around 78.53%. 
ResNet Algorithm: In this Algorithm, the prediction efficiency was around 62.82%. 
DenseNet Algorithm: In this Algorithm, the prediction efficiency was around 62.82%. 
VGG Algorithm: In this Algorithm, the prediction efficiency was around 67.31%. 
Inception Net Algorithm: In this Algorithm, the prediction efficiency was around 70.99%.

Now, by comparing all these algorithms, CNN Algorithm has the best prediction percentage among the four algorithms. Thus, the CNN algorithm was implemented in the web page.


	1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 67ms/step
       PNEUMONIA

![image](https://github.com/user-attachments/assets/178446db-a1ea-4316-9b59-d7cc489f8f34)


      1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 82ms/step
NORMAL

![image](https://github.com/user-attachments/assets/ee1a00e2-5682-45fe-bb90-656a26b385b0)



![image](https://github.com/user-attachments/assets/11a985a2-7cd0-432d-bed4-10e2689eb019)

# CONCLUSION 
•	This project primarily focuses on detecting pneumonia with a given attribute in a span of time. This system will be useful for physicians to have an overall idea of the patient. 
•	The result will be either classified as pneumonia present or not. So that preventive measures will be given to the patients of pneumonia affected.
•	Our Project used Deep Learning Techniques for the prediction. In Deep Learning, CNN Algorithm is chosen and given priority over other algorithms and implemented in the model. 
•	Building such types of systems will be enormously helpful to patients, doctors, and hospitals. Since the system predicts these diseases at a very early stage, doctors can prescribe the medicines to prevent them permanently at the early stage thereby reducing the pneumonia mortality rate. 
•	In conclusion, PneumoVision represents a significant leap forward in the application of deep learning to medical diagnostics. By offering an automated, reliable, and fast pneumonia detection system, it holds the potential to improve patient care and outcomes significantly. 
•	The project's success underscores the importance of continued research and innovation in leveraging artificial intelligence to address critical healthcare challenges. 



