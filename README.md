## Proposed Work
The objective of this project is to present a system to classify the leaves into the four classes i.e. three types of diseases for corn crop and three classes i.e. two types of diseases and healthy categories for potato crop based on the leaf condition by utilizing deep learning techniques. Transfer learning methods are used for classification and the models are compared based on various evaluation metrics such as precision, recall and accuracy. The various models used for classification are RESNET152, Inception V3, EfficientNetB0, DenseNet121 and Multi-input model (proposed model). An interactive GUI was built for predicting if the newly uploaded image of a corn leaf is gray leaf spot, common rust, blight or healthy and for potato leaf it is early blight, late blight, or healthy.

### Methodology
The proposed methodology for classifying the corn leaf and potato leaf diseases is carried out in five phases. The work flow of the steps involved is shown in Figure 3.1.1.
*  Dataset Collection
*  Data Preprocessing
*      RandomOverSampler
*       ImageDataGenerator
* Implementing the models
* Choosing the best performing model
* Interactive web app for prediction.

![image](https://github.com/yashwanth-alapati/Transfer-Learning/assets/145064639/199836e5-d3c0-4e27-b0b1-75e774b79a24)


### Dataset Collection:
The dataset for the potato leaf diseases was obtained from the PlantVillage dataset, which was published by David. P. Hughes, Marcel Salathe. The datasets include over 50,000 images of 14 crops such as potatoes, grapes, tomato, apples etc.

For the corn leavesTotal number of images = 4185
No. of images in Blight = 1146
No. of images in Common rust =1303
No. of images in Green leaf spot = 574
No. of images in Healthy =1162

For the potato leavesTotal number of images = 2152
No. of images in Early Blight = 1000
No. of images in Late Blight =1000
No. of images in Healthy =152

### Data Preprocessing:
For the corn crop the number of images in the green leaf spot category is 574 and the other categories have more than 1000 each. Imbalanced classifications pose a challenge for predictive modeling as most of the deep learning algorithms will have poor predictive accuracy as there isn‟t enough data for the model to learn the features of the minority class thus the images from the minority class will be classified wrongly
#### RandomOverSampler: (Novelty)
To solve the above problem, RandomOverSampler was used which randomly chooses samples from the minority class and duplicates them with replacement in the dataset. This automatically balances the minority classes with the majority class/classes. The major demerit of this method is that it may cause overfitting of data in few models as the same samples of data are being duplicated and used for training purposes.

![image](https://github.com/yashwanth-alapati/Transfer-Learning/assets/145064639/9340046d-99f7-4bba-8217-e4dd484dafdf)



#### ImageDataGenerator:
It is an image augmentation technique that is used when there is not enough data to train the model. This is a great way to expand the size of our dataset. The transformation techniques used were horizontal_flip, rotation_range, zoom_range, width_shift_range, height_shift_range. Using these techniques, new transformed images from the original dataset will be used to train the model. This also helps in the model learning various images in different angles and perspectives and thus reduces the chances of the model overfitting the data.

### Model Implementation
A convolutional neural network (CNN) is a type of artificial neural network used in imagerecognition and processing that is specifically designed to process pixel data. In this study, several deep learning models, namely RESNET152, InceptionV3,DenseNet121, EfficientB0 and Multi-input model were trained to classify diseases in the potato and corn leaf. The results of these models were evaluated on the basis of a few metrics such as accuracy,precision and recall. Transfer learning from the imagenet dataset was used where the model is pre-trained and the weights are being used for classification of images on a different
problem. All four models were trained on the training dataset and the proposed model which is the combination of EfficientNetB0 and DenseNet121; DensNet121 gave the best results with around 97.79% validation accuracy.
The models were run for 20 epochs; batch size = 64,optimizer used - RMSprop, learning rate = 0.002 , test size = 20%, validation split = 20%.


## Conclusion
In this study, a classification problem is exhibited to detect infected leaves in the earlier stages of the infection so that it can be treated as soon as possible. Considerable data augmentation using ImageDataGenerator and RandomOverSampler is done on the data due to the imbalance in the dataset, and the data is then used to train various transfer learning CNN models, namely RESNET 152, DENSENET 121, EFFICIENTNET B0, INCEPTIONV3, PROPOSED MODEL and the results obtained are evaluated and assessed. Further, with these models, a WebApp GUI has been built to facilitate the use of these models by farmers and other relevant communities. We believe that this work will be a notable contribution to the sector of agriculture. We hope that this work proves useful to the research communities to get familiar with the AI-based detection of potato leaves diseases.
The model developed not only works on the benchmark dataset but can also work on realtime images with decent accuracy. In the future, the same work can be extended to include all classes of plants like tomatoes,bell pepper, etc., and the accuracy of classifying real-time images can be improved.
