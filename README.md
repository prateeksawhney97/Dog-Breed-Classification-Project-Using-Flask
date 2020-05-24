# Dog Breed Classification Project Using Flask

Dog Breed classifier project of the Data Scientist Nanodegree by Udacity. A Web Application is developed using Flask through which a user can check if an uploaded image is that of a dog or human. Also, if the uploaded image is that of a human, the algorithm tells the user what dog breed the human resembles the most. The Deep Learning model distinguishes between the 133 classes of dogs with an accuracy of over 82.89%.

### Home Page

![Screenshot from 2020-05-18 13-58-50](https://user-images.githubusercontent.com/34116562/82191705-8d74f400-9910-11ea-8404-5026fb1585fe.png)

### Uploading an image of a dog

![Screenshot from 2020-05-18 13-59-01](https://user-images.githubusercontent.com/34116562/82191710-8f3eb780-9910-11ea-9682-885a692ca17f.png)

### Prediction using Deep Learning

![Screenshot from 2020-05-18 13-59-09](https://user-images.githubusercontent.com/34116562/82191739-9c5ba680-9910-11ea-825e-534d121f6b4d.png)
![Screenshot from 2020-05-18 13-59-13](https://user-images.githubusercontent.com/34116562/82191744-9f569700-9910-11ea-9ca4-1d58e385df62.png)


## Steps Involved:

1. Import Datasets
2. Detect Humans
3. Detect Dogs
4. Create a CNN to Classify Dog Breeds (from Scratch)
5. Use a CNN to Classify Dog Breeds (using Transfer Learning)
6. Create a CNN to Classify Dog Breeds (using Transfer Learning)
7. Writing the Pipeline
8. Testing the Pipeline
9. Creating a Flask application for the same in which a user can upload an image and see the results.

## Libraries Used:

1. Python 3.7+
2. Keras
3. OpenCV
4. Matplotlib
5. NumPy
6. glob
7. tqdm
8. Scikit-Learn
9. Flask
10. Tensorflow

## Project motivation:
The goal of this project is to classify images of dogs according to their breed. When the image of a human is provided, it recommends the best resembling dog breed. I decided to opt for this project as I found the topic of Deep Neural Networks to be very fascinating and wanted to dive deeper into this with some practical work.

## Description of repository:
The repository consists of the Jupyter Notebook files from the Udacity classroom, in both formats: dog_app.html and dog_app.ipynb. All credits for code examples here go to Udacity. Moreover there are files for the web application developed using Flask and contains all code necessary for running the dog breed classifier app on the local machine.

## Running the code:
For running the web app on local machine, following these instructions:

1. Make sure you have all necessary packages installed.
2. Git clone this repository
3. Within command line, cd to the cloned repo, and within the main repository.
4. Run the following command in the parent directory to run the web application. 

```
python main.py
```

Go to http://0.0.0.0:8080/ to view the web app and input new pictures of dogs or humans – the app will tell you the resembling dog breed successfully without any errors.

## Project Definition:
The task was to develop an algorithm that takes an image as an input, pre-processes and transforms the image so that it can be fed into a CNN for classifying the breed of the dog. If a human image is uploaded, it should still tell the user what dog breed the human resembles most.

## Analysis of the Project:

I decided to use a pre-trained ResNet50 model as this has shown very good results with regard to accuracy for image classification. In the provided classroom environment, my tests showed an a test accuracy of 82.8947%. This was accomplished by 25 epochs which ran very quickly on the provided GPU. Thanks to Udacity! The code in the classroom worked pretty well. What I found difficult, was translating the code into a web app. In the beginning I was getting several errors and identified a different keras version as the root cause of that. Therefore I installed the same version as in the udacity classroom and it worked well afterwards.


### Dataset Exploration:

The datasets are provided by Udacity i.e. dog images for training the models and human faces for detector. After loading both the dataset using sklearn, the following conclusions are drawn:

1. There are 133 total dog categories.
2. There are 8351 total dog images.
3. There are 6680 training dog images.
4. There are 835 validation dog images.
5. There are 836 test dog images.
6. The are in total 13233 human images.

### Review:

1. An example of human detection is provided in the following image:

Human is detected in the following image.

![44](https://user-images.githubusercontent.com/34116562/82108644-89e53f80-974d-11ea-9661-2dd62a57e023.png)


2. Even humans will find it difficult to tell the difference between the two dog classes in some categories. An example is shown below:

![Brittany_02625](https://user-images.githubusercontent.com/34116562/82108456-1db60c00-974c-11ea-89c9-c4397c8bc57b.jpg)

Brittany Breed

![Welsh_springer_spaniel_08203](https://user-images.githubusercontent.com/34116562/82108457-1f7fcf80-974c-11ea-9d4f-6ec00b36b05c.jpg)

Welsh Springer Spaniel Breed

3. Also, more distinguishing/challenging categories are shown.

![final](https://user-images.githubusercontent.com/34116562/82108643-88b41280-974d-11ea-86f9-f64ee078518a.png)


## Conclusion:
I was surprised by the good results of the algorithm i.e. Resnet50. Without doing too much fine-tuning, the algorithm was already providing high accuracy and the predictions were mostly correct. An accuracy of over 80%. For human faces it seems easier if the face has distinct features that resembles a certain dog breed. Otherwise, it starts to guess from some features, but the results vary. For higher accuracy, the parameters could be further optimized, maybe also including more layers into the model. Further, number of epochs could be increased to 40 to lower the loss. Also by providing an even bigger training data set, the classification accuracy could be improved further. Another improvement could be made with regard to UI.

## Results:

Using the final model, some examples of predictions are shown below. If a photo of a human is uploaded, it tells the closest match.

#### Prediction: This photo looks like an Afghan hound.

![1](https://user-images.githubusercontent.com/34116562/82108536-bc426d00-974c-11ea-9c9e-eea43de57701.png)

#### Prediction: The predicted dog breed is a Brittany.

![2](https://user-images.githubusercontent.com/34116562/82108537-be0c3080-974c-11ea-9d92-f73a314f70f0.png)

#### Prediction: The predicted dog breed is a Boykin spaniel.

![3](https://user-images.githubusercontent.com/34116562/82108538-bfd5f400-974c-11ea-9426-3437ace3342a.png)

#### Prediction: The predicted dog breed is a Curly-coated retriever.

![4](https://user-images.githubusercontent.com/34116562/82108540-c19fb780-974c-11ea-9a01-6ad7f33d98cc.png)

#### Prediction: The predicted dog breed is a Labrador retriever.

![5](https://user-images.githubusercontent.com/34116562/82108545-c5333e80-974c-11ea-9b21-8876e669061b.png)

#### Prediction: The predicted dog breed is a Labrador retriever.

![6](https://user-images.githubusercontent.com/34116562/82108549-c82e2f00-974c-11ea-98dc-4372bde8627d.png)

#### Prediction: The predicted dog breed is a Labrador retriever.

![7](https://user-images.githubusercontent.com/34116562/82108551-ca908900-974c-11ea-938f-8dfd4bb95c17.png)

### Prediction on a sample image using the web application

![Screenshot from 2020-05-18 13-59-23](https://user-images.githubusercontent.com/34116562/82191755-a2518780-9910-11ea-9180-bf5c3bd84ccc.png)
![Screenshot from 2020-05-18 13-59-26](https://user-images.githubusercontent.com/34116562/82191761-a4b3e180-9910-11ea-926b-a57ad42384f9.png)


