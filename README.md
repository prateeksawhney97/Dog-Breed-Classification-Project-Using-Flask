# Dog Breed Classification Project Using Flask

Dog Breed classifier project of the Data Science Nanodegree by Udacity. A Web Application is developed using Flask through which a user can check if an uploaded image is that of a dog or human. And, the Deep Learning model also distinguishes between the 133 classes of dogs with an accuracy of over 80% with the help of Transfer Learning.


![Screenshot from 2020-05-16 01-59-04](https://user-images.githubusercontent.com/34116562/82094678-2a6e3c00-971b-11ea-832b-36620650bff5.png)

## Steps Involved:
Step 0: Import Datasets
Step 1: Detect Humans
Step 2: Detect Dogs
Step 3: Create a CNN to Classify Dog Breeds (from Scratch)
Step 4: Use a CNN to Classify Dog Breeds (using Transfer Learning)
Step 5: Create a CNN to Classify Dog Breeds (using Transfer Learning)
Step 6: Write your Algorithm
Step 7: Test Your Algorithm

## Libraries Used:
1. Python 3.7+
2. Keras==2.0.9
3. OpenCV
4. Matplotlib
5. NumPy
6. glob
7. tqdm
8. Scikit-Learn
9. Flask
10. Tensorflow 2.0

## Project motivation
The goal of this project is to classify images of dogs according to their breed. When the image of a human is provided, it should recommend the best resembling dog breed. I decided to opt for this project as I found the topic of Deep Neural Networks to be very fascinating and wanted to dive deeper into this with some practical work.

## Description of repository:
The repository consists of the Jupyter Notebook files from the Udacity classroom, in both formats: dog_app.html and dog_app.ipynb. All credits for code examples here go to Udacity. Moreover there are files for the web_app developed using Flask and contains all code necessary for running the dog breed classifier app on your local machine.

## Running the code:
For running the web app on local machine, following these instructions:

1. Make sure you have all necessary packages installed (if version is specified, then please refer to the one mentioned above for running the code without errors on your machine).
2. Git clone this repository
3. Within command line, cd to the cloned repo, and within the main repository.
4. Run the following command in the parent directory to run the web application. 

```
python main.py
```

Go to http://0.0.0.0:8080/ to view the web app and input new pictures of dogs or humans – the app will tell you the resembling dog breed successfully without any errors. The app might give some errors with TensorFlow 1.0+ versions, but it runs smoothly with TensorFlow 2.0.

## Project Definition:
The task was to develop an algorithm that takes an image as an input, pre-processes and transforms the image so that it can be fed into a CNN for classifying the breed of the dog. If a human image is uploaded, it should still tell the user what dog breed the human resembles most.

## Analysis of the Project:

I decided to use a pre-trained ResNet50 model as this has shown very good results with regard to accuracy for image classification. In the provided classroom environment, my tests showed an a test accuracy of 82.8947%. This was accomplished by 25 epochs which ran very quickly on the provided GPU. Thanks to Udacity! The code in the classroom worked pretty well. What I found difficult, was translating the code into a web app. In the beginning I was getting several errors and identified a different keras version as the root cause of that. Therefore I installed the same version as in the classroom and it worked well afterwards. Also, there were errors with version of tensorflow less than 2.0. Updated tensorflow and all things worked very well.

### Dataset Exploration:

The datasets are provided by Udacity i.e. dog images for training the models and human faces for detector. After loading both the dataset using sklearn, the following conclusions are drawn:

1. There are 133 total dog categories.
2. There are 8351 total dog images.
3. There are 6680 training dog images.
4. There are 835 validation dog images.
5. There are 836 test dog images.
6. The are in tota l13233 human images.

Even humans will find it difficult to tell the difference between the two dog classes in some categories. An example is shown below:

![Brittany_02625](https://user-images.githubusercontent.com/34116562/82108456-1db60c00-974c-11ea-89c9-c4397c8bc57b.jpg)

Brittany Breed

![Welsh_springer_spaniel_08203](https://user-images.githubusercontent.com/34116562/82108457-1f7fcf80-974c-11ea-9d4f-6ec00b36b05c.jpg)

Welsh Springer Spaniel Breed


## Conclusion:
I was surprised by the good results of the algorithm i.e. Resnet50. Without doing too much fine-tuning, the algorithm was already providing high accuracy and the predictions were mostly correct. An accuracy of over 80%. For human faces it seems easier if the face has distinct features that resembles a certain dog breed. Otherwise, it starts to guess from some features, but the results vary. For higher accuracy, the parameters could be further optimized, maybe also including more layers into the model. Further, number of epochs could be increased to 40 to lower the loss. Also by providing an even bigger training data set, the classification accuracy could be improved further. Another improvement could be made with regard to UI. It might pose some problems when deployed using GCP Or Azure. 

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

