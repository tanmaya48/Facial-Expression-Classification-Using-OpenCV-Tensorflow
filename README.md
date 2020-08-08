# Facial-Expression-Classification-Using-OpenCV-Tensorflow
Using OpenCV and Tensorflow to make a Classify facial expressions. Only differentiates between neutral and smiling face, as a proof of concept


Step 1 - Creating Training Dataset

We use taking_pictures.py to get lots of Pictures of a facial expression through the WebCam input.
It uses a HAAR cascade face detector and saves the image of the face, multiple times in a second...


Step 2 - Training the model

Before we can train the model, we have to prepare the data

the training data is divided into training and testing subsets
into directories as

training:
data/smile/
data/neutral/

testing:
data/test/smile/
data/test/neutral/

After that, we run the data_pickling.ipynb to convert the data into pickle files so that it becomes easier to use

Next, in the training_model.ipynb we have trained the tensorflow model.


Step 3 - Bringing things together

Finally, in expression_classification.py we load the tensorflow model to classify expressions  


