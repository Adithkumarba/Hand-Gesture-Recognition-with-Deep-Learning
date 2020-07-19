# Hand-Gesture-Recognition-with-Deep-Learning
A dynamic hand gesture recognition system which takes in live video input from the webcam and recognizes the dynamic gesture performed by the user. 
The Model is a 3D CNN model built using Keras and tensorflow.
Dataset used is 20bn-jester dataset https://20bn.com/datasets/jester.
The front end GUI is built with OpenCV library.


The Model detects 8 different gestures :
- Swiping left
- Swiping right
- Zooming in with full hand
- Zooming out with full hand
- Thumbs up
- Thumbs down
- Doing other things
- No gesture

Download the dataset from the 20bn website and save it in a folder named 20bn-jester-v1. Out of the total 27 classes available in the dataset, only 8 classes were chosen and trained which was suitable to the GPU ( GTX 1050 ) and computation power available. More gestures can be trained for higher epochs if a more powerful GPU is available.



The file contents:

1. Trainer.py - This file contains the 3D CNN Model build using Keras and Tensorflow. It also has the Data generator which is used to supply data to the model in the required format. On execution, the model is trained with the data loaded from the dataset and the trained model is saved in the form of .h5 file and .json file

2. additionaltrain.py - This file is similar to the trainer. Was used to train additional epochs on the data

3. gui.py - This file contains the GUI application built using OpenCV library. This starts recording live video stream from the webcam. The trained model is imported from the .h5 and .json files and used to recognize the dynamic gestures performed by the user.

4. annotations folder - This folder contains the .csv files which contain the folder names in the dataset and the gesture of each folder. Each folder has 30-70 frames, with an average frame count of 36. These frames correspond to one dynamic gesture being performed.

5. h5 and json files - These files are the saved trained model. The json file contains the model and the h5 file contains the trained weights.

6. The dataset is too large to be uploaded in this repository. Download and store it in a file named 20bn-jester-v1

