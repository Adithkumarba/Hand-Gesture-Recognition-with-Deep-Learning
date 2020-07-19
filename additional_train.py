import joblib
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Conv3D, BatchNormalization, MaxPool3D, Dense, Flatten, Dropout, ConvLSTM2D, LSTM
from tensorflow.keras.models import model_from_json
import pandas as pd
import numpy as np
import tensorflow as tf
from glob import glob
import skimage.transform
from skimage import io
from sklearn.preprocessing import OneHotEncoder


gesture_list = ['Swiping Right','Swiping Left','Swiping Up','Swiping Down','Stop Sign','No gesture','Zooming In With Full Hand','Zooming Out With Full Hand']
#train
file_prefix = "no_thumbs_8"

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, file_path, batch_size=2, image_dim=(256,256), frames_count=36, n_channels=1, base_dir='./20bn-jester-v1/', n_classes=27,validation=False):
        self.image_dim = image_dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = True 
        self.frames_count = frames_count
        self.df = pd.read_csv(file_path,sep=";")
        self.df.id = self.df.id.map(str)
        self.encoder = joblib.load("{}_encoder.joblib".format(''.join(file_path.split('_')[:-1])))
        self.base_dir = base_dir
        self.on_epoch_end()

    def __len__(self):
        ## Decides step_size
        return self.df.shape[0] // self.batch_size

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = self.df.loc[indexes,"id"].to_list()

        # Generate data
        X, y = self.__data_generation(indexes)
        return X, y 

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.df.shape[0])
        np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        try:
            X = np.empty((self.batch_size,self.frames_count, *self.image_dim, self.n_channels))
            y = np.empty((self.batch_size,1), dtype=str)
            y = []
            for i, ID in enumerate(indexes):
                files_list = self.standardize_frame_count(glob(self.base_dir + self.df.loc[ID,"id"] + "/*.jpg"),self.df.loc[ID])
                for idx,filename in enumerate(files_list):
                    # img = io.imread(filename)
                    # resized_image = skimage.transform.resize(img,self.image_dim, preserve_range=True)
                    # X[i,idx] = resized_image
                    X[i,idx] = tf.keras.preprocessing.image.img_to_array(tf.keras.preprocessing.image.load_img(filename,color_mode='grayscale',target_size=self.image_dim))
                y.append(self.df.loc[ID,"labels"])
            encoded = self.encoder.transform(np.array(y)[:,None])
            return X,encoded
        except Exception as e:
            with open('error.txt','w') as f:
                print(len(files_list),files_list,e,file=f)
            # print(len(files_list),files_list,e)
            print(e)
            return 1,2
        
    def standardize_frame_count(self,files,error_check):
        try:
            shape = len(files)
            if shape < self.frames_count:
                to_add = self.frames_count - shape
                mid  = len(files)//2
                dup = [files[mid]]*to_add
                files = files[:mid] + dup + files[mid+1:]
            elif shape > self.frames_count:
                to_remove = (shape - self.frames_count)
                to_remove = int(to_remove) if int(to_remove) == to_remove else int(to_remove) + 1
                files = files [to_remove:]
            return files
        except Exception as e:
            print(len(files),files,error_check)
            return []

params = {'batch_size': 56,
          'n_classes': 8,
          'n_channels': 3,
          'image_dim': (32,32)
          }

# Generators
training_generator = DataGenerator(file_path="{}_train.csv".format(file_prefix),**params)
validation_generator = DataGenerator(file_path="{}_val.csv".format(file_prefix),**params)

def load_model(file_path):
    with open(file_path+'.json') as loaded_model_json:
        model = model_from_json(loaded_model_json.read())
    model.load_weights("{}.h5".format(file_path))
    return model

model = load_model(file_prefix+'_model')

optimizer = SGD(0.001)

model.compile(
    optimizer,
    'categorical_crossentropy',
    metrics=['acc']
)

model.fit(training_generator,validation_data=validation_generator,validation_steps=32,epochs=1,verbose=1)

def save_model(model,file_path):
    model_json = model.to_json()
    with open(file_path+'.json', "w") as json_file:
        json_file.write(model_json)
    model.save_weights(file_path+'.h5')
save_model(model,'{}_model'.format(file_prefix))

model.evaluate(validation_generator)