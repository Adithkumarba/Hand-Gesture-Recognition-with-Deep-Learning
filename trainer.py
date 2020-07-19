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
import joblib

gesture_list = ['Swiping Right','Swiping Left','Thumb Up','Thumb Down','No gesture','Zooming In With Full Hand','Zooming Out With Full Hand']
#train
file_prefix = "new_with_thumbs"
df = pd.read_csv('./annotations/jester-v1-train.csv',sep=';',header=None,names=['id','labels'])
df = df[df['labels'].isin(gesture_list)]
df.to_csv('{}_train.csv'.format(file_prefix),sep=';',index=False)
# pd.read_csv('jester-v1-8classes_train.csv',sep=";").head()
print(df.shape)
print(df.head())
#val
df = pd.read_csv('./annotations/jester-v1-validation.csv',sep=';',header=None,names=['id','labels'])
df = df[df['labels'].isin(gesture_list)]
df.to_csv('{}_val.csv'.format(file_prefix),sep=';',index=False)
print(df.shape)
print(df.head())

#print('Csvs done')

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
        if "train" in file_path:
            self.encoder = OneHotEncoder(sparse=False)
            self.encoder.fit(self.df.labels.values[:,None])
            joblib.dump(self.encoder,"{}_encoder_joblib.joblib".format('_'.join(file_path.split('_')[:-1])))
            np.save("encoder_classes_{}_npy.npy".format(n_classes),self.encoder.categories_)
        else:
            self.encoder = joblib.load("{}_encoder_joblib.joblib".format('_'.join(file_path.split('_')[:-1])))
        self.base_dir = base_dir
        self.on_epoch_end()

    def __len__(self):
        ## Decides step_size
        return self.df.shape[0] // self.batch_size

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = self.df.loc[indexes,"id"].to_list()
        X, y = self.__data_generation(indexes)
        return X, y 

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.df.shape[0])
        np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        X = np.empty((self.batch_size,self.frames_count, *self.image_dim, self.n_channels))
        y = np.empty((self.batch_size,1), dtype=str)
        y = []
        for i, ID in enumerate(indexes):
            files_list = self.standardize_frame_count(glob(self.base_dir + self.df.loc[ID,"id"] + "/*.jpg"),self.df.loc[ID])
            for idx,filename in enumerate(files_list):
                X[i,idx] = tf.keras.preprocessing.image.img_to_array(tf.keras.preprocessing.image.load_img(filename,color_mode='grayscale',target_size=self.image_dim))
            y.append(self.df.loc[ID,"labels"])
        encoded = self.encoder.transform(np.array(y)[:,None])
        return X,encoded
        
    def standardize_frame_count(self,files,error_check):
        shape = len(files)
        if shape < self.frames_count:
            to_add = self.frames_count - shape
            mid  = len(files)//2
            dup = [files[mid]]*to_add
            files = files[:mid] + dup + files[mid+1:]
        elif shape > self.frames_count:
            to_remove = (shape - self.frames_count)
            to_remove = int(to_remove) if int(to_remove) == to_remove else int(to_remove) + 1
            files = files[to_remove:]
        return files
    
params = {'batch_size': 56,
          'n_classes': len(gesture_list),
          'n_channels': 3,
          'image_dim': (32,32)
          }

# Generators
training_generator = DataGenerator(file_path="{}_train.csv".format(file_prefix),**params)
validation_generator = DataGenerator(file_path="{}_val.csv".format(file_prefix),**params)

def build_model(n_classes=6):
    momentum = 0.99
    
    model = tf.keras.Sequential()
    model.add(Conv3D(64,kernel_size=3,strides=1,padding='valid',activation='elu'))
    model.add(BatchNormalization(momentum=momentum))
    model.add(MaxPool3D(pool_size=(1, 2, 2), strides=(1,2,2),padding='same'))

    model.add(Conv3D(128,kernel_size=3,strides=1,padding='valid',activation='elu'))
    model.add(BatchNormalization(momentum=momentum))
    model.add(MaxPool3D(pool_size=(2, 2, 2), strides=(2,2,2),padding='same'))

    model.add(Conv3D(256,kernel_size=3,strides=1,padding='valid',activation='elu'))
    model.add(BatchNormalization(momentum=momentum))
    model.add(MaxPool3D(pool_size=(2, 2, 2), strides=(2,2,2),padding='same'))


    model.add(Conv3D(256,kernel_size=3,strides=1,padding='valid',activation='elu'))
    model.add(BatchNormalization(momentum=momentum))
    model.add(MaxPool3D(pool_size=(2, 2, 2), strides=(2,2,2),padding='same'))


    model.add(Flatten())

    model.add(Dense(512,activation='elu'))
    model.add(Dense(n_classes,activation='softmax'))
    #batch_size,n_classes
    return model

model = build_model(n_classes=len(gesture_list))
optimizer = SGD(0.001)
model.compile(
    optimizer,
    'categorical_crossentropy',
    metrics=['acc']
)

model.fit(training_generator,validation_data=validation_generator,validation_steps=32,epochs=5,verbose=1)

def save_model(model,file_path):
    model_json = model.to_json()
    with open(file_path+'.json', "w") as json_file:
        json_file.write(model_json)
    model.save_weights(file_path+'.h5')
save_model(model,'{}_model'.format(file_prefix))


model.evaluate(validation_generator)