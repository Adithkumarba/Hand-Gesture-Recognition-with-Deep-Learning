import seaborn as sn
from matplotlib import pyplot as plt
import traceback
import tensorflow as tf
import pandas as pd
from sklearn import metrics
import joblib
from glob import glob
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json

json_file = open(glob('./model/*.json')[0], 'r')
model = json_file.read()
json_file.close()
model = model_from_json(model)
model.load_weights(glob('./model/*.h5')[0])
classes = np.load(glob('./model/encoder*.npy')[0],allow_pickle=True).tolist()[0]

gesture_list = classes
file_prefix = "new_with_thumbs"
# df = pd.read_csv('./annotations/jester-v1-validation.csv',sep=';',header=None,names=['id','labels'])
# df = df[df['labels'].isin(gesture_list)]
# df.to_csv('{}_val.csv'.format(file_prefix),sep=';',index=False)
# print('csv done')
df = pd.read_csv(file_prefix+'_val.csv',sep=";")
df.id = df.id.map(str)
df = df.iloc[:1000]        

def _data_generation(batch_size,image_dim,n_channels,df,frames_count):
    base_dir = './20bn-jester-v1/'
    try:
        X = np.empty((batch_size,frames_count, *image_dim, n_channels))
        y = []
        for i, ID in enumerate(df.index.values.tolist()):
            files_list = standardize_frame_count(glob(base_dir + str(df.loc[ID,"id"]) + "/*.jpg"),df.loc[ID],frames_count)
            for idx,filename in enumerate(files_list):
                X[i,idx] = tf.keras.preprocessing.image.img_to_array(tf.keras.preprocessing.image.load_img(filename,color_mode='grayscale',target_size=image_dim))
            y.append(df.loc[ID,"labels"])
        return X,y
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        return 1,2

def standardize_frame_count(files,error_check,frames_count):
    try:
        shape = len(files)
        if shape < frames_count:
            to_add = frames_count - shape
            mid  = len(files)//2
            dup = [files[mid]]*to_add
            files = files[:mid] + dup + files[mid+1:]
        elif shape > frames_count:
            to_remove = (shape - frames_count)
            to_remove = int(to_remove) if int(to_remove) == to_remove else int(to_remove) + 1
            files = files [to_remove:]
        return files
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        print(len(files),files,error_check)
        return []

def iterator_gen(batch_size,image_dim,n_channels,df,n_classes,frames_count=36):
    for i in range(0,df.shape[0],batch_size):
        X,Y = _data_generation(batch_size,image_dim,n_channels,df[i:i+batch_size],frames_count)
        yield X,Y

params = {'batch_size': 56,
          'n_classes': 8,
          'n_channels': 3,
          'image_dim': (32,32),
          'df':df
          }

# Generators
val_generator = iterator_gen(**params)

Y = []
Y_pred = []
loop_count = 0

while True:
    try:
        print(loop_count)
        loop_count += 1
        x,y = next(val_generator)
        y_pred = model.predict(x)
        Y.extend(y)
        Y_pred.extend(y_pred.argmax(axis=1))
    except StopIteration: 
        break

def map_plot(cm, name, width, height, labels):
    plt.subplots(figsize=(width, height))
    plot = sn.heatmap(cm, annot=True, annot_kws={"size": 12}, xticklabels=labels, yticklabels=labels)
    # fig = plot.get_figure()
    # fig.savefig("conf_matrices/" + str(count) + ".png")
    plt.title(name, fontdict={'fontsize': '14', 'color': '#000000'})
    plt.xlabel("Predicted", fontdict={'fontsize': '14'})
    plt.ylabel("Actual", fontdict={'fontsize': '14'})
    plt.show()

print('mapping')
Y_pred = Y_pred[:len(Y)]
Y_pred = list(map(lambda x : classes[x],Y_pred))
print('conf')
with open('predicted.txt','w') as f:
    f.write('#'.join(Y_pred))
with open('actual.txt','w') as f:
    f.write('#'.join(Y))
cm = metrics.confusion_matrix(Y,Y_pred,labels=classes)
map_plot(cm,"Confusion Matrix",width=16,height=12,labels=classes)
