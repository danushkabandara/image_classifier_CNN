from keras import backend as K
from keras.models import Sequential, Model
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, RMSprop, SGD, Nadam
from keras.applications.resnet50 import ResNet50
#from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
#from keras.applications.inception_v3 import preprocess_input, decode_predictions
from sklearn.metrics import classification_report, confusion_matrix

import resnet
from random_eraser import get_random_eraser
from visual_callbacks import ConfusionMatrixPlotter

from keras.layers import Input, Flatten, Dense, GlobalAveragePooling2D, Dropout
import sklearn.metrics
import matplotlib.pyplot as plt
import numpy as np
import urllib
import keras
import tensorflow as tf
import horovod.keras as hvd
import sys
import json
import time
import itertools
import io

hvd.init()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
K.set_session(tf.Session(config=config))

## main code is from:
## http://agnesmustar.com/2017/05/25/build-vgg16-scratch-part-ii/

datapath = '/home/bandarams/ML/PFA/data/processed/Preforms_FOI/'
target_size=(224,224)
batch_size=64

class_names= ["BAD", "GOOD"]

def get_train_batches(directory, target_size=target_size, batch_size=batch_size, 
                shuffle=False):
    datagen = ImageDataGenerator(#horizontal_flip = True,
                         vertical_flip = False,
			            #preprocessing_function=get_random_eraser(v_l=0, v_h=255, pixel_level=False),
                     
                       # width_shift_range = 0.1,
                      #height_shift_range = 0.2,
                      #zoom_range = 0.2
                        )
    return datagen.flow_from_directory(directory=directory,
                                       target_size=target_size,
                                       batch_size=batch_size,
                                       class_mode='categorical',
                                       shuffle=shuffle)
def get_valid_batches(directory, target_size=target_size, batch_size=batch_size, 
                shuffle=False):
    datagen = ImageDataGenerator()
    return datagen.flow_from_directory(directory=directory,
                                       target_size=target_size,
                                       batch_size=batch_size,
                                       class_mode='categorical',
                                       shuffle=shuffle)
def get_test_batches(directory, target_size=target_size, batch_size=batch_size, 
                shuffle=False):
    datagen = ImageDataGenerator()
    return datagen.flow_from_directory(directory=directory,
                                       target_size=target_size,
                                       batch_size=batch_size,
                                       class_mode='categorical',
                                       shuffle=shuffle)

batches = get_train_batches(datapath+'train', shuffle=True)
valid_batches = get_valid_batches(datapath+'valid', batch_size=400, 
                            shuffle=False)#original batch_size = batch_size*2
test_batches = get_test_batches(datapath+'valid', batch_size=400, 
                            shuffle=False)#get all the test data for conf_mat
                            
if hvd.rank() == 0:
   test_batches, test_labels = next(test_batches)#get the images and labels from generator


if hvd.rank() == 0:
    with open("labels.txt", "w") as labels:
        labels.write(json.dumps(batches.class_indices))

num_classes = len(batches.class_indices)
orig_model = ResNet50(weights='imagenet',include_top=False,classes=num_classes)
#model = resnet.ResnetBuilder.build_resnet_18((3, 224, 224), 2)

#opt = keras.optimizers.Adadelta(0.001)
#opt = SGD(lr=0.1)
opt = Nadam(lr=0.0001)
#opt = Adam(lr = 0.0001, decay = 1e-6)
#opt = RMSprop(lr= 0.0001)
opt = hvd.DistributedOptimizer(opt)

callbacks = [
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    hvd.callbacks.MetricAverageCallback(),            
    #hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=3, verbose=1),
    #keras.callbacks.ReduceLROnPlateau(patience=10, verbose=1),
]

if hvd.rank() == 0:
    callbacks.append(keras.callbacks.TensorBoard(log_dir="./Graph/", histogram_freq=0, batch_size=batch_size, write_graph=True, write_grads=True, write_images=True) )
    plotter = ConfusionMatrixPlotter(X_val=test_batches, classes=class_names, Y_val=test_labels)
    callbacks.append(plotter)
last_layer = orig_model.output
# add a global spatial average pooling layer
x = GlobalAveragePooling2D()(last_layer)
# add fully-connected & dropout layers
x = Dense(512, activation='relu',name='fc-1')(x)
x = Dropout(0.25)(x)
x = Dense(256, activation='relu',name='fc-2')(x)
x = Dropout(0.25)(x)
# a softmax layer for 4 classes
out = Dense(num_classes, activation='softmax',name='output_layer')(x)

model = Model(inputs=orig_model.input, outputs=out)

for layer in model.layers[:-5]: #freeze the layers except for last 10 (for model regularization)
    layer.trainable = False
print(model.summary())
model.compile(optimizer=opt,
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

t0 = time.time()
fit = model.fit_generator(batches, 
                    steps_per_epoch=batches.samples//batch_size//hvd.size()+1, 
                    callbacks=callbacks,
                    nb_epoch=1000,
                    validation_data=valid_batches, 
                    validation_steps=valid_batches.samples//batch_size//hvd.size()+1)

print("fitting time =", time.time() - t0, " at ", hvd.rank())
# serialize model to JSON
# ref: https://machinelearningmastery.com/save-load-keras-deep-learning-models/
# serialize weights to HDF5
if hvd.rank() == 0:
    model_json = model.to_json()
    with open("resnet18FullRes.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("resnet18FullRes.h5")
    print("Saved model to disk")

if (hvd.rank() == hvd.size()-1):
    import pandas as pd
    fname = 'epoch_first.dat'
    pd.DataFrame(fit.history).to_csv(fname,float_format='%.3f', sep=' ')


