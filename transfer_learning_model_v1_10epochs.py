from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout

# Helper libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os

print(tf.__version__)

import pathlib
data_root = pathlib.Path(r'##PATH TO YOUR TRAINING IMAGES##')
test_root = pathlib.Path(r'##PATH TO YOUR TESTING IMAGES##')
print(data_root)
print(test_root)

import random
all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)
image_count = len(all_image_paths)
print(image_count)

all_test_paths = list(test_root.glob('*/*'))
all_test_paths = [str(path) for path in all_test_paths]
random.shuffle(all_test_paths)

test_count = len(all_test_paths)
print(test_count)

image_size = 192 # All images will be resized to 192x192
batch_size = 32

# Rescale all images by 1./255 and apply image augmentation
train_datagen = keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255,
                validation_split=0.2)

# These convert the images to tensors to feed into the neural network
train_generator = train_datagen.flow_from_directory(
    directory=data_root,
    target_size=(image_size, image_size),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42,
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    directory=data_root,
    target_size=(image_size, image_size),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42,
    subset='validation'
)

test_datagen = keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    directory=test_root,
    target_size=(image_size, image_size),
    color_mode="rgb",
    batch_size=1,
    class_mode=None,
    shuffle=False,
    seed=42
)
# This section isn't vital but allows performance monitoring and graph plotting of training data over time
class CollectBatchStats(tf.keras.callbacks.Callback):
  def __init__(self):
    self.batch_losses = []
    self.batch_acc = []

  def on_train_batch_end(self, batch, logs=None):
    self.batch_losses.append(logs['loss'])
    self.batch_acc.append(logs['acc'])
    self.model.reset_metrics()
    
IMG_SHAPE = (image_size, image_size, 3)

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False
base_model.summary()

# Add a new classifier layer on top
model = tf.keras.Sequential([
  base_model,
  keras.layers.GlobalAveragePooling2D(),
  keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()


batch_stats_callback = CollectBatchStats()

checkpoint_path = r"##CHECKPOINT SAVE PATH##\cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

epochs = 10
steps_per_epoch = train_generator.n // batch_size
validation_steps = validation_generator.n // batch_size

# Use the generators to fit the model 
history = model.fit_generator(train_generator,
                              steps_per_epoch = steps_per_epoch,
                              epochs=epochs,
                              workers=4,
                              validation_data=validation_generator,
                              validation_steps=validation_steps,
                              callbacks = [cp_callback, batch_stats_callback])

model.save(r'##TRAINED MODEL SAVE PATH##\my_model.h5')

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=validation_generator.n//validation_generator.batch_size

model.evaluate_generator(generator=validation_generator,
steps=STEP_SIZE_VALID)

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
test_generator.reset()
pred=model.predict_generator(test_generator,
steps=STEP_SIZE_TEST,
verbose=1)

predicted_class_indices=np.argmax(pred,axis=1)

labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

#Create a new CSV of the file name and it's predicted class. This code is handy for Kaggle submissions
filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results.to_csv(r"##RESULTS CSV SAVE PATH##\results.csv",index=False)
