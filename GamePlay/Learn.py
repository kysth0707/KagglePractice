"""
Reference
https://codetorial.net/tensorflow/classifying_the_cats_and_dogs.html
"""

import tensorflow as tf
# import pandas as pd
# import numpy as np
import time
import os

def ReturnPos(loc):
	return "E:\\GithubProjects\\KagglePractice\\GamePlay" + str(loc)


TrainDir = ReturnPos("\\Dataset\\Train")
TestDir = ReturnPos("\\Dataset\\Test")

#Train Set
TrainMinecraftDir = os.path.join(TrainDir, 'Minecraft')
TrainAmongUsDir = os.path.join(TrainDir, 'Among Us')

#Test Set
TestMinecraftDir = os.path.join(TestDir, 'Minecraft')
TestAmongUsDir = os.path.join(TestDir, 'Among Us')


ModelDir = ReturnPos("\\Model")
ImageSize = (640, 360)

model = tf.keras.models.Sequential([
	tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(ImageSize[0], ImageSize[1], 3)),
	tf.keras.layers.MaxPooling2D(2,2),
	tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
	tf.keras.layers.MaxPooling2D(2,2),
	tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
	tf.keras.layers.MaxPooling2D(2,2),
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(512, activation='relu'),
	tf.keras.layers.Dense(1, activation='sigmoid')
])

# model.summary()

model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001), loss='binary_crossentropy', metrics = ['accuracy'])




train_datagen = tf.keras.preprocessing.image.ImageDataGenerator( rescale = 1.0/255. )
test_datagen  = tf.keras.preprocessing.image.ImageDataGenerator( rescale = 1.0/255. )


train_generator = train_datagen.flow_from_directory(TrainDir, batch_size=20, class_mode='binary', target_size = ImageSize)
validation_generator = test_datagen.flow_from_directory(TestDir, batch_size=20, class_mode='binary', target_size = ImageSize)

history = model.fit(train_generator, validation_data=validation_generator, epochs=1)

model.save(ModelDir)