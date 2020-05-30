from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras import metrics
from keras.models import Sequential
from keras.layers import Softmax
from keras.optimizers import Adam
import random



import os

model = Sequential()
model.add(Convolution2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(filters=32,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(filters=32,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

from keras_preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
        '/root/mlops/cnn-cat&dog-dataset/training_set/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
test_set = test_datagen.flow_from_directory(
        '/root/mlops/cnn-cat&dog-dataset/test_set/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

history=model.fit(
        training_set,
        steps_per_epoch=500,
        epochs=5,
        validation_data=test_set,
        validation_steps=500)


# Now we Are Saving Our Model

print(max(history.history['accuracy']))
if (max(history.history['accuracy'])) > .80 :
    model.save('model.h5')


fh = open('/root/mlpos/accuracy.txt','w+')
fh.write (str(history.history['accuracy']))
fh.close()


