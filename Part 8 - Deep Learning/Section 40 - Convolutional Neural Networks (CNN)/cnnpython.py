#CNN

#Building the CNN

#Importing the libraries
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initializing the CNN
classifier = Sequential()

#Convolution
classifier.add(Convolution2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

#Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Flattening
classifier.add(Flatten())

#Full Connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting to the CNN

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                  target_size=(64, 64),
                                                  batch_size=32,
                                                  class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                    steps_per_epoch=8000/32,
                    epochs=25,
                    validation_data=test_set,
                    validation_steps=2000/32)


#Predicting the real results

from skimage.io import imread
from skimage.transform import resize
import numpy as np

class_labels = {v: k for k, v in training_set.class_indices.items()}

img = imread("./cat.jpg") #make sure that path_to_file contains the path to the image you want to predict on. 
img = resize(img,(64,64))
img = np.expand_dims(img,axis=0)

if(np.max(img)>1):
    img = img/255.0 

prediction = classifier.predict_classes(img)
 
print (class_labels[prediction[0][0]])