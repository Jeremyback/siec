import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.image import imread

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D


my_data_dir = 'C:\\Users\\miga1\\programowanie\\handwriting\\siec\\dane3'

train_path = my_data_dir+'\\train'
test_path = my_data_dir+'\\test'

example = train_path+'\\a01'+'\\a01-014-00-00.png'
example_img = imread(example)
# plt.imshow(example_img)
# print(example_img.shape)
# print(example_img.max())

dim1 = []
dim2 = []

for autor in os.listdir(train_path):
    for image_filename in os.listdir(train_path+'\\'+autor):

        img = imread(train_path+'\\'+autor+'\\'+image_filename)
        d1,d2 = img.shape
        dim1.append(d1)
        dim2.append(d2)

# sns.jointplot(dim1,dim2)
# plt.show()

# print(np.mean(dim1))
# print(np.mean(dim2))

img_shape = (76,208)


'''image_gen = ImageDataGenerator(rotation_range=1,
                               width_shift_range=0.01,
                               height_shift_range=0.01,
                               rescale=1/255,
                               shear_range=0.01,
                               zoom_range=0.01,
                              )

plt.imshow(image_gen.random_transform(example_img))
plt.show()'''


model = Sequential()

model.add(Conv2D(filters=8, kernel_size=(3,3),input_shape=img_shape, activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=16, kernel_size=(3,3),input_shape=img_shape, activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=16, kernel_size=(3,3),input_shape=img_shape, activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())


model.add(Dense(128))
model.add(Activation('relu'))


model.add(Dense(1))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
