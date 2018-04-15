from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, AveragePooling2D, Flatten, BatchNormalization, ZeroPadding2D, Convolution2D, Merge
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import SGD

# import AGNetConfig
import os
from keras.applications.vgg16 import VGG16
from keras import Model

X_train = 0
y_train = 0

(channel, img_rows, img_cols) = (3, 64, 64)
num_classes = 128

model = Sequential()
model.add(ZeroPadding2D((1, 1), input_shape=(img_rows, img_cols, channel)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

# Loads ImageNet pre-trained data
model.load_weights('/home/vmc/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

# Truncate and replace softmax layer for transfer learning
# Add Fully Connected Layer
model.add(Flatten())
model.add(Dense(4096, activation='relu'))

model.layers.pop()
model.outputs = [model.layers[-1].output]
model.layers[-1].outbound_nodes = []
model.add(Dense(num_classes, activation='sigmoid'))

print(model.summary())

# Uncomment below to set the first 10 layers to non-trainable (weights will not be updated)
for layer in model.layers[:10]:
    layer.trainable = True

model.compile(optimizer='Adam', loss='mean_square_error', metrics=['mse'])

model.fit(x=[X_train, y_train], y=y_train, batch_size=200, 
        verbose=1, 
        shuffle=True, 
        validation_data=[[X_test, y_test], y_test])