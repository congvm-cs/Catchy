from keras.models import Sequential
from keras.layers import ZeroPadding2D, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import SGD
import utils


def __vgg16_model__():
    """VGG 16 Model 
    Parameters:
    img_rows, img_cols - resolution of inputs
    channel - 1 for grayscale, 3 for color 
    num_classes - number of categories for our classification task
    """
    (channel, img_rows, img_cols) = (3, 64, 64)
    num_classes = 6

    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(img_rows, img_cols, channel)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    # Loads ImageNet pre-trained data
    model.load_weights('/content/Smart-Advertising-Systems/Models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

    # Truncate and replace softmax layer for transfer learning
    # Add Fully Connected Layer
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.2))
    # model.add(Dense(4096, activation='relu'))
    # model.add(Dropout(0.2))
    # model.layers.pop()
    # model.outputs = [model.layers[-1].output]
    # model.layers[-1].outbound_nodes = []
    model.add(Dense(num_classes, activation='sigmoid'))
    # model.load_weights('/content/Smart-Advertising-Systems/Models/AGNet_weights_1-improvement-30-0.22-0.90.hdf5')

    print(model.summary())

    # Uncomment below to set the first 10 layers to non-trainable (weights will not be updated)
    # for layer in model.layers[:10]:
    #     layer.trainable = True

    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.99, nesterov=True)
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

    return model



# Load image

dgData = utils.DeepFashionDataset()
[images, labels] = dgData.load_dataset('/content/data', True)

model = __vgg16_model__()

from sklearn.model_selection import train_test_split
[X_train, X_test, y_train, y_test] = train_test_split(images, labels, random_state=0, test_size=0.1)

model.fit(x=X_train, y=y_train, batch_size=500, 
                                epochs=100,
                                validation_data=(X_test, y_test))


