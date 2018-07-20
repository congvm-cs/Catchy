from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
from keras.models import Input, Model
from configs.configs import *

class CatchyNet():
    
    def __init__(self):        
        # self.__build__(verbose=True)
        self.model = None
    

    def build(self, verbose=False):
        # Construct Model
        l_input = Input(IMAGE_SHAPE)
        x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(l_input)
        x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = MaxPooling2D(strides=(2, 2))(x)
        x = BatchNormalization()(x)

        x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = MaxPooling2D(strides=(2, 2))(x)
        x = BatchNormalization()(x)

        x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = MaxPooling2D(strides=(2, 2))(x)
        x = BatchNormalization()(x)

        x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = MaxPooling2D(strides=(2, 2))(x)
        x = BatchNormalization()(x)

        x = Flatten()(x)
        x = Dropout(0.2)(x)

        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.2)(x)

        l_output = Dense(1024, activation='relu', name='embedded_layer')(x)
        l_output = Dropout(0.2)(l_output)

        gender_output = Dense(BINARY_GENDER_OUTPUTS, activation='sigmoid', name='gender_output')(l_output)
        position_output = Dense(CATEGORICAL_POSITION_OUTPUTS, activation='softmax', name='position_output')(l_output)
        style_output = Dense(CATEGORICAL_STYLE_OUTPUTS, activation='softmax', name='style_output')(l_output)

        self.model = Model(inputs=l_input, outputs=[gender_output, position_output, style_output])

        if verbose == True:
            print(self.model.summary())
    

    def compile(self):
        self.model.compile(optimizer='Adam', 
                        loss={'gender_output': 'binary_crossentropy', 
                            'position_output': 'categorical_crossentropy',
                            'style_output': 'categorical_crossentropy'}, 
                        metrics=['accuracy'],
                        loss_weights={'gender_output': 1.0, 
                                    'position_output': 1.0,
                                    'style_output': 1.0})


    def fit(self, training_generator, validation_generator, callback_list):
        self.model.fit_generator(training_generator,
                    verbose=1,
                    epochs=50,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=10, 
                    callbacks=callback_list)