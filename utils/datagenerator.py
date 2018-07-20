import numpy as np
from keras.utils import Sequence
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import random_rotation, random_zoom, flip_axis
from keras.preprocessing.image import random_brightness, random_shear, random_shift
from configs.configs import *
import cv2


class DataGenerator(Sequence):
  'Generates data for Keras'

  def __init__(self, list_IDs, labels=None, batch_size=32, dim=(32, 32), n_channels=1,
               n_classes=10, shuffle=True, training=False):
    
    'Initialization'
    self.dim = dim                     
    self.batch_size = batch_size
    self.labels = labels
    self.list_IDs = list_IDs
    self.n_channels = n_channels
    self.n_classes = n_classes
    self.shuffle = shuffle
    self.__on_epoch_end()
    self.training = training


  def __len__(self):
    'Denotes the number of batches per epoch'
    return int(np.floor(len(self.list_IDs) / self.batch_size))


  def __getitem__(self, index):
    'Generate one batch of data'
    # Generate indexes of the batch
    indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

    # Find list of IDs
    list_IDs_temp = [self.list_IDs[k] for k in indexes]

    # Generate data
    X, y = self.__data_generation(list_IDs_temp)
    return X, y


  def __on_epoch_end(self):
    'Updates indexes after each epoch'
    self.indexes = np.arange(len(self.list_IDs))
    if self.shuffle == True:
        np.random.shuffle(self.indexes)
        
        
  def __read_image(self, path):
    img = load_img(path)
    img = img_to_array(img)
    
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    
    if self.training == True:
      img = self.__augment_data(img)
      
    img = img/255.0
    img = self.__prewhiten(img)
    return img

  
  def __read_label(self, path):
    label = path.split('/')[-2]
    return label
  
  
  def __prewhiten(self, x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y 
  
  
  def __categorize_labels(self, file_path):
    NUM_LABELS = 21

    STYLES = ['Denim', 'Jackets_Vests', 'Pants', 'Shirts_Polos', 'Shorts', 
              'Suiting', 'Sweaters', 'Sweatshirts_Hoodies', 'Tees_Tanks', 
              'Blouses_Shirts', 'Cardigans', 'Dresses','Graphic_Tees',
              'Jackets_Coats', 'Leggings', 'Rompers_Jumpsuits', 'Skirts']
    
    OUTFITS_TOP = ['Jackets_Vests', 'Sweaters', 'Shirts_Polos', 'Shorts', 'Suiting', 
                   'Blouses_Shirts', 'Sweatshirts_Hoodies', 'Tees_Tanks', 
                   'Cardigans', 'Graphic_Tees', 'Jackets_Coats'] 

    OUTFITS_BOTTOM = ['Denim', 'Pants', 'Leggings', 'Dresses']
    OUTFITS_FULL = ['Skirts', 'Rompers_Jumpsuits']
    
    labels = np.zeros((1, NUM_LABELS))  # 0:     gender
                                        # 1-18 : style
    
    # file_path: /content/img/MEN/Denim/id_00002243/01_1_front.jpg 
    style = file_path.split('/')[-3]
    gender = file_path.split('/')[-4]
    
    # Gender
    if gender == 'WOMEN':
      labels[0, 0] = 1   # Female
    else:
      labels[0, 0] = 0   # Male
    
    # Position
    # Top, Bottom, Full
    if style in OUTFITS_TOP:
      labels[0, 1] = 1 
    if style in OUTFITS_BOTTOM:
      labels[0, 2] = 1 
    if style in OUTFITS_FULL:
      labels[0, 3] = 1 
    
    # Style
    for idx, style_in_arr in enumerate(STYLES):
      if style == style_in_arr:
        labels[0, idx+4] = 1
    labels = np.asarray(labels).reshape(-1)
    return labels

  def __augment_data(self, image):
    """
    if np.random.random() > 0.5:
        images[i] = random_crop(images[i],4)
    """
    if np.random.random() > 0.75:
        image = random_rotation(image, 20, row_axis=0, col_axis=1, channel_axis=2)
    if np.random.random() > 0.75:
        image = random_shear(image, 0.2, row_axis=0, col_axis=1, channel_axis=2)
    if np.random.random() > 0.75:
        image = random_shift(image, 0.2, 0.2, row_axis=0, col_axis=1, channel_axis=2)
    if np.random.random() > 0.75:
        image = random_zoom(image, [0.8, 1.2], row_axis=0, col_axis=1, channel_axis=2)
    if np.random.random() > 0.5:
        image = flip_axis(image, axis=1)
    return image


  def __data_generation(self, list_IDs_temp):
    'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
    # Initialization
    X = np.empty((self.batch_size, *self.dim, self.n_channels))
    y = np.empty((self.batch_size, self.n_classes), dtype=int)
  
    # Generate data
    for i, ID in enumerate(list_IDs_temp):
        # Store sample
        X[i,] = self.__read_image(ID)

        # Store class
        y[i] = self.__categorize_labels(ID)

    return X, [y[:, 0], y[:, 1:4], y[:, 4::]]


# Generators
def generating(train_paths, test_paths, train_params, val_params):
    train_params = train_params
    val_params = val_params

    training_generator = DataGenerator(train_paths, **train_params)
    validation_generator = DataGenerator(test_paths, **val_params)

    return training_generator, validation_generator