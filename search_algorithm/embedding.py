import glob
import os
import keras.backend as K 
from keras.models import model_from_json
import numpy as np 
import tqdm
import keras
import cv2
import pickle
import h5py

IMAGE_SIZE = 128
embedding = []
def __read_image(path):
  img = keras.preprocessing.image.load_img(path)
  img = keras.preprocessing.image.img_to_array(img)

  img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

  img = img/255.0
  img = __prewhiten(img)
  return img

def __prewhiten(x):
  mean = np.mean(x)
  std = np.std(x)
  std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
  y = np.multiply(np.subtract(x, mean), 1/std_adj)
  return y

with open('model.json', 'r') as f:
    model = model_from_json(f.read())
model.load_weights('Catchy_train-weights_1-6hdf5')
input_1 = model.input
embedded_output = model.get_layer('embedded_layer').output
get_output = K.function(inputs = [input_1], outputs = [embedded_output])
file_path = sorted(glob.glob("/Users/ngocphu/Documents/Deep Fashion/search algorithm/output_sorted/*/*/*/*/*.jpg"))

print(len(file_path))
print(file_path[:5])

for image in tqdm.tqdm(file_path):
    img= __read_image(image)
    img = img.reshape((-1, img.shape[0], img.shape[1], 3))
    embedded = get_output([img])
    embedding.append(embedded)

with open("embbeded.pkl", 'wb') as f:
    pickle.dump(embedding, f)
