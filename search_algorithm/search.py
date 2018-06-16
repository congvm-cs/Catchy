import glob
import hdidx
import pickle
import keras
from keras.models import model_from_json
import keras.backend as K 
import cv2
import numpy as np
import matplotlib.pyplot as plt  
i = 0


# print(index_paths)


data_embedded = 'embedded.pkl'
idx = 'Indexer_8.pkl'
querry_path = '/Users/ngocphu/Documents/Deep Fashion/search algorithm/output_sorted/img/MEN/Jackets_Vests/id_00000168/443_02_1_front.jpg'
image_path = sorted(glob.glob('/Users/ngocphu/Documents/Deep Fashion/search algorithm/output_sorted/*/*/*/*/*.jpg'))
# print(len(image_path))
results = []
with open('model.json', 'r') as f:
    model = model_from_json(f.read())
model.load_weights('Catchy_train-weights_1-6hdf5')
input_1 = model.input
embedded_output = model.get_layer('embedded_layer').output
get_output = K.function(inputs = [input_1], outputs = [embedded_output])
IMAGE_SIZE = 128

# def find_pic(ids):
#     pic_path = image_path[ids]
#     return pic_path

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


query_img = __read_image(querry_path)
query_img1 = query_img.reshape((-1, query_img.shape[0], query_img.shape[1], 3))
query_embedded = np.array(get_output([query_img1])).reshape((1,1024))
# print(query_embedded)
pos = 0

with open(data_embedded, 'rb') as file:
    database = np.array(pickle.load(file))
    print(database.shape)
    database = database.reshape((52712,1024))


with open(idx, 'rb') as fi:
    indexer = pickle.load(fi)
    # indexer.add(database)
    ids, dis = indexer.search(query_embedded, 3)
querry = cv2.imread(querry_path)
querry = cv2.cvtColor(querry, cv2.COLOR_BGR2RGB)
print(ids)


# i = ids[-1][1]
# print(i)
# pic_1 = image_path[i]
# pic = cv2.imread(pic_1)
# cv2.imshow('find',pic)
# print(pic.shape)

# cv2.waitKey()
index = 1
#rint(results)
for i in ids[-1]:
    print(i)
    plt.subplot(2, 2, index)
    pic = image_path[i]
    pic = cv2.imread(pic)
    pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
    plt.imshow(pic)
    plt.title('Result: ' + str(index + 1))
    index += 1

plt.subplot(2, 2, 4)
plt.imshow(querry)
plt.title('Retrieval')
plt.savefig('example_8_1.png')
plt.show()

# dtype = [('pos', int), ('ids', int), ('dis', float)]
# #results = np.array(results)
# results= np.array(results, dtype=dtype)


# fitest = np.sort(results, order='dis')

# print(fitest)  




    