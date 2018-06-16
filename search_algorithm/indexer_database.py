# import keras
from keras.datasets import mnist
import numpy as np
from keras.models import load_model
import keras.backend as K 
import pickle
import hdidx
import glob


# database = []
p = 'embedded.pkl'
i = 0 

with open(p, 'rb') as file:
    database = np.array(pickle.load(file))
    print(database.shape)
    database = database.reshape((52712,1024))
    print(database.shape)
    idx = hdidx.indexer.IVFPQIndexer()
    # build indexer
    idx.build({'vals': database, 'nsubq': 8})
    # # add database items to the indexer
    idx.add(database)
    with open('Indexer_8.pkl', 'wb') as f:
        pickle.dump(idx, f, pickle.HIGHEST_PROTOCOL)
    print('Done')
""""
model = load_model('mnist_model.h5')
print(model.summary())


(x_train, y_train), (x_test, y_test) = mnist.load_data()


print(x_train.shape)

# x_train = x_train.reshape((-1, 28, 28, 1))
input_l = model.input
output_l = model.layers[5].output

get_output = K.function(inputs=[input_l], outputs=[output_l])

# # Create database
# for img in x_train[0]:
#     img = img.reshape((-1, 28, 28, 1))

#     embedded_arr = get_output(img)

#     print(embedded_arr.shape)

img = x_test[10, :, :]

img_test = img / 255.0
img_test = img_test.reshape((-1, 28, 28, 1))

query_arr = np.array(get_output([img_test])).flatten()
query_arr =query_arr.reshape([1,128])
print(query_arr.shape)


database = np.array(database)
print(database.shape)


# # Product quantization

with open('hdidx_wt_0.pkl', 'rb') as file:
    idx = pickle.load(file)


# idx = hdidx.indexer.IVFPQIndexer()
# # build indexer
idx.build({'vals': database1, 'nsubq': 8})
print('xong')
# # add database items to the indexer
# idx.add(database)
# # searching in the database, and return top-10 items for each query
# ids, dis = idx.search(query_arr, 5)

# with open('hdidx_wt_0.pkl', 'wb') as file:
#     pickle.dump(idx, file, pickle.HIGHEST_PROTOCOL)


idx.add(database)

# print('done')
ids, dis = idx.search(query_arr, 3)

import matplotlib.pyplot as plt

index = 1
for i in ids[-1]:
    print(i)
    print(x_train.shape)
    plt.subplot(2, 2, index)
    # im = np.reshape(    x_train[i, 28, 28]
    plt.imshow(x_train[i, :, :])
    plt.title('Result: ' + str(index + 1))
    index += 1

plt.subplot(2, 2, 4)
plt.imshow(img)
plt.title('Retrieval')
plt.show()
"""