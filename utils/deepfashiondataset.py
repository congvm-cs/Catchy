import cv2
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator    

class DeepFashionDataset():

    def __remove_whitespace__(self, arr):
        result = []
        for item in arr:
            if item != '':
                result.append(item)

        return result


    def __load_anotation_file__(self, anotation_dir):
        file_path = []
        coordinations = []
        with open(anotation_dir, 'r') as file:
            lines = file.readlines()

            for line in lines[2::]:
                txt_arr = line.split(' ')
                result = self.__remove_whitespace__(txt_arr)
                file_path.append(result[0])
                coordinations.append([result[1], result[2], result[3], result[4]])
        
        return file_path, coordinations


    def __aligned_crop__(self, img, coordinations):
        x1 = int(coordinations[0])
        y1 = int(coordinations[1])
        x2 = int(coordinations[2])
        y2 = int(coordinations[3])

        width = np.abs(x2 - x1)
        height = np.abs(y2 - y1)

        if width < height:
            print('__aligned_crop__: {}'.format(1))
            center = (x2 + x1)/2
            print('center :', center)
            offset = height/2
            print('offset :', offset)

            x1 = int(center - offset) if int(center - offset) >= 0 else 0
            x2 = int(center + offset)


            cv2.circle(img, (int(center), y1), 2, (0, 255, 255), 2)

        elif width > height:
            print('__aligned_crop__: {}'.format(2))
            center = (y2 + y1)/2
            offset = width/2
            y1 = int(center - offset) if int(center - offset) >= 0 else 0 
            y2 = int(center + offset)

        print(x1)
        print(y1)
        print(x2)
        print(y2)
        
        return img[y1:y2, x1:x2, :]


    def __save_image_with_labels(self, file_path, img):
        pass


    def load_images_with_notations(self, data_dir, anotation_dir, save_image=False, saved_file_path=None):
        file_path, coordinations = self.__load_anotation_file__(anotation_dir)

        
        for file_name, coord in zip(file_path, coordinations):
            print(coord)
            image_path = os.path.join(data_dir, file_name)
            original_img = cv2.imread(image_path)

            print('image_path: {}'.format(image_path))
            x1 = int(coord[0])
            y1 = int(coord[1])
            x2 = int(coord[2])
            y2 = int(coord[3])
            cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cropped_img  = self.__aligned_crop__(original_img, coord) 

            
            cv2.imshow("original_img", original_img)
            cv2.imshow("cropped_img", cropped_img)
            cv2.waitKey(0)

            if save_image == True:
                assert saved_file_path is None
                self.__save_image_with_labels(saved_file_path, cropped_img)


    def __load_encode_images__(self, image_paths):
        # img_arr = []
        img = cv2.imread(image_paths)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img/255.0
        # img_arr.append(img)
        return img

    # def data_augment(self, folder_dir):
        
    #     datagen = ImageDataGenerator(
    #             rotation_range=20,
    #             width_shift_range=0.1,
    #             height_shift_range=0.1,
    #             # rescale=1./255,
    #             # shear_range=0.2,
    #             zoom_range=0.1,
    #             horizontal_flip=True,
    #             fill_mode='constant')

    #     [image_paths, counter] = self.__get_image_path__(folder_dir)

    #     print(counter)
    #     for image_path in image_paths:
    #         stored_folder = os.path.split(image_path)[0]

    #         origin_I_train = cv2.imread(image_path)
    #         origin_I_train = cv2.cvtColor(origin_I_train, cv2.COLOR_BGR2RGB)
    #         ret = self.__check_size__(origin_I_train, 256)
    #         if ret == True:
    #             i = 0
    #             print('--> {}'.format(image_path))
    #             origin_I_train = np.reshape(origin_I_train, (1, origin_I_train.shape[0], origin_I_train.shape[1], 3)) # this is a Numpy array with shape (1, 3, 150, 150
    #             for _ in datagen.flow(origin_I_train, batch_size=1,
    #                                 save_to_dir=stored_folder, save_prefix='aug_img', save_format='jpg'):
    #                 i += 1
    #                 if i > 2:
    #                     break  # otherwise the generator would loop indefinitely
    #         else:
    #             print('Wrong Size: {}'.format(image_path))

    def __check_size__(self, img, desired_size):
        img = np.array(img)
        if (img.shape[0] == desired_size) and (img.shape[1] == desired_size):
            return True
        else:
            return False


    def __categorical_labels__(self, image_path):
        ''' Rename label as a image rely on category
        '''
        categories = ['Denim', 'Jackets_Vests', 'Pants', 'Shirts_Polos', 'Shorts', 'Suiting', 'Sweaters',
                    'Sweatshirts_Hoodies', 'Tees_Tanks', 'Blouses_Shirts', 'Cardigans', 'Dresses',
                    'Graphic_Tees', 'Jackets_Coats', 'Leggings', 'Rompers_Jumpsuits', 'Skirts'] # 17
        genders = ['MAN', 'WOMAN']
        
        # File name example: 
        # Dataset/In-shop Clothes Retrieval Benchmark/WOMEN/Denim/id_00000055/image_name.jpg

        # image_name = image_path.split('/')[-1:]
        
        categorical_name = image_path.split('/')[-3]
        gender_splitted = image_path.split('/')[-4]

        # print(gender_splitted)
        label = np.zeros(shape=[18, 1], dtype=int) # 1     : gender
                                                    # 2-18  : categorical 

        # print(gender_splitted)
        if gender_splitted == 'MEN':
            label[0] = 0
            print('Hello')
        else:
            label[0] = 1

        for idx, category in enumerate(categories):
            if categorical_name == category:
                label[idx + 1] = 1
                break

        return label


    def load_dataset(self, folder_path, is_load_data_arr=False):
        images = []
        labels = []

        print('Loading ...')
        for subfolder_1_name in os.listdir(folder_path):
            subfolder_1_path = os.path.join(folder_path, subfolder_1_name)

            for subfolder_2_name in os.listdir(subfolder_1_path):
                subfolder_2_path = os.path.join(subfolder_1_path, subfolder_2_name)

                for subfolder_3_name in os.listdir(subfolder_2_path):
                    subfolder_3_path = os.path.join(subfolder_2_path, subfolder_3_name)
                        
                    for file_name in os.listdir(subfolder_3_path):

                        file_path = os.path.join(subfolder_3_path, file_name)
                        # print('>> {}'.format(file_path))
                        label = self.__categorical_labels__(file_path)

                        if is_load_data_arr == False:
                            images.append(file_path)
                        
                        elif is_load_data_arr == True:
                            img_arr = self.__load_encode_images__(file_path)
                            images.append(img_arr)

                        labels.append(label)

        print('#No. images in \'{}\': {} image(s)'.format(folder_path, len(images)))
        return [images, labels]
