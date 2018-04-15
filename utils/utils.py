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


    def load_images_and_labels(self, data_dir, anotation_dir, save_image=False, saved_file_path=None):
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


    def data_augment(self, folder_dir):
        
        datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.1,
                height_shift_range=0.1,
                # rescale=1./255,
                # shear_range=0.2,
                zoom_range=0.1,
                horizontal_flip=True,
                fill_mode='constant')

        [image_paths, counter] = self.__get_image_path__(folder_dir)

        print(counter)
        for image_path in image_paths:
            stored_folder = os.path.split(image_path)[0]

            origin_I_train = cv2.imread(image_path)
            origin_I_train = cv2.cvtColor(origin_I_train, cv2.COLOR_BGR2RGB)
            ret = self.__check_size__(origin_I_train, 256)
            if ret == True:
                i = 0
                print('--> {}'.format(image_path))
                origin_I_train = np.reshape(origin_I_train, (1, origin_I_train.shape[0], origin_I_train.shape[1], 3)) # this is a Numpy array with shape (1, 3, 150, 150
                for _ in datagen.flow(origin_I_train, batch_size=1,
                                    save_to_dir=stored_folder, save_prefix='aug_img', save_format='jpg'):
                    i += 1
                    if i > 2:
                        break  # otherwise the generator would loop indefinitely
            else:
                print('Wrong Size: {}'.format(image_path))

    def __get_image_path__(self, folder_dir):
        image_paths = []
        counter = 0
        subfolder_name_arr = os.listdir(folder_dir)

        for subfolder_name in subfolder_name_arr:
            subfolder_dir_arr = os.path.join(folder_dir, subfolder_name)
            id_subfolder_dir_arr = os.listdir(subfolder_dir_arr)
                
            for id_subfolder_name in id_subfolder_dir_arr:
                id_subfolder_dir = os.path.join(subfolder_dir_arr, id_subfolder_name)
                file_name_arr = os.listdir(id_subfolder_dir)

                for file_name in file_name_arr:
                    file_path = os.path.join(id_subfolder_dir, file_name)
                    # print('--> {}'.format(file_path))
                    image_paths.append(file_path)
                    counter += 1

        return [image_paths, counter]


    def __check_size__(self, img, desired_size):
        img = np.array(img)
        if (img.shape[0] == desired_size) and (img.shape[1] == desired_size):
            return True
        else:
            return False

def main():
    input_data_dir = '/mnt/Data/Dataset/Dataset/In-shop Clothes Retrieval Benchmark/MEN/'
    # txt_anotation_path = '/media/vmc/12D37C49724FE954/Well-Look/Dataset/DeepFashion/Anno/list_bbox.txt'
    catchy = DeepFashionDataset() 
    catchy.data_augment(input_data_dir)


if __name__ == '__main__':
    main()