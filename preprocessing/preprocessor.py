import cv2
import numpy as np
import os

# Preprocessing Images
class DeepFashionProcessor():
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
            # print('__aligned_crop__: {}'.format(1))
            center = (x2 + x1)/2
            # print('center :', center)
            offset = height/2
            # print('offset :', offset)

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

    def __resize__(self, image, dsize):
        image = cv2.resize(image, (dsize, dsize))
        return image


    def load_images_with_notations(self, data_dir, anotation_dir, save_image=False, saved_file_path=None, new_size=None):
        ''' Load images and notations from hard disk

            Parameter(s):
                data_dir        [str]       data directory      
                anotation_dir   [str]       bbox anotation directory
                save_image      [bool]      save image in harddisk?
                saved_file_path [str]       saved file path
                new_size        [int]       new size for resizing

            Return(s):
                image array
        '''
        file_path, coordinations = self.__load_anotation_file__(anotation_dir)

        for file_name, coord in zip(file_path, coordinations):
            print(coord)
            image_path = os.path.join(data_dir, file_name)
            original_img = cv2.imread(image_path)

            print('image_path: {}'.format(image_path))
            # x1 = int(coord[0])
            # y1 = int(coord[1])
            # x2 = int(coord[2])
            # y2 = int(coord[3])
    #           cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cropped_img  = self.__aligned_crop__(original_img, coord) 

            if save_image:
                output_name = os.path.join(saved_file_path, file_name)
                subfolder = os.path.split(output_name)[0]
                
                if not os.path.isdir(subfolder):
                    os.makedirs(subfolder)

                if new_size is not None:
                    cropped_img = self.__resize__(cropped_img, 128)

                cv2.imwrite(output_name, cropped_img)

        return cropped_img