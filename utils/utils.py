import cv2
import os
import numpy as np

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
        # file_path = anotation_arr[:][0]
        # coordinations = anotation_arr[::][1::]
        # print(file_path[0])
        # print(coordinations[0])
        # print(coordinations)

        
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

                
def main():
    input_data_dir = '/media/vmc/12D37C49724FE954/Well-Look/Dataset/DeepFashion/'
    txt_anotation_path = '/media/vmc/12D37C49724FE954/Well-Look/Dataset/DeepFashion/Anno/list_bbox.txt'
    catchy = DeepFashionDataset() 
    catchy.load_images_and_labels(input_data_dir, txt_anotation_path)

if __name__ == '__main__':
    main()