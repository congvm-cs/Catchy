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
                coordinations.append([result[3], result[4], result[5], result[6]])

        return file_path, coordinations


    def __aligned_crop__(self, img, coordinations):
        x1 = int(int(coordinations[0])*4.4)
        y1 = int(int(coordinations[1])*4.4)
        x2 = int(int(coordinations[2])*4.3)
        y2 = int(int(coordinations[3])*4.3)

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


    def load_images_with_notations(self, data_dir, anotation_dir, save_image=False, saved_file_path=None, new_size=512, add_padding= False):
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
            if add_padding:
                WHITE = (255, 255, 255)
                original_img = constant=cv2.copyMakeBorder(original_img,0 ,0 ,175, 175,cv2.BORDER_CONSTANT,value=WHITE)
            cropped_img  = self.__aligned_crop__(original_img, coord) 

            if save_image:
                output_name = os.path.join(saved_file_path, file_name)
                subfolder = os.path.split(output_name)[0]
                
                if not os.path.isdir(subfolder):
                    os.makedirs(subfolder)

                if new_size is not None:
                    cropped_img = self.__resize__(cropped_img, new_size)
                cv2.imwrite(output_name, cropped_img)

        return cropped_img


    def categorize_labels(self, file_path):
        # Labels = ['Male, Female, Top, Bottom, Full, [STYLES]']
    #     NUM_LABELS = 18
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

newx = DeepFashionProcessor()
newx.load_images_with_notations(data_dir = '/Users/ngocphu/Documents/Deep_Fashion' , anotation_dir = '/Users/ngocphu/Documents/Deep_Fashion/list_bbox_inshop.txt', saved_file_path = '/Users/ngocphu/Documents/Deep_Fashion/ouput_highres', 
                                save_image= True, add_padding = True)