# Config Model
IMAGE_SIZE = 128
IMAGE_DEPTH = 3
IMAGE_SHAPE = [IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH]
BINARY_GENDER_OUTPUTS = 1
CATEGORICAL_STYLE_OUTPUTS = 17
CATEGORICAL_POSITION_OUTPUTS = 3
NUM_LABELS = 21
COMPRESSED_DATASET_NAME = 'deepfashion_output.tar.gz'
OPTIMIZER = 'Adam'
LOSS = []
# STYLES = ['Denim', 'Jackets_Vests', 'Pants', 'Shirts_Polos', 'Shorts', 
#         'Suiting', 'Sweaters', 'Sweatshirts_Hoodies', 'Tees_Tanks', 
#         'Blouses_Shirts', 'Cardigans', 'Dresses','Graphic_Tees',
#         'Jackets_Coats', 'Leggings', 'Rompers_Jumpsuits','Skirts']

STYLES = ['Denim', 'Jackets_Vests', 'Pants', 'Shirts_Polos', 'Shorts', 
        'Suiting', 'Sweaters', 'Sweatshirts_Hoodies', 'Tees_Tanks', 
        'Blouses_Shirts', 'Cardigans', 'Dresses','Graphic_Tees',
        'Jackets_Coats', 'Leggings', 'Rompers_Jumpsuits','Skirts']


TRAIN_PARAMS = {'dim': (IMAGE_SIZE, IMAGE_SIZE),
          'batch_size': 100,
          'n_classes': NUM_LABELS,
          'n_channels': IMAGE_DEPTH,
          'shuffle': True,
          'training': False}


VAL_PARAMS = {'dim': (IMAGE_SIZE, IMAGE_SIZE),
            'batch_size': 100,
            'n_classes': NUM_LABELS,
            'n_channels': IMAGE_DEPTH,
            'shuffle': True,
            'training': False}