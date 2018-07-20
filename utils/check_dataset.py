import os
import sys
from configs.configs import *

def check_dataset():
    current_path = os.path.curdir
    if not os.path.isdir(os.path.join(current_path, COMPRESSED_DATASET_NAME)):
        # Download dataset
        pass
    
    else:
        print('Dataset found')
