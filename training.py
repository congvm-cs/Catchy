import utils
from model.catchynet import CatchyNet
from utils.get_args import get_args
from utils.check_dataset import check_dataset
from utils.datagenerator import DataGenerator
from configs.configs import *


def main():
    check_dataset()
    args = get_args()
    
    # Load train_path
    # Load test_path
    train_paths = []
    test_paths = []

    training_generator = DataGenerator(train_paths, **TRAIN_PARAMS)
    validation_generator = DataGenerator(test_paths, **VAL_PARAMS)

    # Load model
    model = CatchyNet().build(True)
    model.compile()
    
if __name__ == '__main__':
    main()