import numpy as np
import time
from tqdm import tqdm
from tqdm import trange


import os
import sys

def main():

    for e in range(10):
        for i in tqdm(range(10),file=sys.stdout):
            time.sleep(0.1)

        print("something")




if __name__ == '__main__':

    main()


    # ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    # saved_models_dir = os.path.join(ROOT_DIR, 'saved_models/test_simple/saved_models_train')
    # model_path = os.path.join(saved_models_dir, 'model_40')
    #
    # print("root: ",ROOT_DIR)
    # print(model_path)

    #main()


