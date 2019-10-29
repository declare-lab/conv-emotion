import os

DATASET_DIRECTORY_PATH = "./datasets/"
GENERATIVE_WEIGHTS_DIRECTORY_PATH = "./generative_weights/"

if not os.path.exists(DATASET_DIRECTORY_PATH):
    os.makedirs(DATASET_DIRECTORY_PATH)


if not os.path.exists(GENERATIVE_WEIGHTS_DIRECTORY_PATH):
    os.makedirs(GENERATIVE_WEIGHTS_DIRECTORY_PATH)