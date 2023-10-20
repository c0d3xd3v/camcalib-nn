# export set PYTHONPATH=/home/kai/Development/densnet-pytorch/DeepCalib/dataset/:/home/kai/Development/densnet-pytorch/DeepCalib/network_training/Regression/Single_net/

import os, time, glob

import DataSetGeneration.continuous_dataset_generation as DeepCalibDataset

def clearOutputFolder(output_dir):
    out_360_image_paths = glob.glob(output_dir+"/*.*")
    for imname in out_360_image_paths:
        os.remove(imname)
        print(imname)
    os.makedirs(output_dir, exist_ok = True)

def generate(path_to_360_images, output_dir):
    starttime = time.process_time()

    list_360_image_paths = glob.glob(path_to_360_images)
    for impath in list_360_image_paths:
        DeepCalibDataset.generateSingleImageProjections(impath, output_dir)

    print("elapsed time ", time.process_time() - starttime)

def generateNumImages(path_to_360_images, output_dir, num):
    starttime = time.process_time()

    list_360_image_paths = glob.glob(path_to_360_images)
    for i in range(num):
        impath = list_360_image_paths[i]
        DeepCalibDataset.generateSingleImageProjections(impath, output_dir)

    print("elapsed time ", time.process_time() - starttime)

if __name__ == '__main__':
    path_to_360_images = 'data/*.jpg'
    output_dir = "continouse_dataset/"

    clearOutputFolder(output_dir)
    generateNumImages(path_to_360_images, output_dir, 300)
    #generate(path_to_360_images, output_dir)
