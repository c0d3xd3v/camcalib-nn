import os, sys, time, glob
from os import path

import DataSetGeneration.continuous_dataset_generation as DeepCalibDataset


def clearOutputFolder(output_dir):
    out_360_image_paths = glob.glob(output_dir+"/*.*")
    for imname in out_360_image_paths:
        os.remove(imname)
        print(imname)
    os.makedirs(output_dir, exist_ok = True)
# ##############################################################################


def generate(path_to_360_images, output_dir, samples):
    starttime = time.process_time()

    list_360_image_paths = glob.glob(path_to_360_images)
    for impath in list_360_image_paths:
        DeepCalibDataset.generateSingleImageProjections(impath, output_dir, samples)

    print("elapsed time ", time.process_time() - starttime)
# ##############################################################################


def generateNumImages(path_to_360_images, output_dir, num, samples):
    starttime = time.process_time()

    list_360_image_paths = glob.glob(path_to_360_images)
    for i in range(num):
        impath = list_360_image_paths[i]
        DeepCalibDataset.generateSingleImageProjections(impath, output_dir, samples)

    print("elapsed time ", time.process_time() - starttime)
# ##############################################################################


if __name__ == '__main__':
    num_samples_peer_image = int(sys.argv[1])
    num_images = int(sys.argv[2])
    path_to_360_images = sys.argv[3] + '*.jpg'
    output_dir = sys.argv[4] # "continouse_dataset/"

    if os. path. exists(output_dir):
        if len(os.listdir(output_dir)) == 0:
            if num_images == -1:
                generate(path_to_360_images, output_dir,
                         num_samples_peer_image)
            else:
                generateNumImages(path_to_360_images,
                                  output_dir,
                                  num_images,
                                  num_samples_peer_image)
        else:
            print("use cached data ...")
    else:
        os.mkdir(output_dir)
        if num_images == -1:
            generate(path_to_360_images,
                     output_dir,
                     num_samples_peer_image)
        else:
            generateNumImages(path_to_360_images,
                              output_dir,
                              num_images,
                              num_samples_peer_image)
