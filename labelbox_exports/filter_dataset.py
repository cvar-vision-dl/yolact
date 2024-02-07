import cv2
import numpy as np
import json 
import os
import sys

def load_dataset(path):
    with open(os.path.join(path, 'annotations.json'), 'r') as file:
        dataset_json = json.load(file)

    images = []
    for dr in dataset_json:
        im_name = dr["data_row"]["external_id"]
        im_path = os.path.join(path, im_name) 
        im = cv2.imread(im_path)
        images.append((im, im_name))

    return dataset_json, images


def filter_dataset(dataset, pred):
    annotations, images = dataset

    annotations_to_remove = []
    for dr in annotations:
        if pred(dr):
            annotations_to_remove.append(dr)

    for dr in annotations_to_remove:
        annotations.remove(dr)

    return annotations, images

def is_thermal(dr):
    return "_T_" in dr["data_row"]["external_id"]

if __name__ == "__main__":

    filter_func = is_thermal

    if len(sys.argv) < 3:
        print("Usage: python operate_with_dataset.py <path_to_dataset> <output_file>")
        sys.exit(1)

    dataset_path = sys.argv[1]
    output_file = sys.argv[2]

    dataset = load_dataset(dataset_path)
    filtered_dataset = filter_dataset(dataset, filter_func)
    filtered_annotations, filtered_images = filtered_dataset

    with open(output_file, 'w') as file:
        json.dump(filtered_annotations, file, indent=4)
         