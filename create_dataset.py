'''
Contains functions for data preparation
'''

import re
import os
import cv2
import glob
import yaml
import pickle
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import argparse


def make_labels(path_file):
    """
    Parsing function for .txt files with labeling.
    Returns dict {screenshot_number : classes_with_coordinates + screen_size}

    path_file : str
        Path to .txt labeling file.
    """
    with open(path_file) as f:
        len_f = sum(1 for _ in f)
    print("Number of lines in file:", len_f)
    f = open(path_file, "r")
    screen_size = f.readline().rstrip("\n").split(" ")
    screen_size = (int(screen_size[0]), int(screen_size[1]))
    labels = {}
    i = 0
    k = 0
    while i < len_f:
        s = f.readline()
        if s == "" or s == "\n":
            break
        if len(s.split(" ")) == 2:
            screen_size = (int(s.split(" ")[0]), int(s.split(" ")[1]))
            s = f.readline()
        num = int(re.findall("\d+", s)[0])
        label = []
        for j in range(int(num)):
            str_l = f.readline().rstrip("\n").split(" ")
            pr = ""
            for j in range(len(str_l) - 4):
                pr += str_l[j]
            new_str = [pr]
            for i in range(len(str_l) - 4, len(str_l)):
                new_str.append(int(str_l[i]))
            label.append(new_str)
        _ = f.readline()
        label.append(screen_size)
        labels[k] = label
        i += num
        k += 1
    print("Total_number:", k)
    return labels


def get_bw_mask(labels, classes):
    """
    Recieves list of labels from make_labels function and screen size.

    labels : list
        List of lists of ints which correspond to X, Y, width and height. Last is screen size.
    """

    screen_size = labels[-1]
    mask = np.tile(0, screen_size)
    i = 0
    for label in reversed(labels[0:-1]):
        process, x, y, width, height = label
        if process in classes:
            if i == 0:
                mask[y: (y + height), x: min(x + width, screen_size[1])] = np.tile(
                    0, (height, width)
                )
            else:
                if -2000 < x and -2000 < y:
                    x = max(x, 0)
                    y = max(y, 0)
                    mask[
                    y: min(y + height, screen_size[0]),
                    x: min(x + width, screen_size[1]),
                    ] = np.tile(
                        classes[process],
                        (
                            min(height, screen_size[0] - y),
                            min(width, screen_size[1] - x),
                        ),
                    )
        i += 1
    return mask


def make_save_bw(lab, path_to_save, classes):
    """
    Recieves list of labels for images to save them to specified directory.
    lab : list of str
        List of path to images given by concate function.
    path_to_save : str
        Path to save labeling to.
    """
    path_to_save += 'original_masks/'
    Path(path_to_save).mkdir(parents=True, exist_ok=True)
    paths_to_imgs = list(lab)
    i = 0
    pbar = tqdm(total=len(paths_to_imgs), initial=i)
    Path(path_to_save).mkdir(parents=True, exist_ok=True)
    for x in paths_to_imgs:
        bw_mask = get_bw_mask(lab[x], classes)
        (Image.fromarray(bw_mask.astype("uint8"))).save(
            path_to_save + x.split("/")[-1]
        )
        i += 1
        pbar.update(1)


def concate(paths_to_folder):
    """
    Combines multiple path to folders with both images and .txt files with labeling.
    Then makes singular list of images and singular list of .txt files, which can be inputted in make_save_bw.
    paths_to_folder : list of str
        List of folders with screenshots and .txt files to concantenate.
    """

    total_paths, paths_to_folders = [], []
    read_files = [str(x + "/test.txt") for x in paths_to_folder]
    total_labe = []
    for f in read_files:
        labels = make_labels(f)
        pa = ""
        for x in f.split("/")[0:-1]:
            pa += x + "/"
        labels["path"] = pa
        total_labe.append(labels)
    total_labels = {}
    for x in total_labe:
        for i in range(len(x) - 1):
            total_labels[x["path"] + str(i) + ".png"] = x[i]
    for path in paths_to_folders:
        paths = sorted(
            glob.glob(path + "/*.png"),
            key=lambda x: int(x.split("/")[-1].split(".")[0]),
        )
        for x in paths:
            total_paths.append(x)

    return total_paths, total_labels


def create_biection_dict(paths_to_folder, output_folder):
    biection_dict = {}
    for path in paths_to_folder:
        images_list = glob.glob(str(path + "/*.png"))

        for img_path in images_list:
            biection_dict[img_path] = (
                    output_folder + 'original_masks/' + img_path.split("/")[-1]
            )

    for key in biection_dict.keys():
        if not os.path.isfile(key) or not os.path.isfile(biection_dict[key]):
            print(f"File {key} does not exist")
    return biection_dict


def create_splits(biection, train_perc, val_perc, test_perc):
    keylist = list(biection.keys())
    random.shuffle(keylist)
    total = len(keylist)
    train = keylist[: int(total * train_perc / 100)]
    val = keylist[
          int(total * train_perc / 100): int(total * train_perc / 100)
                                         + int(total * val_perc / 100)
          ]
    test = keylist[int(total * train_perc / 100) + int(total * val_perc / 100):]
    # Create biections
    train_dict = {}
    for path in train:
        train_dict[path] = biection[path]
    val_dict = {}
    for path in val:
        val_dict[path] = biection[path]
    test_dict = {}
    for path in test:
        test_dict[path] = biection[path]

    return train_dict, val_dict, test_dict


def resize_biection(biection, output_path, resize_to=256):
    masks_path = output_path + "resized_masks"
    images_path = output_path + "resized_images"
    Path(masks_path).mkdir(parents=True, exist_ok=True)
    Path(images_path).mkdir(parents=True, exist_ok=True)

    resulting_biection = {}
    for image in tqdm(biection.keys()):
        image_to = images_path + "/" + image.split("/")[-1]
        img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        img_resized = cv2.resize(
            img, (resize_to, resize_to), interpolation=cv2.INTER_AREA
        )
        cv2.imwrite(image_to, img_resized)

        mask_to = masks_path + "/" + biection[image].split("/")[-1]
        msk = cv2.imread(biection[image], cv2.IMREAD_UNCHANGED)
        msk_resized = cv2.resize(
            msk, (resize_to, resize_to), interpolation=cv2.INTER_NEAREST
        )
        cv2.imwrite(mask_to, msk_resized)

        resulting_biection[image_to] = mask_to
    return resulting_biection


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", help="Path to dataset parsing config file in yaml format"
    )
    args = parser.parse_args()

    print(f"Opening config file from {args.config_path}")
    with open(args.config_path, "r") as stream:
        parsed = yaml.safe_load(stream)

    total_paths, total_labels = concate(parsed["input_folders"])
    print(f"Total number of labels: {len(total_labels)}")
    print(f"Total number of paths: {len(total_paths)}")

    make_save_bw(total_labels, parsed["output_folder"], parsed["classes"])

    biection = create_biection_dict(parsed["input_folders"], parsed["output_folder"])
    train, val, test = create_splits(
        biection,
        parsed["train_percentage"],
        parsed["val_percentage"],
        parsed["test_percentage"],
    )

    if parsed["resize"] > 0:
        train = resize_biection(
            train, parsed["output_folder"], resize_to=parsed["resize"]
        )
        val = resize_biection(val, parsed["output_folder"], resize_to=parsed["resize"])
        test = resize_biection(
            test, parsed["output_folder"], resize_to=parsed["resize"]
        )

    Path(parsed["output_folder"] + "/pickles").mkdir(parents=True, exist_ok=True)
    with open(parsed["output_folder"] + "/pickles/train.pickle", "wb") as file:
        pickle.dump(train, file)

    with open(parsed["output_folder"] + "/pickles/val.pickle", "wb") as file:
        pickle.dump(val, file)

    with open(parsed["output_folder"] + "/pickles/test.pickle", "wb") as file:
        pickle.dump(test, file)
