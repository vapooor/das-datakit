import pdb
import cv2
import numpy as np

def write_img(group, name, data, img_new_width):
    ori_height, ori_width, channels = None, None, None
    ori_dtype = None
    for d in data:
        if d is not None:
            ori_height, ori_width, channels = d.shape
            ori_dtype = d.dtype
    
    new_data = [d if d is not None else np.zeros((ori_height, ori_width, channels), dtype=ori_dtype) for d in data]

    if img_new_width > 0:
        width = img_new_width
        height = max(1, int(ori_height * img_new_width / ori_width))
        print(f" ====> Using new shape: [{height}, {width}], original: [{ori_height}, {ori_width}]")
    else:
        width, height = ori_width, ori_height

    images_ds = group.create_dataset(
        name,
        shape=(len(new_data), height, width, channels),
        dtype=ori_dtype,
        compression="gzip",
        compression_opts=4
    )
    for i, img in enumerate(new_data):
        if img_new_width > 0:
            img = cv2.resize(img, (width, height))
        images_ds[i] = img

def write_array(group, name, data):
    group.create_dataset(
        name,
        data=data,
        dtype=data.dtype,
        compression="gzip",
        compression_opts=4
    )