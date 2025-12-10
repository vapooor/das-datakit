import pdb
import cv2

def write_img(group, name, data, img_new_width):
    first_image = data[0]
    ori_height, ori_width, channels = first_image.shape
    if img_new_width > 0:
        width = img_new_width
        height = max(1, int(ori_height * img_new_width / ori_width))
        print(f" ====> Using new shape: [{height}, {width}], original: [{ori_height}, {ori_width}]")
    else:
        width, height = ori_width, ori_height

    images_ds = group.create_dataset(
        name,
        shape=(len(data), height, width, channels),
        dtype=first_image.dtype,
        compression="gzip",
        compression_opts=4
    )
    for i, img in enumerate(data):
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