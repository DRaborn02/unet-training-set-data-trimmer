# Unet Training Set Data Trimmer

This utility script automates the preprocessing of image–label pairs for training a U-Net segmentation model. It generates consistent, properly split, and sequentially numbered image tiles for training and validation, while automatically filtering out patches that contain no useful data (such as images with no sidewalk data present). The resulting data is designed to be used for [unetboe2025](https://github.com/yealina/unetboe2025/)

## How to Use

### Dependencies

This program uses the following libraries:
* Pillow
* Numpy

Install dependencies with:
pip install -r requirements.txt

### Usage

* Place your full-resolution raw images in:
unedited_images/
unedited_labels/

* Run the script:
python data_trimmer.py

* The processed, trimmed dataset will appear under:
membrane/train/
membrane/validation/

## Folder Structure

### Input folders:

unedited_images/
unedited_labels/

### Output folders (created automatically):

membrane/
├── train/
│   ├── image/
│   └── label/
├── validation/
│   ├── image/
│   └── label/
└── test/

## Notes

* The script expects grayscale (L-mode) images for both input and labels.

* Blank patches (above a certain white pixel threshold) are automatically removed.

* To include a testing dataset, you can manually copy colored cropped images to membrane/test/. One has been provided in the files for now.

* The current images provided in unedited_images and unedited_labels are from [Sidewalk Concrete Slab Joint Dataset](https://www.yuhanjiang.com/dataset/). Our eventual goal is to train the membrane on our own lidar scans from the rover.

### Future Plans

Our next step would likely be to create a script that creates the unedited images we need from our lidar scans. We would need to manually draw in the labels using some sort of photo editing software such as Krita or Gimp.