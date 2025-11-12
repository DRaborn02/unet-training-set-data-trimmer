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

* Place your .las or .ply files in:

pointclouds/raw

* Modify any settings you'd like for the resulting training membrain at the top of main.py, particularly:

validation_percentage = 0.2  # 20% of patches for validation

sidewalkThreshold = 0.99  # Threshold for skipping mostly white patches


* Ensure you're in /src and run the script:

python main.py

* The pointcloud files will then be turned into .las files (if they were .ply), divide the .las files into smaller sections, and turn those sections into orthoimages.

* You will then be prompted to label the cracks in each orthoimage. The images/labels will end up in the /orthoImages folder

* Finally, the image/label pairs will be processed into a trimmed dataset under:

membrane/train/

membrane/validation/

* You can then use the same settings in main.py and the resulting membrane folder to train a new unet model using [unetboe2025](https://github.com/yealina/unetboe2025/).

## Folder Structure

### Input folders:

orthoImages/

├── images/ # premade label/image pairs go here

├── labels/ # premade label/image pairs go here

pointclouds/

├── raw/ # unlabeled .las/.ply files go here

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

### Future Plans

