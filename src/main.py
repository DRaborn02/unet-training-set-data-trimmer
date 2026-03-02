from PIL import Image
import os
import random
import numpy
import shutil
from pointcloud2orthoimage import p2o_main
from crack_labeler import label_all_images
from convert_to_las import convert_to_las
from split_las import split_las

# Input folders
unlabeled_image_dir = "../orthoImages/unlabeled_images/"
image_dir = "../orthoImages/images/"
label_dir = "../orthoImages/labels/"

raw_dir = "../pointclouds/raw/"
split_dir = "../pointclouds/split/"
processed_dir = "../pointclouds/processed/"

# Output folders
train_img_dir = "../membrane/train/image"
train_lbl_dir = "../membrane/train/label"
val_img_dir = "../membrane/validation/image"
val_lbl_dir = "../membrane/validation/label"
test_dir = "../membrane/test"

# Create output directories if they don't exist
os.makedirs(unlabeled_image_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)

os.makedirs(raw_dir, exist_ok=True)
os.makedirs(split_dir, exist_ok=True)
os.makedirs(processed_dir, exist_ok=True)

os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(train_lbl_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(val_lbl_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Settings
validation_percentage = 0.2  # 20% of patches for validation
sidewalkThreshold = 0.99  # Threshold for skipping mostly white patches

TARGET_HEIGHT = 1024   # Resize target height before patching
PATCH_SIZE = 512       # Size of each cropped patch
STRIDE = 256           # Horizontal stride for sliding window
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

random.seed(42)  # For reproducibility

def convert_test_to_bw(test_dir):
    """
    Converts all images in the test folder to black & white (grayscale).
    Overwrites them in place.
    """
    for fname in os.listdir(test_dir):
        fpath = os.path.join(test_dir, fname)
        if not os.path.isfile(fpath):
            continue  # skip subfolders
        try:
            with Image.open(fpath) as im:
                im = im.convert("L")  # grayscale
                im.save(fpath)        # overwrite original
        except Exception as e:
            print(f"Skipping {fpath}: {e}")

def process_pair(image_path, label_path):
    """Returns a list of (image_patch, label_patch) pairs in memory."""
    img = Image.open(image_path).convert("L")
    lbl = Image.open(label_path).convert("L")

    # Resize so height = 512 while keeping aspect ratio
    scale = TARGET_HEIGHT / img.height
    new_width = int(img.width * scale)
    img = img.resize((new_width, TARGET_HEIGHT), Image.BILINEAR)
    lbl = lbl.resize((new_width, TARGET_HEIGHT), Image.NEAREST)

    patches = []
    for x in range(0, new_width - PATCH_SIZE + 1, STRIDE):
        for y in range(0, TARGET_HEIGHT - PATCH_SIZE + 1, STRIDE):
            img_patch = img.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE))
            lbl_patch = lbl.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE))
            
            # --- Skip blank (mostly white) patches ---
            arr = numpy.array(lbl_patch)
            white_ratio = numpy.mean(arr > 250)  # % of white pixels
            if white_ratio > sidewalkThreshold:             # adjustable threshold
                continue

            patches.append((img_patch, lbl_patch))
    return patches

def auto_find_pairs():
    """Automatically matches image and label files by name."""
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(IMAGE_EXTS)]
    label_files = [f for f in os.listdir(label_dir) if f.lower().endswith(IMAGE_EXTS)]

    pairs = []
    for img_name in image_files:
        # Match label name (same base name or base + '_label')
        base = os.path.splitext(img_name)[0]
        possible_labels = [base + "_label", base]
        match = next(
            (lbl for lbl in label_files if os.path.splitext(lbl)[0] in possible_labels),
            None
        )
        if match:
            pairs.append((os.path.join(image_dir, img_name),
                          os.path.join(label_dir, match)))
        else:
            print(f"No label found for {img_name}")
    return pairs

def clear_directory(dir_path):
    """Deletes all files in a directory (but keeps the folder)."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        return
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # remove file or symlink
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # remove subfolders (rare)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

def createDataSet():
    pairs = auto_find_pairs()
    if not pairs:
        print("No matching image/label pairs found. Check your folder names and extensions.")
        exit(1)

    print(f"Found {len(pairs)} image/label pairs. Generating patches...")

    # Collect all patches in memory
    all_patches = []
    for img_path, lbl_path in pairs:
        all_patches.extend(process_pair(img_path, lbl_path))

    total_patches = len(all_patches)

    # Randomly shuffle and split
    random.shuffle(all_patches)
    val_count = int(total_patches * validation_percentage)
    val_patches = all_patches[:val_count]
    train_patches = all_patches[val_count:]

    # Save training patches sequentially
    for i, (img_patch, lbl_patch) in enumerate(train_patches):
        img_patch.save(os.path.join(train_img_dir, f"{i}.png"))
        lbl_patch.save(os.path.join(train_lbl_dir, f"{i}.png"))

    # Save validation patches sequentially
    for i, (img_patch, lbl_patch) in enumerate(val_patches):
        img_patch.save(os.path.join(val_img_dir, f"{i}.png"))
        lbl_patch.save(os.path.join(val_lbl_dir, f"{i}.png"))

    print("\nDataset creation complete!")
    print(f"Total patches created: {total_patches}")
    print(f"Validation percentage: {validation_percentage}")
    print(f"{len(train_patches)} training patches")
    print(f"{len(val_patches)} validation patches")

    if os.path.isdir(test_dir):
        convert_test_to_bw(test_dir)
        print("Converted test images to grayscale.")


if __name__ == "__main__":
    
    # Step 1: Detect and convert new pointclouds in 'raw'
    raw_files = [f for f in os.listdir(raw_dir)
                if os.path.isfile(os.path.join(raw_dir, f)) and f.lower().endswith((".ply", ".las"))]

    if raw_files:
        print(f"Found {len(raw_files)} new pointcloud(s) in raw. Beginning conversion...")
        las_files_created = convert_to_las(raw_dir)

        # Move processed PLY files to 'processed'
        for f in raw_files:
            if f.lower().endswith(".ply"):
                src = os.path.join(raw_dir, f)
                dst = os.path.join(processed_dir, f)
                try:
                    os.rename(src, dst)
                    print(f"Moved processed PLY -> {dst}")
                except Exception as e:
                    print(f"Failed to move {f}: {e}")
    else:
        print("No new pointclouds found in raw.")


    # # Step 2: Split LAS files in 'raw' -> 'split' (then move processed to 'processed')
    # las_files_raw = [f for f in os.listdir(raw_dir) if f.lower().endswith(".las")]
    # if las_files_raw:
    #     print(f"Splitting {len(las_files_raw)} LAS file(s) from raw...")
    #     for las_file in las_files_raw:
    #         las_path = os.path.join(raw_dir, las_file)
    #         try:
    #             split_output_paths = split_las(las_path, split_dir)
    #             print(f"Split {las_file} -> {len(split_output_paths)} files.")
                
    #             # Move original LAS to 'processed'
    #             dest_path = os.path.join(processed_dir, las_file)
    #             os.replace(las_path, dest_path)
    #         except Exception as e:
    #             print(f"Error splitting {las_file}: {e}")
    # else:
    #     print("No new LAS files to split in raw.")

    # # Step 3: Convert all new LAS files in 'split' to orthoimages
    # las_files_split = [f for f in os.listdir(split_dir) if f.lower().endswith(".las")]
    # if las_files_split:
    #     print(f"Generating orthoimages for {len(las_files_split)} LAS file(s)...")
    #     p2o_main(split_dir, unlabeled_image_dir)
    #     # Move processed LAS to 'processed'
    #     for las_file in las_files_split:
    #         os.rename(os.path.join(split_dir, las_file),
    #                   os.path.join(processed_dir, las_file))
    # else:
    #     print("No new LAS files in split to process into orthoimages.")

    # # Step 4: Label unlabeled orthoimages
    # unlabeled_imgs = [f for f in os.listdir(unlabeled_image_dir)
    #                   if f.lower().endswith(IMAGE_EXTS)]
    # if unlabeled_imgs:
    #     print(f"Labeling {len(unlabeled_imgs)} unlabeled image(s)...")
    #     label_all_images(unlabeled_image_dir, image_dir, label_dir)
    # else:
    #     print("No unlabeled images found to label.")

    # print("\n--- Preprocessing pipeline complete ---")
    
    # #Step 5: Delete previous dataset and create new dataset from labeled images
    # print("Clearing old training and validation data...")
    # clear_directory(train_img_dir)
    # clear_directory(train_lbl_dir)
    # clear_directory(val_img_dir)
    # clear_directory(val_lbl_dir)
    # print("Dataset directories cleared.")

    # print("Creating new dataset from labeled images...")
    # createDataSet()