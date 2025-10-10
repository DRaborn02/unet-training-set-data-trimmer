from PIL import Image
import os
import random

# Input folders
image_dir = "unedited_images"
label_dir = "unedited_labels"

# Output folders
train_img_dir = "membrane/train/image"
train_lbl_dir = "membrane/train/label"
val_img_dir = "membrane/validation/image"
val_lbl_dir = "membrane/validation/label"
test_dir = "membrane/test"

# Create output directories if they don't exist
os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(train_lbl_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(val_lbl_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs("membrane/train/aug", exist_ok=True)
os.makedirs("membrane/validation/aug", exist_ok=True)

# Settings
validation_percentage = 0.2  # 20%
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