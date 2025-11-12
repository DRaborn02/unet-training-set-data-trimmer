import laspy
import numpy as np
import os

def split_las(input_path, output_dir, tile_width=5.0):
    las = laspy.read(input_path)
    x, y, z = las.x, las.y, las.z

    min_x, max_x = np.min(x), np.max(x)
    total_width = max_x - min_x
    num_tiles = int(np.ceil(total_width / tile_width))

    print(f"Splitting {os.path.basename(input_path)}...")
    print(f"Total width: {total_width:.2f}, creating {num_tiles} tiles")

    output_paths = []  # store all new split file paths

    for i in range(num_tiles):
        start_x = min_x + i * tile_width
        end_x = start_x + tile_width
        mask = (x >= start_x) & (x < end_x)
        count = np.sum(mask)
        print(f"Tile {i+1}: {count} points, range {start_x:.2f}-{end_x:.2f}")
        if count == 0:
            continue

        # Make new header & create new LAS file with empty points
        sub_header = las.header.copy()
        sub_las = laspy.LasData(sub_header)

        # Create a new point record with correct count
        sub_las.points = laspy.ScaleAwarePointRecord.zeros(count, point_format=las.header.point_format, scales=las.header.scales, offsets=las.header.offsets)


        # Copy each available attribute
        for dim in las.point_format.dimensions:
            setattr(sub_las, dim.name, getattr(las, dim.name)[mask])


        # Preserve coordinate scales and offsets
        sub_las.header.offsets = las.header.offsets
        sub_las.header.scales = las.header.scales

        # Write file
        out_path = os.path.join(
            output_dir,
            f"{os.path.splitext(os.path.basename(input_path))[0]}_part_{i+1}.las"
        )
        sub_las.write(out_path)
        output_paths.append(out_path)
        print(f"  Saved: {out_path}")

    print(f"Finished splitting {os.path.basename(input_path)}")
    return output_paths
    



def split_all_in_folder(input_folder, output_folder, tile_width=5.0):
    """
    Splits all LAS files in a folder.
    """
    os.makedirs(output_folder, exist_ok=True)
    for file in os.listdir(input_folder):
        if file.lower().endswith(".las"):
            input_path = os.path.join(input_folder, file)
            split_las(input_path, output_folder, tile_width)

if __name__ == "__main__":
    input_folder = "pointclouds/" #change to point to folder with LAS files
    output_folder = os.path.join(input_folder, "split")
    split_all_in_folder(input_folder, output_folder, tile_width=5.0)
