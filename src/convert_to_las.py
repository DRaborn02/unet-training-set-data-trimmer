import os
import numpy as np
import laspy
import open3d as o3d

def convert_to_las(pointcloud_dir):
    """
    Converts .ply files in pointcloud_dir to .las.
    Skips .las files.
    Raises an error for unsupported formats.
    Returns a list of paths to the generated LAS files.
    """
    las_files_created = []

    for file_name in os.listdir(pointcloud_dir):
        file_path = os.path.join(pointcloud_dir, file_name)
        if not os.path.isfile(file_path):
            continue

        base, ext = os.path.splitext(file_name)
        ext = ext.lower()

        if ext == ".las":
            print(f"Skipping LAS (already converted): {file_name}")
            continue

        elif ext == ".ply":
            las_path = os.path.join(pointcloud_dir, f"{base}.las")

            try:
                # Load directly as point cloud (avoids Open3D warning)
                pcd = o3d.io.read_point_cloud(file_path)
                if not pcd.has_points():
                    print(f"No points found in {file_name}, skipping.")
                    continue

                vertices = np.asarray(pcd.points)
                colors = np.asarray(pcd.colors) if pcd.has_colors() else None

                header = laspy.LasHeader(point_format=3, version="1.2")
                las = laspy.LasData(header)
                las.x, las.y, las.z = vertices[:, 0], vertices[:, 1], vertices[:, 2]

                if colors is not None and len(colors) == len(vertices):
                    scaled_colors = (colors * 65535).astype(np.uint16)
                    las.red, las.green, las.blue = (
                        scaled_colors[:, 0],
                        scaled_colors[:, 1],
                        scaled_colors[:, 2],
                    )

                las.write(las_path)
                print(f"Converted PLY → LAS: {las_path}")
                las_files_created.append(las_path)

            except Exception as e:
                print(f"Error converting {file_name}: {e}")

        else:
            raise ValueError(f"Unsupported file format: {file_name}. Only .ply and .las are supported.")

    print("All conversions complete!")
    return las_files_created
