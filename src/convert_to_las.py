import os
import numpy as np
import laspy
from plyfile import PlyData


def _get_optimal_xz_rotation(x, z):
    """
    Find rotation angle that minimizes the xz bounding box area.
    Uses PCA on x-z coordinates.
    Returns the rotation angle in radians and the minimal area.
    """
    xz = np.column_stack((x, z))
    cov = np.cov(xz.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    
    # eigenvector corresponding to largest eigenvalue
    principal_axis = eigenvectors[:, np.argmax(eigenvalues)]
    
    # angle from x-axis (negate to rotate coordinates TO align with principal axis)
    angle = -np.arctan2(principal_axis[1], principal_axis[0])
    return angle

def _rotate_point(x, y, z, angle):
    """Rotate point around y-axis by angle (radians)."""
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    x_rot = x * cos_a - z * sin_a
    z_rot = x * sin_a + z * cos_a
    return x_rot, y, z_rot


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
                # ----- read ply header manually to avoid memory spikes -----
                def _parse_header(fp):
                    fmt = None
                    vertex_count = None
                    props = []  # list of (name, ply_type)
                    current_elem = None
                    while True:
                        line = fp.readline().decode('ascii')
                        if not line:
                            raise ValueError("Unexpected end of file while reading PLY header")
                        parts = line.strip().split()
                        if not parts:
                            continue
                        if parts[0] == "format":
                            fmt = parts[1]  # e.g. 'binary_little_endian'
                        elif parts[0] == "element":
                            current_elem = parts[1]
                            if current_elem == "vertex":
                                vertex_count = int(parts[2])
                        elif parts[0] == "property" and current_elem == "vertex":
                            # property <type> <name>
                            props.append((parts[2], parts[1]))
                        elif parts[0] == "end_header":
                            break
                    return fmt, vertex_count, props

                def _ply_dtype(props, fmt):
                    # map ply types to numpy dtype codes
                    ply2np = {
                        'char': 'i1', 'uchar': 'u1',
                        'short': 'i2', 'ushort': 'u2',
                        'int': 'i4', 'uint': 'u4',
                        'float': 'f4', 'double': 'f8'
                    }
                    endian = '<' if 'little' in fmt else '>'
                    fields = []
                    for name, ptype in props:
                        if ptype not in ply2np:
                            raise ValueError(f"Unsupported PLY type '{ptype}'")
                        fields.append((name, endian + ply2np[ptype]))
                    return np.dtype(fields)

                # open the PLY file and stream vertex records
                with open(file_path, 'rb') as ply_f:
                    fmt, n_vertices, properties = _parse_header(ply_f)
                    if n_vertices is None or fmt is None:
                        raise ValueError(f"PLY header missing required information: {file_path}")

                    if fmt.startswith('ascii'):
                        # simple ascii reader with rotation
                        def _iter_ascii():
                            for _ in range(n_vertices):
                                line = ply_f.readline().decode('ascii')
                                if not line:
                                    break
                                vals = line.strip().split()
                                yield vals

                        # sample first to compute rotation angle
                        sampled_x = []
                        sampled_z = []
                        sample_every = max(1, n_vertices // 10000)
                        ascii_lines = list(_iter_ascii())
                        
                        for i, vals in enumerate(ascii_lines[::sample_every][:10000]):
                            data = dict(zip([p[0] for p in properties], vals))
                            sampled_x.append(float(data.get('x', 0)))
                            sampled_z.append(float(data.get('z', 0)))
                        
                        sampled_x = np.array(sampled_x)
                        sampled_z = np.array(sampled_z)
                        rotation_angle = _get_optimal_xz_rotation(sampled_x, sampled_z)
                        print(f"Optimal rotation angle for xz plane: {np.degrees(rotation_angle):.2f} degrees")
                        
                        header = laspy.LasHeader(point_format=3, version="1.2")
                        with laspy.open(las_path, mode="w", header=header) as writer:
                            for vals in ascii_lines:
                                # map values by property order
                                data = dict(zip([p[0] for p in properties], vals))
                                x = float(data.get('x', 0))
                                y = float(data.get('y', 0))
                                z = float(data.get('z', 0))
                                
                                x_rot, y_rot, z_rot = _rotate_point(
                                    np.array([x]), np.array([y]), np.array([z]), rotation_angle
                                )
                                
                                points = laspy.ScaleAwarePointRecord.zeros(1, header=header)
                                points.x = x_rot[0]
                                points.y = y_rot[0]
                                points.z = z_rot[0]
                                if {'red','green','blue'}.issubset(data.keys()):
                                    points.red   = int(data['red']) * 257
                                    points.green = int(data['green']) * 257
                                    points.blue  = int(data['blue']) * 257
                                writer.write_points(points)
                    else:
                        dtype = _ply_dtype(properties, fmt)
                        # read in manageable chunks
                        chunk_size = 500_000
                        
                        # ===== First pass: sample coords to find optimal rotation =====
                        sampled_x = []
                        sampled_z = []
                        temp_file_pos = ply_f.tell()
                        sample_every = max(1, n_vertices // 50000)  # sample up to 50k points
                        
                        while True:
                            arr_sample = np.fromfile(ply_f, dtype=dtype, count=100000)
                            if arr_sample.size == 0:
                                break
                            for i in range(0, arr_sample.size, sample_every):
                                if i < arr_sample.size:
                                    sampled_x.append(arr_sample['x'][i])
                                    sampled_z.append(arr_sample['z'][i])
                        
                        sampled_x = np.array(sampled_x)
                        sampled_z = np.array(sampled_z)
                        rotation_angle = _get_optimal_xz_rotation(sampled_x, sampled_z)
                        print(f"Optimal rotation angle for xz plane: {np.degrees(rotation_angle):.2f} degrees")
                        
                        # ===== Second pass: convert with rotation applied =====
                        ply_f.seek(temp_file_pos)
                        header = laspy.LasHeader(point_format=3, version="1.2")
                        with laspy.open(las_path, mode="w", header=header) as writer:
                            remaining = n_vertices
                            while remaining > 0:
                                to_read = min(chunk_size, remaining)
                                arr = np.fromfile(ply_f, dtype=dtype, count=to_read)
                                if arr.size == 0:
                                    break
                                remaining -= arr.size

                                # extract coords and convert to float for rotation
                                x = arr['x'].astype(np.float64)
                                y = arr['y'].astype(np.float64)
                                z = arr['z'].astype(np.float64)
                                
                                # apply rotation
                                x_rot, y_rot, z_rot = _rotate_point(x, y, z, rotation_angle)

                                has_color = {'red','green','blue'}.issubset(arr.dtype.names)
                                if has_color:
                                    colors = np.column_stack((arr['red'], arr['green'], arr['blue'])).astype(np.uint16) * 257
                                else:
                                    colors = None

                                points = laspy.ScaleAwarePointRecord.zeros(arr.size, header=header)
                                points.x = x_rot
                                points.y = y_rot
                                points.z = z_rot
                                if colors is not None:
                                    points.red   = colors[:,0]
                                    points.green = colors[:,1]
                                    points.blue  = colors[:,2]
                                writer.write_points(points)
                # end manual reading
                print(f"Converted PLY → LAS: {las_path}")
                las_files_created.append(las_path)

            except Exception as e:
                print(f"Error converting {file_name}: {e}")

        else:
            raise ValueError(f"Unsupported file format: {file_name}. Only .ply and .las are supported.")

    print("All conversions complete!")
    return las_files_created