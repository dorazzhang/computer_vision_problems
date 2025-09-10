import numpy as np
import os
import shutil
from scipy.spatial.transform import Rotation as R

def read_aligned_poses(filepath):
    """
    Reads aligned poses from the generated text file.
    Returns a dictionary mapping filenames to pose matrices (R, t).
    """
    poses = {}
        
    with open(filepath, 'r') as f:
        content = f.read()
    
    sections = content.split('\n\n')
    
    for section in sections:
        if not section.strip():
            continue
        
        lines = section.strip().split('\n')
        
        # parse filename
        if lines[0].startswith("# Filename:"):
            fname = lines[0].split(": ")[1].strip()
            
            # parse R matrix
            R_lines = [l.strip() for l in lines[2:5]]
            R_data = []
            for r_line in R_lines:
                r_line_clean = r_line.replace('[', '').replace(']', '').replace(',', '')
                R_data.extend(list(map(float, r_line_clean.split())))
            R_matrix = np.array(R_data).reshape(3, 3)

            # parse t vector
            t_data = []
            for i in range(3):
                t_line = lines[6 + i].strip()
                t_line_clean = t_line.replace('[', '').replace(']', '')
                t_data.append(float(t_line_clean))
            t_vector = np.array(t_data).reshape(3, 1)

            # create a 4x4 pose matrix (R | t)
            pose_matrix = np.eye(4)
            pose_matrix[:3, :3] = R_matrix
            pose_matrix[:3, 3] = t_vector.T
            
            poses[fname] = pose_matrix

    return poses

def load_colmap_images(images_file):
    """Load original COLMAP images data."""
    images_data = {}
        
    with open(images_file, 'r') as f:
        lines = f.readlines()

    for i in range(4, len(lines), 2):
        if i < len(lines):
            parts = lines[i].strip().split()
            if len(parts) >= 10:
                image_id = parts[0]
                qw, qx, qy, qz = map(float, parts[1:5])
                tx, ty, tz = map(float, parts[5:8])
                camera_id = parts[8]
                image_name = parts[9]
                
                images_data[image_name] = {
                    'id': image_id,
                    'qw': qw, 'qx': qx, 'qy': qy, 'qz': qz,
                    'tx': tx, 'ty': ty, 'tz': tz,
                    'camera_id': camera_id
                }
    
    return images_data

def pose_matrix_to_colmap_format(pose_matrix):
    """Convert 4x4 pose matrix to COLMAP quaternion format."""
    # extract rotation matrix and translation
    R_matrix = pose_matrix[:3, :3]
    t_vector = pose_matrix[:3, 3]
    
    # convert rotation matrix to quaternion
    rotation = R.from_matrix(R_matrix)
    qx, qy, qz, qw = rotation.as_quat()  # scipy returns [x, y, z, w]
    
    return qw, qx, qy, qz, t_vector[0], t_vector[1], t_vector[2]

def write_colmap_images(aligned_poses, original_images_data, output_file):
    """Write aligned poses in COLMAP images.txt format."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(aligned_poses)}\n")
        
        for image_name, pose_matrix in aligned_poses.items():
            if image_name in original_images_data:
                img_data = original_images_data[image_name]
                
                qw, qx, qy, qz, tx, ty, tz = pose_matrix_to_colmap_format(pose_matrix)
                
                f.write(f"{img_data['id']} {qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {tx:.6f} {ty:.6f} {tz:.6f} {img_data['camera_id']} {image_name}\n")
                f.write("\n")

def create_rigs_file(output_path):
    """Create rigs.txt file - defines the camera rig configuration"""
    rigs_content = """# Rig calib list with one line of data per calib:
    #   RIG_ID, NUM_SENSORS, REF_SENSOR_TYPE, REF_SENSOR_ID, SENSORS[] as (SENSOR_TYPE, SENSOR_ID, HAS_POSE, [QW, QX, QY, QZ, TX, TY, TZ])
    # Number of rigs: 1
    1 1 CAMERA 1
    """
    with open(output_path, 'w') as f:
        f.write(rigs_content)

def create_frames_file(output_path, aligned_poses):
    """Create frames.txt file - defines frame poses in rig coordinate system"""
    frames_content = """# Frame list with one line of data per frame:
    #   FRAME_ID, RIG_ID, RIG_FROM_WORLD[QW, QX, QY, QZ, TX, TY, TZ], NUM_DATA_IDS, DATA_IDS[] as (SENSOR_TYPE, SENSOR_ID, DATA_ID)
    # Number of frames: {}\n""".format(len(aligned_poses))
    
    frame_id = 1
    for image_name, pose_matrix in aligned_poses.items():
        pose_array = np.array(pose_matrix)
        R_matrix = pose_array[:3, :3]
        t_vector = pose_array[:3, 3]
        
        rotation = R.from_matrix(R_matrix)
        quat = rotation.as_quat()
        
        frames_content += "{} 1 {} {} {} {} {} {} {} 1 CAMERA 1 {}\n".format(
            frame_id,
            quat[3], quat[0], quat[1], quat[2],
            t_vector[0], t_vector[1], t_vector[2],
            frame_id
        )
        frame_id += 1
    
    with open(output_path, 'w') as f:
        f.write(frames_content)

def copy_colmap_files(sparse_dir, output_dir):
    # copy other necessary COLMAP files.
    files_to_copy = ['cameras.txt', 'points3D.txt']
    
    for filename in files_to_copy:
        src_file = os.path.join(sparse_dir, filename)
        dst_file = os.path.join(output_dir, filename)
        shutil.copy2(src_file, dst_file)

def main():
    aligned_poses_file = "aligned_poses.txt"
    colmap_images_file = "sparse/0_txt/images.txt"
    output_dir = "colmap_aligned"
    sparse_dir = "sparse/0_txt"
    
    print("Generating COLMAP files...")

    aligned_poses = read_aligned_poses(aligned_poses_file)
    if aligned_poses is None:
        return

    original_images_data = load_colmap_images(colmap_images_file)
    if original_images_data is None:
        return

    os.makedirs(output_dir, exist_ok=True)

    output_images_file = os.path.join(output_dir, "images.txt")
    write_colmap_images(aligned_poses, original_images_data, output_images_file)

    copy_colmap_files(sparse_dir, output_dir)

    rigs_path = os.path.join(output_dir, "rigs.txt")
    create_rigs_file(rigs_path)

    frames_path = os.path.join(output_dir, "frames.txt")
    create_frames_file(frames_path, aligned_poses)
    
    print("\nComplete COLMAP Reconstruction Generated. Run [colmap gui]")

if __name__ == "__main__":
    main()
