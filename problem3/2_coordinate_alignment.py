import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from scipy.linalg import svd

def umeyama(X, Y):
    """
    computes the least squares similarity transformation between two point clouds.
    """

    # find centroids
    centroid_X = X.mean(axis=0)
    centroid_Y = Y.mean(axis=0)

    # center the point clouds
    X_centered = X - centroid_X
    Y_centered = Y - centroid_Y

    # compute the covariance matrix H
    H = X_centered.T @ Y_centered

    # perform SVD on H
    U, S, Vt = svd(H)

    # compute the rotation matrix R
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R_align = Vt.T @ D @ U.T

    # compute the scale factor s
    s_align = (np.sum(S * np.diag(D)) / np.sum(X_centered**2))
    
    # compute the translation vector t
    t_align = centroid_Y - s_align * R_align @ centroid_X
    
    return R_align, t_align.reshape(3, 1), s_align

def read_colmap_poses(filepath):
    """
    reads COLMAP poses from the images.txt file.
    returns a dictionary mapping filenames to pose matrices (R, t).
    """
    poses = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith('#'):
            i += 1
            continue

        parts = line.split()
        if len(parts) < 10:
            i += 1
            continue

        image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name = parts
        
        # convert quaternion to rotation matrix (COLMAP: w,x,y,z while SciPy: x,y,z,w)
        quat = np.array([float(qx), float(qy), float(qz), float(qw)])
        R_colmap = R.from_quat(quat).as_matrix()

        t_colmap = np.array([float(tx), float(ty), float(tz)]).reshape(3, 1)
        
        poses[name] = {'R': R_colmap, 't': t_colmap}
        
        i += 2
            
    return poses

def read_chessboard_poses(filepath):
    """
    reads poses from chessboard poses file.
    """
    poses = {}
    with open(filepath, 'r') as f:
        content = f.read()

    sections = content.split('\n\n')

    for section in sections:
        if not section.strip():
            continue

        lines = section.strip().split('\n')

        if lines[0].startswith("# Filename:"):
            fname = os.path.basename(lines[0].split(": ")[1].strip()).lower()
            
            # parse R matrix
            R_lines = [l.strip() for l in lines[2:5]]
            R_data = []
            for r_line in R_lines:
                r_line_clean = r_line.replace('[', '').replace(']', '').replace(',', '')
                R_data.extend(list(map(float, r_line_clean.split())))
            R_matrix = np.array(R_data).reshape(3, 3)

            # parse t vector
            t_data_str = ""
            found_t = False
            for line in lines[5:]:
                if line.startswith("t:"):
                    found_t = True
                elif found_t:
                    t_data_str += line.strip() + " "
            
            t_line_clean = t_data_str.replace('[', '').replace(']', '').replace(',', '')
            t_data = list(map(float, t_line_clean.split()))
            t_vector = np.array(t_data).reshape(3, 1)

            poses[fname] = {'R': R_matrix, 't': t_vector}
    
    return poses

def apply_coordinate_transform(R_colmap, t_colmap):
    """
    applies a transform to COLMAP poses to align with the chessboard system (used for trial and error).
    currently rotating 180 degrees around X.
    """
    transform_matrix = np.array([
    [1,  0,  0], 
    [ 0,  -1,  0],  
    [ 0,  0, -1] 
    ])
    
    R_transformed = transform_matrix @ R_colmap
    t_transformed = transform_matrix @ t_colmap
    
    return R_transformed, t_transformed

def get_corresponding_camera_centers(poses_ref, poses_target):
    """
    finds corresponding camera centers from two pose dictionaries.
    returns two lists of centers (3D points).
    """
    centers_ref = []
    centers_target = []
    
    common_fnames = sorted(list(set(poses_ref.keys()) & set(poses_target.keys())))

    for fname in common_fnames:
        R_ref = poses_ref[fname]['R']
        t_ref = poses_ref[fname]['t']
        
        R_target = poses_target[fname]['R']
        t_target = poses_target[fname]['t']

        # camera center
        C_ref = -R_ref.T @ t_ref
        C_target = -R_target.T @ t_target

        centers_ref.append(C_ref)
        centers_target.append(C_target)

    return np.array(centers_ref).squeeze(), np.array(centers_target).squeeze()

def calculate_errors(poses_ref, poses_target, R_align, t_align, s_align):
    """
    calculates and prints the alignment errors.
    """
    rotational_errors = []
    for fname, pose_data_colmap in poses_target.items():
        if fname in poses_ref:
            R_colmap = pose_data_colmap['R']
            R_chessboard = poses_ref[fname]['R']
            
            R_aligned = R_align @ R_colmap
            R_diff = R_aligned @ R_chessboard.T
            rotational_errors.append(R.from_matrix(R_diff).magnitude() * 180 / np.pi)

    rot_error = np.mean(rotational_errors) if rotational_errors else 0

    colmap_centers, chessboard_centers = get_corresponding_camera_centers(poses_ref, poses_target)    
    transformed_colmap_centers = (s_align * R_align @ colmap_centers.T + t_align).T
    
    trans_errors = [np.linalg.norm(chessboard_centers[i] - transformed_colmap_centers[i])
                    for i in range(len(transformed_colmap_centers))]
    
    trans_error_mean = np.mean(trans_errors)

    print("\n- Alignment Errors -")
    print(f"Rotational Error: {rot_error:.2f} degrees")
    print(f"Translational Error: {trans_error_mean:.6f} meters")

    return rot_error, trans_error_mean

def write_aligned_poses(filepath, aligned_poses):
    """
    writes the aligned poses to a text file.
    """
    with open(filepath, 'w') as f:
        for fname, pose_data in aligned_poses.items():
            f.write(f"# Filename: {fname}\n")
            f.write(f"R:\n{pose_data['R']}\n")
            f.write(f"t:\n{pose_data['t']}\n\n")

def main():
    print("Starting pose alignment...")

    chessboard_poses_file = 'chessboard_poses.txt'
    colmap_images_file = 'sparse/0_txt/images.txt'

    chessboard_poses = read_chessboard_poses(chessboard_poses_file)
    colmap_poses = read_colmap_poses(colmap_images_file)

    transformed_colmap_poses = {}
    for fname, pose_data in colmap_poses.items():
        R_src, t_src = pose_data['R'], pose_data['t']
        # R_transformed, t_transformed = apply_coordinate_transform(R_src, t_src) # comment out if don't want manual transform
        transformed_colmap_poses[fname] = {'R': R_src, 't': t_src}

    colmap_centers, chessboard_centers = get_corresponding_camera_centers(transformed_colmap_poses, chessboard_poses)


    print(f"Found {colmap_centers.shape[0]} corresponding poses for alignment.")

    R_align, t_align, s_align = umeyama(colmap_centers, chessboard_centers)

    # s_align = 0.7 # manual scale alignment, comment out

    print("\nCalculated alignment transformation:")
    print("Rotation (R):\n", R_align)
    print("Translation (t):\n", t_align)
    print(f"Final Scale (s): {s_align:.6f}")

    aligned_poses = {}
    for fname, pose_data in transformed_colmap_poses.items():
        R_colmap, t_colmap = pose_data['R'], pose_data['t']
        R_aligned = R_align @ R_colmap
        t_aligned = s_align * (R_align @ t_colmap) + t_align
        aligned_poses[fname] = {'R': R_aligned, 't': t_aligned}

    calculate_errors(chessboard_poses, aligned_poses, R_align, t_align, s_align)
    write_aligned_poses('aligned_poses.txt', aligned_poses)

if __name__ == "__main__":
    main()