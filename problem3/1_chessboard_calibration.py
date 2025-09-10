import numpy as np 
import cv2 as cv 
import glob 
import os 

# pre-calibrated camera parameters from cameras.txt 
fx = 1645.6731103571551 
fy = 1647.0945393118011 
cx = 531 
cy = 944 
k1 = 0.11861387954882849 
k2 = -0.32177210622798663 
p1 = 0.00043702188677298144 
p2 = -0.00050583433027418662 

mtx = np.array([[fx, 0, cx], 
                [0, fy, cy], 
                [0, 0, 1]], dtype=np.float32) 

dist = np.array([k1, k2, p1, p2, 0], dtype=np.float32) 

checkerboard_rows = 10 
checkerboard_cols = 7 
square_size_meters = 0.01 

objp = np.zeros((checkerboard_rows * checkerboard_cols, 3), np.float32) 
objp[:, :2] = np.mgrid[0:checkerboard_cols, 0:checkerboard_rows].T.reshape(-1, 2) 
objp *= square_size_meters 

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001) 

output_file = 'chessboard_poses.txt' 

def save_pose(f, fname, rvec, tvec): 
    """ 
    save the pose to the output file without inversion.
    """ 
    # convert rotation vector to rotation matrix 
    R_mat, _ = cv.Rodrigues(rvec) 
      
    # write the filename, rotation matrix, and translation vector to the file 
    f.write(f"# Filename: {os.path.basename(fname)}\n") 
    f.write(f"R:\n{R_mat}\n") 
    f.write(f"t:\n{tvec.T}\n\n") 

def process_image(f, fname): 
    img = cv.imread(fname) 
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 

    # find chessboard 
    found, corners = cv.findChessboardCorners( 
        gray, (checkerboard_cols, checkerboard_rows), None 
    ) 

    if not found: 
        print(f"Chessboard not found: {fname}") 
        # show the original image 
        cv.imshow("chessboard", img) 
        cv.waitKey(1) 
        return 

    # refine corners 
    corners2 = cv.cornerSubPix( 
        gray, corners, (11, 11), (-1, -1), criteria 
    ) 

    # pose estimation 
    ok, rvec, tvec = cv.solvePnP(objp, corners2, mtx, dist) 
    if not ok: 
        print(f"Pose estimation failed: {fname}") 
        cv.drawChessboardCorners( 
            img, (checkerboard_cols, checkerboard_rows), corners2, found 
        ) 
        cv.imshow("chessboard", img) 
        cv.waitKey(1) 
        return 

    # save pose 
    save_pose(f, fname, rvec, tvec) 

    # draw corners for visualization 
    cv.drawChessboardCorners( 
        img, (checkerboard_cols, checkerboard_rows), corners2, found 
    ) 

    # draw an axis with X red, Y green, Z blue 
    axis_length = 0.05  # meters 
    axis = np.float32([ 
        [axis_length, 0, 0],     # X 
        [0, axis_length, 0],     # Y 
        [0, 0, -axis_length],    # Z 
    ]).reshape(-1, 3) 

    imgpts, _ = cv.projectPoints(axis, rvec, tvec, mtx, dist) 
    origin = tuple(np.round(corners2[0].ravel()).astype(int)) 

    img = cv.line(img, origin, tuple(np.round(imgpts[0].ravel()).astype(int)), (0, 0, 255), 3) 
    img = cv.line(img, origin, tuple(np.round(imgpts[1].ravel()).astype(int)), (0, 255, 0), 3) 
    img = cv.line(img, origin, tuple(np.round(imgpts[2].ravel()).astype(int)), (255, 0, 0), 3) 

    # show and save images 
    cv.imshow("chessboard", img) 
    out_path = os.path.join("vis_out", os.path.basename(fname).replace(".png", "_axes.png")) 
    cv.imwrite(out_path, img) 
    cv.waitKey(1) 

def main(): 
    """ 
    main method to find images, estimate poses, and save results.
    """ 
    os.makedirs("vis_out", exist_ok=True) 
    images = sorted(glob.glob('images/*.png')) 

    with open(output_file, 'w') as f: 
        for fname in images: 
            process_image(f, fname) 

    cv.destroyAllWindows() 

if __name__ == "__main__": 
    main()