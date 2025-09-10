# A series of 3D computer vision problems.

# How to execute problem 2:

cd problem2

python problem2.py \
  --input goldwin_smith.jpg \
  --output face1.png \
  --fov 90 --roll 0 --pitch 0 --yaw 0 --size 600

# How to execute problem 3:

cd problem3

calculates camera poses using chessboard
python 1_chessboard_calibration.py
aligns the two sets of camera poses of colmap and opencv
python 2_coordinate_alignment.py
converts the final poses to colmap format for viewing purposes
python 3_convert_to_colmap.py

outputs: viz_out (set of images with chessboard axis drawn), chessboard_poses.txt (poses generated from opencv), aligned_poses.txt (poses generated from aligning SfM to real world), colmap_aligned (colmap folder of final poses)

use NerF or other software to train reconstruction and convert to mesh (macbook)

colmap model_converter --input_path . --output_path . --output_type bin

ns-process-data images --data ./colmap_data/images --output-dir ./nerfstudio_data --colmap-model-path /Users/dora/Desktop/cv_dora/problem3/nerf/colmap_data/sparse/0 --skip-colmap

PYTORCH_ENABLE_MPS_FALLBACK=1 ns-train nerfacto --data ./nerfstudio_data --machine.device-type mps --mixed-precision False

ns-export poisson --load-config PATH_TO_CONFIG --output-dir ./exported_mesh

