import numpy as np
import cv2
import argparse
import os

def rpy_matrix(roll_deg, pitch_deg, yaw_deg):
    """
    Build rotation matrix applying roll -> pitch -> yaw.
    Camera axes: x=right, y=up, z=forward.
    """

    # convert to radians and adjust signs to match convention
    r = -np.deg2rad(roll_deg)
    p = -np.deg2rad(pitch_deg)
    y =  -np.deg2rad(yaw_deg)

    Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    Ry = np.array([[ np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])

    # apply roll then pitch then yaw, reverse for matrix mult.
    return Ry @ Rx @ Rz

def equirect_to_perspective(erm, fov=90, roll=0, pitch=0, yaw=0, out_size=800):
    """
    returns perspective view given erm and settings.
    """
    H, W = erm.shape[:2]
    fov = np.deg2rad(fov)
    half = np.tan(fov / 2.0)

    # create camera-space directions for each pixel
    i = np.linspace(-half, half, out_size)
    j = np.linspace(-half, half, out_size)
    xx, yy = np.meshgrid(i, -j)

    # 3d ray direction vector ([x, y, 1])
    dirs_cam = np.stack([xx, yy, np.ones_like(xx)], axis=-1)
    dirs_cam /= np.linalg.norm(dirs_cam, axis=-1, keepdims=True)

    # rotate to world (spherical)
    R = rpy_matrix(roll, pitch, yaw)
    dirs_world = dirs_cam @ R.T

    # convert to lat & long
    xw = dirs_world[..., 0]
    yw = dirs_world[..., 1]
    zw = dirs_world[..., 2]
    lam = np.arctan2(zw, xw)
    phi = np.arcsin(np.clip(yw, -1.0, 1.0))

    # map to ERM pixel coords
    u = (lam + np.pi) * (W / (2.0 * np.pi))
    v = (np.pi / 2.0 - phi) * (H / np.pi)

    # make sure u and v stay within width and height borders
    u = u.astype(np.float32) % W
    v = np.clip(v, 0, H - 1).astype(np.float32)

    # copy color over, use bilinear interpolation to smooth out results/ connect border seamlessly
    persp = cv2.remap(erm, u, v, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    return persp

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--fov", type=float, default=90.0)
    p.add_argument("--roll", type=float, default=25.0)
    p.add_argument("--pitch", type=float, default=10.0)
    p.add_argument("--yaw", type=float, default=30.0)
    p.add_argument("--size", type=int, default=800)
    return p.parse_args()

def main():
    args = parse_args()

    erm = cv2.imread(args.input, cv2.IMREAD_COLOR)

    view = equirect_to_perspective(
        erm=erm,
        fov=args.fov,
        roll=args.roll,
        pitch=args.pitch,
        yaw=args.yaw,
        out_size=args.size
    )

    ok = cv2.imwrite(args.output, view)
    print(f"Settings: fov={args.fov}°, yaw={args.yaw}°, pitch={args.pitch}°, roll={args.roll}°, size={args.size}px")

if __name__ == "__main__":
    main()