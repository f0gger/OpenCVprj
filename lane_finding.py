#!/usr/bin/env python3
"""Part 1: chessboard camera calibration only.

Run from any directory with: python part1_calibration.py
The input/output paths below are based on this script's location, so place
this file beside the project's camera_cal/ directory.
"""
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
CALIBRATION_DIR = ROOT / "camera_cal"
CALIBRATION_FILE = ROOT / "output" / "camera_calibration.npz"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
REF_SIZE = (1280, 720)
BOARD = (9, 6)

def load_calibration(path):
    d = np.load(path, allow_pickle=False)
    return {'K': d['camera_matrix'], 'dist': d['dist_coeffs'], 'w': int(d['image_width']), 'h': int(d['image_height']), 'rms': float(d['rms'])}

def calibrate_camera(calibration_dir, save_file):
    paths = sorted((p for p in Path(calibration_dir).rglob('*') if p.is_file() and p.suffix.lower() in IMG_EXT))
    objp = np.zeros((BOARD[0] * BOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD[0], 0:BOARD[1]].T.reshape(-1, 2)
    objs = []
    imgs = []
    good = []
    bad = []
    for p in paths:
        im = cv2.imread(str(p))
        if im is None:
            bad.append(p.name)
            continue
        if (im.shape[1], im.shape[0]) != REF_SIZE:
            im = cv2.resize(im, REF_SIZE)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        ok, c = cv2.findChessboardCorners(gray, BOARD, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if (not ok or c is None or len(c) != BOARD[0] * BOARD[1]) and hasattr(cv2, 'findChessboardCornersSB'):
            ok, c = cv2.findChessboardCornersSB(gray, BOARD)
        if not ok or c is None or len(c) != BOARD[0] * BOARD[1]:
            bad.append(p.name)
            continue
        c = np.asarray(c, np.float32).reshape(-1, 1, 2)
        c = cv2.cornerSubPix(gray, c, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        objs.append(objp.copy())
        imgs.append(c)
        good.append(p.name)
    if len(objs) < 5:
        raise RuntimeError(f'Only {len(objs)} usable chessboard images')
    rms, K, d, _, _ = cv2.calibrateCamera(objs, imgs, REF_SIZE, None, None)
    save_file = Path(save_file)
    save_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(save_file, camera_matrix=K, dist_coeffs=d, image_width=np.int32(1280), image_height=np.int32(720), rms=np.float64(rms), accepted=np.asarray(good), rejected=np.asarray(bad))
    print(f'Calibration: {len(good)}/{len(paths)} images accepted; reprojection RMS={rms:.4f}px; skipped={bad}', flush=True)
    return load_calibration(save_file)

def main():
    if not CALIBRATION_DIR.is_dir():
        raise FileNotFoundError(
            f"Calibration folder not found: {CALIBRATION_DIR}\n"
            "Place this script beside the camera_cal/ folder, or edit "
            "CALIBRATION_DIR near the top of this file."
        )
    calibrate_camera(CALIBRATION_DIR, CALIBRATION_FILE)
    print(f"Saved calibration parameters to: {CALIBRATION_FILE}")


if __name__ == "__main__":
    main()
