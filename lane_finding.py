#!/usr/bin/env python3
"""Part 2: calibration and still-image lane recognition.

Run with no arguments to process every image under test_images/.
Optional flags let you change folders or force recalibration.
This stage intentionally does not include video processing or temporal tracking.
"""
from pathlib import Path
import argparse
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
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

def undistort(im, cal, cache):
    h, w = im.shape[:2]
    K = cal['K'].copy().astype(np.float64)
    K[0, 0] *= w / cal['w']
    K[0, 2] *= w / cal['w']
    K[1, 1] *= h / cal['h']
    K[1, 2] *= h / cal['h']
    key = (w, h)
    if key not in cache:
        newK, _ = cv2.getOptimalNewCameraMatrix(K, cal['dist'], (w, h), 0, (w, h))
        cache[key] = cv2.initUndistortRectifyMap(K, cal['dist'], None, newK, (w, h), cv2.CV_16SC2)
    return cv2.remap(im, *cache[key], interpolation=cv2.INTER_LINEAR)

def threshold_markings(im):
    h, w = im.shape[:2]
    hls = cv2.cvtColor(im, cv2.COLOR_BGR2HLS)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    L = hls[:, :, 1]
    S = hls[:, :, 2]
    white = cv2.inRange(hls, np.array([0, 190, 0], np.uint8), np.array([180, 255, 160], np.uint8))
    yh = cv2.inRange(hls, np.array([12, 70, 55], np.uint8), np.array([42, 255, 255], np.uint8))
    yv = cv2.inRange(hsv, np.array([12, 70, 70], np.uint8), np.array([42, 255, 255], np.uint8))
    yellow = cv2.bitwise_or(yh, yv)
    sx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    mn, mx, _, _ = cv2.minMaxLoc(sx)
    peak = max(abs(mn), abs(mx))
    grad = cv2.inRange(cv2.convertScaleAbs(sx, alpha=255 / max(1, peak)), 22, 150) if peak > 0 else np.zeros((h, w), np.uint8)
    grad = cv2.bitwise_and(grad, cv2.inRange(L, 55, 255))
    grad = cv2.bitwise_and(grad, cv2.bitwise_or(cv2.inRange(S, 0, 175), yellow))
    binary = cv2.bitwise_or(cv2.bitwise_or(white, yellow), grad)
    roi = np.zeros((h, w), np.uint8)
    poly = np.array([[[int(0.04 * w), h - 1], [int(0.42 * w), int(0.56 * h)], [int(0.58 * w), int(0.56 * h)], [int(0.98 * w), h - 1]]], np.int32)
    cv2.fillPoly(roi, poly, 255)
    binary = cv2.bitwise_and(binary, roi)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    return (binary > 0).astype(np.uint8)

def perspective_matrices(w, h):
    src = np.float32([[0.455 * w, 0.64 * h], [0.545 * w, 0.64 * h], [0.875 * w, 0.99 * h], [0.155 * w, 0.99 * h]])
    dst = np.float32([[0.25 * w, 0], [0.75 * w, 0], [0.75 * w, h - 1], [0.25 * w, h - 1]])
    return (cv2.getPerspectiveTransform(src, dst), cv2.getPerspectiveTransform(dst, src))

def warp(im, M, interp=cv2.INTER_LINEAR):
    h, w = im.shape[:2]
    return cv2.warpPerspective(im, M, (w, h), flags=interp)

def fit_lanes(b):
    h, w = b.shape
    hist = np.sum(b[h // 2:, :], axis=0)
    mid = w // 2
    ls = max(0, int(0.08 * w))
    le = min(mid, int(0.44 * w))
    rs = max(mid, int(0.56 * w))
    re = min(w, int(0.92 * w))
    left = ls + int(np.argmax(hist[ls:le])) if le > ls and hist[ls:le].max(initial=0) > 0 else None
    right = rs + int(np.argmax(hist[rs:re])) if re > rs and hist[rs:re].max(initial=0) > 0 else None
    if left is None and hist[:mid].max(initial=0) > 0:
        left = int(np.argmax(hist[:mid]))
    if right is None and hist[mid:].max(initial=0) > 0:
        right = mid + int(np.argmax(hist[mid:]))
    if left is None or right is None or right <= left:
        return None
    nwin = 9
    margin = max(35, int(0.075 * w))
    minpix = max(15, int(6e-05 * w * h))
    winh = max(1, h // nwin)
    lc, rc = (left, right)
    lx = []
    ly = []
    rx = []
    ry = []
    for k in range(nwin):
        ylo = h - (k + 1) * winh
        yhi = h - k * winh if k < nwin - 1 else h
        yy, xx = b[ylo:yhi].nonzero()
        yy = yy + ylo
        il = (xx >= lc - margin) & (xx < lc + margin)
        ir = (xx >= rc - margin) & (xx < rc + margin)
        if il.any():
            lx.append(xx[il])
            ly.append(yy[il])
        if ir.any():
            rx.append(xx[ir])
            ry.append(yy[ir])
        if il.sum() > minpix:
            lc = int(xx[il].mean())
        if ir.sum() > minpix:
            rc = int(xx[ir].mean())
    lx = np.concatenate(lx) if lx else np.array([])
    ly = np.concatenate(ly) if ly else np.array([])
    rx = np.concatenate(rx) if rx else np.array([])
    ry = np.concatenate(ry) if ry else np.array([])
    if len(lx) < max(60, int(h * 0.25)) or len(rx) < max(60, int(h * 0.25)):
        return None
    lf = np.polyfit(ly, lx, 2)
    rf = np.polyfit(ry, rx, 2)
    y = np.arange(h, dtype=float)
    lcurve = np.polyval(lf, y)
    rcurve = np.polyval(rf, y)
    wb = rcurve[-1] - lcurve[-1]
    wt = rcurve[int(0.25 * h)] - lcurve[int(0.25 * h)]
    if not (0.22 * w <= wb <= 0.78 * w and 0.16 * w <= wt <= 0.82 * w):
        return None
    support = cv2.dilate(b, np.ones((1, 2 * max(4, int(0.008 * w)) + 1), np.uint8))

    def occ(f):
        yy = np.arange(int(0.25 * h), h, max(1, h // 220))
        xx = np.rint(np.polyval(f, yy)).astype(int)
        ok = (xx >= 0) & (xx < w)
        return float(np.mean(support[yy[ok], xx[ok]] > 0)) if ok.any() else 0
    lo, ro = (occ(lf), occ(rf))
    sty = lambda a: 'solid' if a >= 0.68 else 'dashed' if a >= 0.16 else 'uncertain'
    return {'lf': lf, 'rf': rf, 'y': y, 'lx': lcurve, 'rx': rcurve, 'lt': sty(lo), 'rt': sty(ro)}

def measure(m, w, h):
    lane_px = max(1.0, float(np.polyval(m['rf'], h - 1) - np.polyval(m['lf'], h - 1)))
    xm = 3.7 / lane_px
    ym = 30.0 / h
    y = (h - 1) * ym

    def conv(p):
        return np.array([p[0] * xm / ym ** 2, p[1] * xm / ym, p[2] * xm])

    def rad(p):
        a, b, _ = p
        den = abs(2 * a)
        return np.inf if den < 1e-09 else (1 + (2 * a * y + b) ** 2) ** 1.5 / den
    rs = [rad(conv(p)) for p in (m['lf'], m['rf'])]
    rs = [x for x in rs if np.isfinite(x)]
    curv = float(np.mean(rs)) if rs else np.inf
    center = (np.polyval(m['lf'], h - 1) + np.polyval(m['rf'], h - 1)) / 2
    off = float((center - w / 2) * xm)
    return (curv, off)

def overlay(im, m, Minv):
    h, w = im.shape[:2]
    y = np.arange(h, dtype=float)
    lx = np.clip(np.polyval(m['lf'], y), 0, w - 1)
    rx = np.clip(np.polyval(m['rf'], y), 0, w - 1)
    layer = np.zeros_like(im)
    polygon = np.vstack([np.column_stack([lx, y]), np.column_stack([rx[::-1], y[::-1]])]).astype(np.int32)
    cv2.fillPoly(layer, [polygon], (0, 170, 0))
    color = lambda t: (0, 220, 255) if t == 'solid' else (255, 220, 0) if t == 'dashed' else (200, 0, 220)
    thick = max(3, w // 320)
    for x, t in [(lx, m['lt']), (rx, m['rt'])]:
        cv2.polylines(layer, [np.column_stack([x, y]).astype(np.int32).reshape(-1, 1, 2)], False, color(t), thick)
    out = cv2.addWeighted(im, 1.0, warp(layer, Minv), 0.32, 0)
    curv, off = measure(m, w, h)
    ctext = f'Radius of curvature: {curv:,.0f} m' if np.isfinite(curv) else 'Radius of curvature: very large'
    otext = 'Vehicle near lane center' if abs(off) < 0.02 else f'Vehicle {abs(off):.2f} m left of lane center' if off > 0 else f'Vehicle {abs(off):.2f} m right of lane center'
    scale = max(0.55, w / 1500)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for yy, txt, col, sc in [(int(0.055 * h), ctext, (255, 255, 255), scale), (int(0.1 * h), otext, (255, 255, 255), scale), (int(0.15 * h), f"Left marking: {m['lt']}", color(m['lt']), scale * 0.8), (int(0.19 * h), f"Right marking: {m['rt']}", color(m['rt']), scale * 0.8)]:
        cv2.putText(out, txt, (int(0.035 * w), yy), font, sc, col, 2, cv2.LINE_AA)
    return (out, {'curvature_m': curv, 'offset_m': off, 'left_type': m['lt'], 'right_type': m['rt']})

class Pipeline:
    """Apply the image-processing stages to one still image."""
    def __init__(self, cal):
        self.cal = cal
        self.cache = {}

    def process(self, img, debug=False):
        und = undistort(img, self.cal, self.cache)
        binary = threshold_markings(und)
        h, w = binary.shape
        M, Minv = perspective_matrices(w, h)
        warped_binary = warp(binary, M, cv2.INTER_NEAREST)
        fit = fit_lanes(warped_binary)

        if fit is None:
            out = und.copy()
            cv2.putText(
                out,
                "Lane markings not confidently detected",
                (int(.035 * w), int(.07 * h)),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(.55, w / 1500),
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            measurements = {
                "curvature_m": None,
                "offset_m": None,
                "left_type": "unknown",
                "right_type": "unknown",
            }
        else:
            out, measurements = overlay(und, fit, Minv)

        if debug:
            return out, measurements, {
                "undistorted": und,
                "binary": binary * 255,
                "warped_road": warp(und, M),
                "warped_binary": warped_binary * 255,
            }
        return out, measurements


def process_images(indir, outdir, calfile, debugdir):
    pipe = Pipeline(load_calibration(calfile))
    rows = []
    for p in sorted(Path(indir).rglob('*')):
        if not p.is_file() or p.suffix.lower() not in IMG_EXT:
            continue
        im = cv2.imread(str(p))
        if im is None:
            continue
        out, met, dbg = pipe.process(im, True)
        op = Path(outdir) / (p.stem + '_annotated.jpg')
        op.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(op), out)
        if debugdir:
            Path(debugdir).mkdir(parents=True, exist_ok=True)
            for k, v in dbg.items():
                cv2.imwrite(str(Path(debugdir) / (p.stem + '_' + k + '.png')), v)
        rows.append({'input': p.name, 'output': op.name, **met})
        print(f"Image {p.name}: {met['left_type']}/{met['right_type']}", flush=True)
    return rows

def main():
    parser = argparse.ArgumentParser(
        description="Calibrate the camera if needed, then process still images."
    )
    parser.add_argument("--calibration-dir", type=Path, default=ROOT / "camera_cal")
    parser.add_argument("--calibration-file", type=Path, default=ROOT / "output" / "camera_calibration.npz")
    parser.add_argument("--input-dir", type=Path, default=ROOT / "test_images")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "images")
    parser.add_argument("--debug-dir", type=Path, default=ROOT / "output" / "debug")
    parser.add_argument("--recalibrate", action="store_true", help="Recalculate calibration even if a saved file exists.")
    parser.add_argument("--no-debug", action="store_true", help="Do not save intermediate images.")
    args = parser.parse_args()

    if args.recalibrate or not args.calibration_file.exists():
        calibrate_camera(args.calibration_dir, args.calibration_file)
    elif not args.calibration_file.is_file():
        raise FileNotFoundError(f"Calibration file not found: {args.calibration_file}")

    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Image folder not found: {args.input_dir}")

    debug_dir = None if args.no_debug else args.debug_dir
    rows = process_images(args.input_dir, args.output_dir, args.calibration_file, debug_dir)
    if not rows:
        print(f"No supported images found in {args.input_dir}")
        return 1
    print(f"Processed {len(rows)} image(s). Results: {args.output_dir}")
    if debug_dir:
        print(f"Intermediate images: {debug_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
