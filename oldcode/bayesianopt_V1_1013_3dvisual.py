import os
import sys
import argparse
import pickle
import re
from typing import Optional, Tuple, List

import numpy as np
from nibabel.freesurfer import read_geometry, read_annot, read_label


def find_surface_and_label(root_dir: str, hemi: str = "lh") -> Tuple[str, Optional[str], Optional[str]]:
    """
    Search under root_dir for FreeSurfer surface and V1 label/annot files.

    Priority:
      - surface: {hemi}.pial -> {hemi}.white
      - label: {hemi}.V1.label (or *V1*.label)
      - annot: {hemi}.V1.annot -> any annot containing a V1 entry

    Returns: (surf_path, label_path_or_None, annot_path_or_None)
    """
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Surface root directory not found: {root_dir}")

    # --- find surface ---
    # Prefer T1-aligned surfaces if present
    candidate_surfs = [f"{hemi}.pial.T1", f"{hemi}.white.T1", f"{hemi}.pial", f"{hemi}.white"]
    surf_path = None
    for name in candidate_surfs:
        for base, _, files in os.walk(root_dir):
            if name in files:
                surf_path = os.path.join(base, name)
                break
        if surf_path:
            break
    if surf_path is None:
        raise FileNotFoundError(f"Cannot find {hemi}.pial or {hemi}.white under {root_dir}")

    # --- find label ---
    label_path = None
    # exact {hemi}.V1.label
    target_label = f"{hemi}.V1.label"
    for base, _, files in os.walk(root_dir):
        if target_label in files:
            label_path = os.path.join(base, target_label)
            break
    # any *V1*.label
    if label_path is None:
        for base, _, files in os.walk(root_dir):
            for fn in files:
                if fn.endswith(".label") and "V1" in fn and fn.startswith(hemi + "."):
                    label_path = os.path.join(base, fn)
                    break
            if label_path:
                break

    # --- find annot ---
    annot_path = None
    # prefer {hemi}.V1.annot
    target_annot = f"{hemi}.V1.annot"
    for base, _, files in os.walk(root_dir):
        if target_annot in files:
            annot_path = os.path.join(base, target_annot)
            break
    # else any annot for hemi
    if annot_path is None:
        for base, _, files in os.walk(root_dir):
            for fn in files:
                if fn.endswith(".annot") and fn.startswith(hemi + "."):
                    # pick the first; later we will verify it contains V1
                    annot_path = os.path.join(base, fn)
                    break
            if annot_path:
                break

    return surf_path, label_path, annot_path


def load_mesh_as_pyvista(verts: np.ndarray, faces_tri: np.ndarray):
    """
    Lazy import pyvista and return PolyData from FreeSurfer geometry arrays.
    faces for PyVista must be packed with a leading '3' for triangles.
    """
    import pyvista as pv  # local import to avoid hard dependency on import time
    faces_pv = np.hstack([np.full((faces_tri.shape[0], 1), 3, dtype=faces_tri.dtype), faces_tri]).ravel()
    return pv.PolyData(verts, faces_pv)


def read_v1_mask(hemi: str, n_vertices: int, label_path: Optional[str], annot_path: Optional[str]) -> np.ndarray:
    """
    Build boolean mask (len = n_vertices) for V1 on the given hemisphere.
    Tries label first; if not available, tries annot with name matching.
    """
    mask = np.zeros(n_vertices, dtype=bool)

    # Try FreeSurfer label file: contains vertex indices
    if label_path and os.path.isfile(label_path):
        try:
            indices = read_label(label_path)
            indices = indices.astype(int)
            indices = indices[(indices >= 0) & (indices < n_vertices)]
            mask[indices] = True
            return mask
        except Exception as exc:
            print(f"Warning: failed to read label {label_path}: {exc}")

    # Try annot: returns per-vertex labels and table of names
    if annot_path and os.path.isfile(annot_path):
        try:
            _, labels, names = read_annot(annot_path)
            # names: list of bytes, map any entry whose decoded text contains 'V1'
            v1_name_indices: List[int] = []
            for idx, nm in enumerate(names):
                try:
                    text = nm.decode("utf-8") if isinstance(nm, (bytes, bytearray)) else str(nm)
                except Exception:
                    text = str(nm)
                if "V1" in text:
                    v1_name_indices.append(idx)
            if not v1_name_indices:
                # common alternates
                for idx, nm in enumerate(names):
                    t = nm.decode("utf-8") if isinstance(nm, (bytes, bytearray)) else str(nm)
                    if any(k in t for k in ["V1v", "V1d", "Primary_Visual", "hOc1", "area_17"]):
                        v1_name_indices.append(idx)
            if v1_name_indices:
                # labels array contains color table indices per vertex
                for k in v1_name_indices:
                    mask |= (labels == k)
                return mask
            else:
                print(f"Warning: no V1-like entry found in annot {annot_path}")
        except Exception as exc:
            print(f"Warning: failed to read annot {annot_path}: {exc}")

    raise FileNotFoundError("Could not build V1 mask: neither valid label nor annot with V1 was found.")


def rotation_yaw_pitch(alpha_deg: float, beta_deg: float) -> np.ndarray:
    """Return unit direction for yaw(beta about Z) then pitch(alpha about X) applied to +Z axis."""
    a = np.deg2rad(float(alpha_deg))
    b = np.deg2rad(float(beta_deg))
    Rz = np.array([[np.cos(b), -np.sin(b), 0.0],
                   [np.sin(b),  np.cos(b), 0.0],
                   [0.0,        0.0,       1.0]])
    Rx = np.array([[1.0, 0.0,        0.0],
                   [0.0, np.cos(a), -np.sin(a)],
                   [0.0, np.sin(a),  np.cos(a)]])
    d = (Rz @ Rx) @ np.array([0.0, 0.0, 1.0])
    n = np.linalg.norm(d)
    if n == 0:
        return np.array([0.0, 0.0, 1.0])
    return d / n


def validate_ranges(alpha: float, beta: float, offset: float, shank_len: float) -> None:
    msgs = []
    if not (-90.0 <= alpha <= 90.0):
        msgs.append(f"alpha out of range [-90,90]: {alpha}")
    # beta range can depend on hemisphere; just sanity check wide range
    if not (-180.0 <= beta <= 180.0):
        msgs.append(f"beta out of range [-180,180]: {beta}")
    if not (0.0 <= offset <= 60.0):
        msgs.append(f"offset out of range [0,60] mm: {offset}")
    if not (1.0 <= shank_len <= 80.0):
        msgs.append(f"shank_length out of range [1,80] mm: {shank_len}")
    if msgs:
        print("Warning: parameter range checks:\n - " + "\n - ".join(msgs))


def visualize(surf_path: str,
              v1_mask: np.ndarray,
              alpha: float,
              beta: float,
              offset_mm: float,
              shank_len_mm: float,
              out_png: Optional[str] = None,
              num_channels: int = 10,
              contact_spacing_mm: float = 1.5,
              shank_radius_mm: float = 0.3,
              shank_opacity: float = 0.85,
              show_ref: bool = False,
              show_main: bool = False,
              show_alpha90: bool = True,
              global_yaw_deg: float = 90.0,
              recenter: str = "hemi_medial",
              anchor_mode: str = "origin") -> None:
    import pyvista as pv

    verts, faces = read_geometry(surf_path)

    # Global yaw rotation (around Z) to align axes toward posterior if needed
    by = np.deg2rad(float(global_yaw_deg))
    Rg = np.array([[np.cos(by), -np.sin(by), 0.0],
                   [np.sin(by),  np.cos(by), 0.0],
                   [0.0,         0.0,        1.0]])

    verts_rot = (Rg @ verts.T).T

    # choose origin on one side of hemisphere and recenter mesh
    rec = (recenter or "").lower()
    if rec == "mesh_center":
        origin_point = verts_rot.mean(axis=0)
    elif rec == "v1":
        origin_point = np.median(verts_rot[v1_mask], axis=0)
    elif rec == "hemi_lateral":
        origin_point = verts_rot[np.argmax(np.abs(verts_rot[:, 0]))]
    elif rec == "anterior":
        origin_point = verts_rot[np.argmax(verts_rot[:, 1])]
    elif rec == "posterior":
        origin_point = verts_rot[np.argmin(verts_rot[:, 1])]
    elif rec == "keepx_ymax":
        # compute medial-based x,z, but move to maximum along +Y while keeping X,Z
        abs_x = np.abs(verts_rot[:, 0])
        thresh = np.percentile(abs_x, 5.0)
        medial_idx = abs_x <= thresh
        if medial_idx.any():
            medial_center = verts_rot[medial_idx].mean(axis=0)
        else:
            medial_center = verts_rot[np.argmin(abs_x)]
        origin_point = np.array([medial_center[0], np.max(verts_rot[:, 1]), medial_center[2]])
    else:  # default: hemi_medial (center of medial surface region)
        abs_x = np.abs(verts_rot[:, 0])
        thresh = np.percentile(abs_x, 5.0)
        medial_idx = abs_x <= thresh
        if medial_idx.any():
            origin_point = verts_rot[medial_idx].mean(axis=0)
        else:
            origin_point = verts_rot[np.argmin(abs_x)]

    verts_ctr = verts_rot - origin_point
    mesh = load_mesh_as_pyvista(verts_ctr, faces)

    # p0 at median of V1 vertices
    if v1_mask.sum() == 0:
        raise ValueError("V1 mask is empty; cannot compute p0")
    v1_points = verts_ctr[v1_mask]
    # anchor electrode either at origin (moved side center) or V1 median
    if (anchor_mode or "").lower() == "origin":
        p0 = np.array([0.0, 0.0, 0.0])
    else:
        p0 = np.median(v1_points, axis=0)

    # direction and points
    d = rotation_yaw_pitch(alpha, beta)
    d = Rg @ d
    p1 = p0 + offset_mm * d
    p2 = p1 + shank_len_mm * d

    plotter = pv.Plotter(window_size=(1000, 800))
    plotter.add_mesh(mesh, color="tan", opacity=0.00, specular=0.1)


    # V1 points cloud (optional visual aid)
    v1_cloud = v1_points
    if v1_cloud.shape[0] > 20000:
        sel = np.random.choice(v1_cloud.shape[0], size=20000, replace=False)
        v1_cloud = v1_cloud[sel]
    plotter.add_mesh(pv.PolyData(v1_cloud), color="white", render_points_as_spheres=True, point_size=2)

    # main electrode (yellow/gray/red) — optional
    if show_main:
        # trajectory line and offset sphere
        plotter.add_mesh(pv.Line(p0, p2), color="yellow", line_width=5)
        plotter.add_mesh(pv.Sphere(radius=max(shank_len_mm, 10.0) * 0.02, center=p1), color="cyan", opacity=0.9)

        # electrode channels as red points along the shank from p1
        if num_channels and num_channels > 0 and contact_spacing_mm > 0:
            channel_positions = [p1 + i * contact_spacing_mm * d for i in range(num_channels)]
            chan_cloud = np.vstack(channel_positions)
            plotter.add_mesh(pv.PolyData(chan_cloud), color="red", render_points_as_spheres=True, point_size=10)

        # add a tube to represent the inserted electrode shank (from p1 to p2)
        if shank_radius_mm and shank_radius_mm > 0.0:
            shank_line = pv.Line(p1, p2)
            shank_tube = shank_line.tube(radius=shank_radius_mm)
            plotter.add_mesh(shank_tube, color="gray", opacity=float(np.clip(shank_opacity, 0.05, 1.0)))

    # reference electrode with alpha=0, beta=0 (baseline insertion)
    if show_ref:
        d_ref = rotation_yaw_pitch(0.0, 0.0)
        d_ref = Rg @ d_ref
        p1_ref = p0 + offset_mm * d_ref
        p2_ref = p1_ref + shank_len_mm * d_ref
        plotter.add_mesh(pv.Line(p0, p2_ref), color="magenta", line_width=3)
        plotter.add_mesh(pv.Sphere(radius=max(shank_len_mm, 10.0) * 0.02, center=p1_ref), color="magenta", opacity=0.8)
        if num_channels and num_channels > 0 and contact_spacing_mm > 0:
            channel_positions_ref = [p1_ref + i * contact_spacing_mm * d_ref for i in range(num_channels)]
            chan_cloud_ref = np.vstack(channel_positions_ref)
            plotter.add_mesh(pv.PolyData(chan_cloud_ref), color="magenta", render_points_as_spheres=True, point_size=8)

    # auxiliary electrode with alpha=90, beta=0 for comparison
    if show_alpha90:
        d_90 = rotation_yaw_pitch(90.0, 0.0)
        d_90 = Rg @ d_90
        p1_90 = p0 + offset_mm * d_90
        p2_90 = p1_90 + shank_len_mm * d_90
        plotter.add_mesh(pv.Line(p0, p2_90), color="lime", line_width=3)
        plotter.add_mesh(pv.Sphere(radius=max(shank_len_mm, 10.0) * 0.02, center=p1_90), color="lime", opacity=0.8)
        if num_channels and num_channels > 0 and contact_spacing_mm > 0:
            channel_positions_90 = [p1_90 + i * contact_spacing_mm * d_90 for i in range(num_channels)]
            chan_cloud_90 = np.vstack(channel_positions_90)
            plotter.add_mesh(pv.PolyData(chan_cloud_90), color="lime", render_points_as_spheres=True, point_size=8)

    # axes at the recentered origin (hemi side center)
    axis_x = Rg @ np.array([20.0, 0.0, 0.0])
    axis_y = Rg @ np.array([0.0, 20.0, 0.0])
    axis_z = Rg @ np.array([0.0, 0.0, 20.0])
    plotter.add_arrows(np.array([[0.0, 0.0, 0.0]]), np.array([axis_x]), mag=1.0, color="red")
    plotter.add_arrows(np.array([[0.0, 0.0, 0.0]]), np.array([axis_y]), mag=1.0, color="blue")
    plotter.add_arrows(np.array([[0.0, 0.0, 0.0]]), np.array([axis_z]), mag=1.0, color="green")

    if show_main:
        label = f"α={alpha:.1f}°, β={beta:.1f}°, off={offset_mm:.1f}mm, L={shank_len_mm:.1f}mm"
        plotter.add_point_labels([p0], [label], point_size=20, text_color="white", font_size=14)

    if out_png:
        # Ensure out directory exists
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plotter.show(screenshot=out_png, auto_close=True)
        print(f"Saved screenshot to: {out_png}")
    else:
        plotter.show()


def _parse_params_from_txt(txt_path: str) -> Tuple[float, float, float, float]:
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()
    def find_float(pat: str) -> float:
        m = re.search(pat, content)
        if not m:
            raise KeyError(f"Pattern not found: {pat}")
        return float(m.group(1))
    alpha = find_float(r"Alpha:\s*([\d\.-]+)")
    beta = find_float(r"Beta:\s*([\d\.-]+)")
    offset = find_float(r"Offset from base:\s*([\d\.-]+)")
    shank = find_float(r"Shank length:\s*([\d\.-]+)")
    return alpha, beta, offset, shank


def _safe_pickle_load(path: str):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        # Fallback: ignore any unknown globals (e.g., local function 'f')
        class IgnoreUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                try:
                    return super().find_class(module, name)
                except Exception:
                    return None
        with open(path, "rb") as f:
            return IgnoreUnpickler(f).load()


def load_params_from_pkl(pkl_path: str) -> Tuple[float, float, float, float]:
    # Try reading parameters from sibling TXT first to avoid pickle issues
    txt_path = os.path.splitext(pkl_path)[0] + ".txt"
    if os.path.isfile(txt_path):
        try:
            return _parse_params_from_txt(txt_path)
        except Exception:
            pass

    data = _safe_pickle_load(pkl_path)
    # Expect top-level keys; also support nested common alternatives
    if isinstance(data, dict) and all(k in data for k in ("alpha", "beta", "offset", "shank_length")):
        return float(data["alpha"]), float(data["beta"]), float(data["offset"]), float(data["shank_length"]) 
    # common nesting: data['best_params']
    if isinstance(data, dict) and "best_params" in data:
        bp = data["best_params"]
        return float(bp["alpha"]), float(bp["beta"]), float(bp["offset"]), float(bp["shank_length"]) 
    # common alt names
    mapping = {
        "alpha": ["alpha", "pitch"],
        "beta": ["beta", "yaw"],
        "offset": ["offset", "offset_mm"],
        "shank_length": ["shank_length", "shank_len", "shank_len_mm"],
    }
    vals = {}
    if isinstance(data, dict):
        for key, alts in mapping.items():
            for k in alts:
                if k in data:
                    vals[key] = float(data[k])
                    break
        if len(vals) == 4:
            return vals["alpha"], vals["beta"], vals["offset"], vals["shank_length"]

    # skopt 결과를 슬림 dict로 저장한 리스트 포맷 처리: 첫 원소에 {'x': [...]} 포함
    if isinstance(data, (list, tuple)):
        for elem in data:
            if isinstance(elem, dict) and "x" in elem and hasattr(elem["x"], "__len__") and len(elem["x"]) >= 4:
                x = elem["x"]
                return float(x[0]), float(x[1]), float(x[2]), float(x[3])
    # Some older files store a list; try to recover from accompanying TXT
    if os.path.isfile(txt_path):
        return _parse_params_from_txt(txt_path)
    raise KeyError("Could not find alpha, beta, offset, shank_length in PKL/TXT")


def main():
    parser = argparse.ArgumentParser(description="Visualize V1 implantation trajectory from parameters.")
    parser.add_argument("--pkl", type=str, default=r"C:\\Users\\user\\YongtaeC\\vimplant0812\\data\\output\\102311\\102311_LH_V1_n1000_1x10_dice-yield-HD_0.05_targ-inner.pkl",
                        help="Path to PKL containing alpha, beta, offset, shank_length")
    parser.add_argument("--surf_root", type=str, default=r"C:\\Users\\user\\YongtaeC\\vimplant0812\\data\\3dvisual\\100610_surf",
                        help="Root directory to search for FreeSurfer surfaces and labels")
    parser.add_argument("--hemi", type=str, default="lh", choices=["lh", "rh"], help="Hemisphere")
    parser.add_argument("--out", type=str, default=None, help="Output PNG path; if omitted a window opens")
    parser.add_argument("--num_channels", type=int, default=10, help="Number of electrode contacts to display as red points")
    parser.add_argument("--contact_spacing_mm", type=float, default=1.5, help="Spacing between contacts in mm")
    parser.add_argument("--shank_radius_mm", type=float, default=0.3, help="Radius of electrode shank tube (mm)")
    parser.add_argument("--shank_opacity", type=float, default=0.85, help="Opacity of electrode shank [0-1]")
    parser.add_argument("--show_ref", dest="show_ref", action="store_true", default=False,
                        help="Show reference electrode with alpha=0, beta=0 (default: False)")
    parser.add_argument("--no_show_ref", dest="show_ref", action="store_false",
                        help="Do not show the reference electrode")
    parser.add_argument("--show_main", dest="show_main", action="store_true", default=False,
                        help="Show the main (yellow) electrode")
    parser.add_argument("--no_show_main", dest="show_main", action="store_false",
                        help="Do not show the main electrode (default)")
    parser.add_argument("--show_alpha90", dest="show_alpha90", action="store_true", default=True,
                        help="Show auxiliary electrode with alpha=90, beta=0 (default: True)")
    parser.add_argument("--no_show_alpha90", dest="show_alpha90", action="store_false",
                        help="Do not show the alpha=90 electrode")
    parser.add_argument("--global_yaw_deg", type=float, default=90.0,
                        help="Global yaw rotation (deg) applied to mesh and directions (default: 90)")

    args = parser.parse_args()

    print(f"Loading params from: {args.pkl}")
    alpha, beta, offset_mm, shank_len_mm = load_params_from_pkl(args.pkl)
    print(f"Parameters -> alpha: {alpha}, beta: {beta}, offset: {offset_mm} mm, shank_length: {shank_len_mm} mm")

    validate_ranges(alpha, beta, offset_mm, shank_len_mm)

    print(f"Searching surface/labels under: {args.surf_root}")
    surf_path, label_path, annot_path = find_surface_and_label(args.surf_root, hemi=args.hemi)
    print(f"Surface: {surf_path}")
    print(f"Label:   {label_path if label_path else 'None'}")
    print(f"Annot:   {annot_path if annot_path else 'None'}")

    verts, faces = read_geometry(surf_path)
    try:
        v1_mask = read_v1_mask(args.hemi, verts.shape[0], label_path, annot_path)
    except Exception as e:
        print(f"Warning: {e}. Falling back to full-surface mask.")
        v1_mask = np.ones(verts.shape[0], dtype=bool)
    print(f"V1 vertices (mask): {int(v1_mask.sum())} / {verts.shape[0]}")

    out_path = args.out
    if out_path is None:
        # default screenshot next to surf_root
        out_path = os.path.join(args.surf_root, f"implant_view_{args.hemi.upper()}.png")

    visualize(surf_path,
              v1_mask,
              alpha,
              beta,
              offset_mm,
              shank_len_mm,
              out_png=out_path,
              num_channels=args.num_channels,
              contact_spacing_mm=args.contact_spacing_mm,
              shank_radius_mm=args.shank_radius_mm,
              shank_opacity=args.shank_opacity,
              show_ref=args.show_ref,
              show_main=args.show_main,
              show_alpha90=args.show_alpha90,
              global_yaw_deg=args.global_yaw_deg,
              recenter="keepx_ymax",
              anchor_mode="origin")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


