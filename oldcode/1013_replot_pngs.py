# -*- coding: utf-8 -*-
import os
import re
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path

from electphos import create_grid_v2, implant_grid, get_phosphenes, prf_to_phos, get_cortical_magnification, cortical_spread


WINDOWSIZE = 1000
view_angle = 90
amp = 100


def parse_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    subj = int(re.search(r"Subject:\s*(\d+)", content).group(1))
    hem = re.search(r"Hemisphere:\s*(LH|RH)", content).group(1)
    alpha = float(re.search(r"Alpha:\s*([\d\.-]+)", content).group(1))
    beta = float(re.search(r"Beta:\s*([\d\.-]+)", content).group(1))
    offset = float(re.search(r"Offset from base:\s*([\d\.-]+)", content).group(1))
    shank = float(re.search(r"Shank length:\s*([\d\.-]+)", content).group(1))
    return subj, hem, alpha, beta, offset, shank


def load_subject_maps(subject_id, data_root):
    data_dir = os.path.join(data_root, str(subject_id)) + "/"
    fname_ang = 'inferred_angle.mgz'
    fname_ecc = 'inferred_eccen.mgz'
    fname_sigma = 'inferred_sigma.mgz'
    fname_aparc = 'aparc+aseg.mgz'
    fname_label = 'inferred_varea.mgz'

    ang_img = nib.load(data_dir + fname_ang)
    polar_map = ang_img.get_fdata()
    ecc_img = nib.load(data_dir + fname_ecc)
    ecc_map = ecc_img.get_fdata()
    sigma_img = nib.load(data_dir + fname_sigma)
    sigma_map = sigma_img.get_fdata()
    aparc_img = nib.load(data_dir + fname_aparc)
    aparc_roi = aparc_img.get_fdata()
    label_img = nib.load(data_dir + fname_label)
    label_map = label_img.get_fdata()

    return polar_map, ecc_map, sigma_map, aparc_roi, label_map


def build_coords(aparc_roi, label_map, hem, polar_map, ecc_map):
    if hem == 'LH':
        gm_coords = np.where(aparc_roi > 2000)
    else:
        gm_coords = np.where((aparc_roi >= 1000) & (aparc_roi < 2000))
    good_coords = np.asarray(np.where((ecc_map * polar_map) != 0.0))
    return good_coords, gm_coords


def compute_phosphene_map(alpha, beta, offset_from_base, shank_length,
                          polar_map, ecc_map, sigma_map, gm_coords, good_coords):
    xl, yl, zl = gm_coords
    GM = np.array([xl, yl, zl]).T
    median = [np.median(xl), np.median(yl), np.median(zl)]

    orig_grid = create_grid_v2(
        median,
        shank_length=shank_length,
        n_x=10, n_y=10, n_z=10,
        spacing_x=1, spacing_y=1,
        offset_from_origin=0
    )

    new_angle = (float(alpha), float(beta), 0)
    _, contacts_xyz_moved, _, _, _, _, _, _, grid_valid = implant_grid(GM, orig_grid, median, new_angle, offset_from_base)

    phosphenes = get_phosphenes(contacts_xyz_moved, good_coords, polar_map, ecc_map, sigma_map)
    M = 1 / get_cortical_magnification(phosphenes[:,1], 'wedge-dipole')
    spread = cortical_spread(amp)
    sizes = spread * M
    sigmas = sizes / 2
    phosphenes[:,2] = sigmas

    phospheneMap = np.zeros((WINDOWSIZE, WINDOWSIZE), 'float32')
    phospheneMap = prf_to_phos(phospheneMap, phosphenes, view_angle=view_angle, phSizeScale=1)
    m = phospheneMap.max()
    if m > 0:
        phospheneMap /= m
    s = phospheneMap.sum()
    if s > 0:
        phospheneMap /= s
    return phospheneMap, grid_valid


def replot_base(base_dir, data_root):
    base = Path(base_dir)
    txt_files = list(base.rglob('*.txt'))
    for txt in txt_files:
        try:
            subj, hem, alpha, beta, offset, shank = parse_txt(txt)
            polar_map, ecc_map, sigma_map, aparc_roi, label_map = load_subject_maps(subj, data_root)
            good_coords, gm_coords = build_coords(aparc_roi, label_map, hem, polar_map, ecc_map)
            phospheneMap, _ = compute_phosphene_map(alpha, beta, offset, shank, polar_map, ecc_map, sigma_map, gm_coords, good_coords)
            out_png = txt.with_name(txt.stem.replace('.txt', '') + '_phosphene_map.png')
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.imshow(phospheneMap, cmap='seismic', origin='lower', vmin=0, vmax=np.max(phospheneMap)/100 if np.max(phospheneMap)>0 else 1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(out_png, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[RENDER] {out_png}")
        except Exception as e:
            print(f"[SKIP] {txt} -> {e}")


if __name__ == '__main__':
    # 예시: newtarget 결과 또는 spacing 결과 베이스 폴더 지정
    # data_root는 data/input/{subject}/ 구조의 상위 폴더
    replot_base("C:/Users/user/YongtaeC/vimplant0812/data/output/100610_newtarget_1013/", "C:/Users/user/YongtaeC/vimplant0812/data/input/")
    # replot_base("C:/Users/user/YongtaeC/vimplant0812/data/output/100610_spacing/", "C:/Users/user/YongtaeC/vimplant0812/data/input/")


