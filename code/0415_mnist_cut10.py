# D:\yongtae\vimplant\code\0415_mnist_cut10.py

import numpy as np
from pathlib import Path

INPUT_NPZ  = Path(r"D:\yongtae\vimplant\data\letters\emnist_letters_v3_halfright.npz")
OUTPUT_NPZ = Path(r"D:\yongtae\vimplant\data\letters\mnist10.npz")

with np.load(INPUT_NPZ, allow_pickle=False) as data:
    keys = data.files
    print(f"Found keys: {keys}")

    output_dict = {}
    for key in keys:
        arr = data[key]
        # 첫 번째 차원이 N(이미지 수)인 배열만 슬라이싱, 나머지는 그대로
        if arr.ndim >= 2:
            output_dict[key] = arr[:10]
            print(f"  {key}: {arr.shape} -> {output_dict[key].shape}")
        else:
            output_dict[key] = arr
            print(f"  {key}: {arr.shape} (복사, 슬라이싱 안 함)")

np.savez_compressed(OUTPUT_NPZ, **output_dict)
print(f"\nSaved: {OUTPUT_NPZ}")

# 확인
with np.load(OUTPUT_NPZ, allow_pickle=False) as check:
    for key in check.files:
        print(f"  verified '{key}': {check[key].shape}, {check[key].dtype}")