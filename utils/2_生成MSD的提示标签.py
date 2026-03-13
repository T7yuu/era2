'''import os
import cv2
import numpy as np
from tqdm import tqdm


def get_scaled_bbox_from_mask(mask_path, scale_factor=1.1):
    try:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"\n警告：无法读取图片 {os.path.basename(mask_path)}，已跳过。")
            return "null"

        rows, cols = np.where(mask == 255)

        if len(rows) == 0:
            return "null"

        img_h, img_w = mask.shape

        x_min, x_max = np.min(cols), np.max(cols)
        y_min, y_max = np.min(rows), np.max(rows)

        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min

        new_width = width * scale_factor
        new_height = height * scale_factor

        new_x_min = center_x - new_width / 2
        new_y_min = center_y - new_height / 2
        new_x_max = center_x + new_width / 2
        new_y_max = center_y + new_height / 2

        final_x_min = max(0, new_x_min)
        final_y_min = max(0, new_y_min)
        final_x_max = min(img_w - 1, new_x_max)
        final_y_max = min(img_h - 1, new_y_max)

        return [int(final_x_min), int(final_y_min), int(final_x_max), int(final_y_max)]

    except Exception as e:
        print(f"\n处理文件 {os.path.basename(mask_path)} 时发生未知错误: {e}")
        return "null"


def process_directory_recursively(src_root, dest_root, scale_factor=1.1):
    if not os.path.isdir(src_root):
        print(f"警告：源文件夹 '{src_root}' 不存在，已跳过。")
        return

    print(f"正在处理任务: '{os.path.basename(src_root)}'")
    print(f"  -> 源目录: {src_root}")
    print(f"  -> 目标目录: {dest_root}")

    dir_walk = list(os.walk(src_root))
    for current_dir, _, files in tqdm(dir_walk, desc="  -> 进度"):

        image_files = [f for f in files if
                       f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.nii.gz'))]
        if not image_files:
            continue

        relative_path = os.path.relpath(current_dir, src_root)

        output_dir = os.path.join(dest_root, relative_path)
        os.makedirs(output_dir, exist_ok=True)

        for filename in image_files:
            mask_path = os.path.join(current_dir, filename)

            result = get_scaled_bbox_from_mask(mask_path, scale_factor)

            if isinstance(result, str):
                bbox_str = result
            else:
                bbox_str = ' '.join(map(str, result))

            base_filename = os.path.splitext(filename)[0]
            if base_filename.lower().endswith('.nii'):
                base_filename = os.path.splitext(base_filename)[0]

            output_path = os.path.join(output_dir, f"{base_filename}.txt")

            with open(output_path, 'w') as f:
                f.write(bbox_str)


if __name__ == "__main__":
    #folder_mapping = {
    #    r"D:\Datasets\MSD\Task02_Heart\labelsTr": r"D:\Datasets\MedSAM_Prompts\MSD\task02_heart",
    #    r"D:\Datasets\MSD\Task04_Hippocampus\labelsTr": r"D:\Datasets\MedSAM_Prompts\MSD\task04_hippocampus",
    #    r"D:\Datasets\MSD\Task05_Prostate\labelsTr": r"D:\Datasets\MedSAM_Prompts\MSD\task05_prostate",
   #     r"D:\Datasets\MSD\Task09_Spleen\labelsTr": r"D:\Datasets\MedSAM_Prompts\MSD\task09_spleen"
    #}

    folder_mapping = {
        r"E:\Datasets\MSD\Task09_Spleen\labelsTr": r"E:\Datasets\MSD\task09_spleen\prompts"
    }

    for src, dest in folder_mapping.items():
        process_directory_recursively(src, dest)
        print("-" * 60)

    print("\n所有任务处理完成！")
'''
import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
from pathlib import Path


def get_bboxes_from_nifti(nii_path, scale_factor=1.1):
    """
    加载 3D NIfTI 文件，并返回每一层切片的 BBox 字典
    """
    try:
        # 使用 nibabel 读取 3D 数据
        nii_img = nib.load(str(nii_path))
        # 获取数据并统一转为 [H, W, Slices] 格式
        data = nii_img.get_fdata()

        # 处理可能的维度异常 (确保是 3D)
        if len(data.shape) == 4:
            data = data[:, :, :, 0]

        img_h, img_w, num_slices = data.shape
        slice_bboxes = {}

        # 遍历每一层切片
        for s in range(num_slices):
            slice_mask = data[:, :, s]

            # 这里的阈值设为 0.5 是因为 get_fdata() 可能返回浮点数
            rows, cols = np.where(slice_mask > 0.5)

            if len(rows) == 0:
                slice_bboxes[s] = "null"
                continue

            # 计算基础 BBox
            x_min, x_max = np.min(cols), np.max(cols)
            y_min, y_max = np.min(rows), np.max(rows)

            # 缩放逻辑 (Scale Factor)
            center_x, center_y = (x_min + x_max) / 2, (y_min + y_max) / 2
            w, h = (x_max - x_min), (y_max - y_min)

            new_w, new_h = w * scale_factor, h * scale_factor

            f_x_min = max(0, int(center_x - new_w / 2))
            f_y_min = max(0, int(center_y - new_h / 2))
            f_x_max = min(img_w - 1, int(center_x + new_w / 2))
            f_y_max = min(img_h - 1, int(center_y + new_h / 2))

            slice_bboxes[s] = f"{f_x_min} {f_y_min} {f_x_max} {f_y_max}"

        return slice_bboxes

    except Exception as e:
        print(f"\n读取 NIfTI 文件 {os.path.basename(nii_path)} 失败: {e}")
        return None


def process_msd_tasks(folder_mapping, scale_factor=1.1):
    for src_root, dest_root in folder_mapping.items():
        src_path = Path(src_root)
        if not src_path.exists():
            continue

        print(f"\n正在处理任务: {src_path.parent.name}")

        nii_files = [f for f in src_path.glob("*.nii.gz") if not f.name.startswith("._")]

        for nii_file in tqdm(nii_files, desc="处理病例"):
            case_name = nii_file.name.replace('.nii.gz', '')
            # 为每个病例创建一个子文件夹
            case_output_dir = Path(dest_root) / case_name
            case_output_dir.mkdir(parents=True, exist_ok=True)

            # 获取该病例所有切片的 BBox
            all_bboxes = get_bboxes_from_nifti(nii_file, scale_factor)

            if all_bboxes:
                for slice_idx, bbox_str in all_bboxes.items():
                    # 文件命名格式：spleen_1_50.txt (对应第 50 层切片)
                    output_file = case_output_dir / f"{case_name}_{slice_idx}.txt"
                    with open(output_file, 'w') as f:
                        f.write(bbox_str)


if __name__ == "__main__":
    # 路径配置
    folder_mapping = {
        r"E:\Datasets\MSD\Task09_Spleen\labelsTr": r"E:\Datasets\MSD\task09_spleen\prompts"
    }

    process_msd_tasks(folder_mapping)
    print("\n✅ 所有 MSD 提示标签生成完成！")
    print(f"提示文件已保存至: {list(folder_mapping.values())[0]}")