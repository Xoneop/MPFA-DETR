# import json
# import os
# import shutil
# #此脚本为GPT5.1生成,仅供参考，完成从coco转yolo数据格式
#
# # ========== 路径设置 ==========
# ROOT = "F:\\目标检测\\无人机目标检测\\UAV-DETR Efficient End-to-End Object Detection for Unmanned\\UAV-DETR-main\\dataset\\dataset_uavvaste\\uavvaste"
# IMG_DIR = os.path.join(ROOT, "images")
# ANN_JSON = os.path.join(ROOT, "annotations", "annotations.json")
# SPLIT_JSON = os.path.join(ROOT, "annotations", "train_val_test_distribution_file.json")
# OUT_ROOT = ROOT  # 输出路径就在 uavvaste 内
#
# # ========== 加载 json ==========
# with open(ANN_JSON, 'r') as f:
#     coco = json.load(f)
#
# with open(SPLIT_JSON, 'r') as f:
#     split = json.load(f)
#
# # 建立 file_name → image_info 映射
# file_to_image = {img["file_name"]: img for img in coco["images"]}
#
# # image_id → 所有 annotations
# from collections import defaultdict
# image_to_anno = defaultdict(list)
# for ann in coco["annotations"]:
#     image_to_anno[ann["image_id"]].append(ann)
#
# # ========== 创建 YOLO 输出目录结构 ==========
# def prepare_dir(split_name):
#     split_path = os.path.join(OUT_ROOT, split_name)
#     os.makedirs(os.path.join(split_path, "images"), exist_ok=True)
#     os.makedirs(os.path.join(split_path, "labels"), exist_ok=True)
#     return split_path
#
# train_dir = prepare_dir("train")
# val_dir   = prepare_dir("val")
# test_dir  = prepare_dir("test")
#
# # ========== 将 bbox 转 YOLO 格式 ==========
# def convert_bbox_to_yolo(bbox, img_w, img_h):
#     x, y, w, h = bbox
#     x_c = x + w / 2
#     y_c = y + h / 2
#     return [
#         x_c / img_w,
#         y_c / img_h,
#         w / img_w,
#         h / img_h
#     ]
#
# # ========== 处理每个 split ==========
# def process_split(split_name, file_list):
#     split_path = os.path.join(OUT_ROOT, split_name)
#     images_out = os.path.join(split_path, "images")
#     labels_out = os.path.join(split_path, "labels")
#
#     for fname in file_list:
#         if fname not in file_to_image:
#             print(f"[WARNING] {fname} 在 annotations.json 中找不到")
#             continue
#
#         img_info = file_to_image[fname]
#         img_id = img_info["id"]
#         img_w, img_h = img_info["width"], img_info["height"]
#
#         # 复制图片
#         src_img = os.path.join(IMG_DIR, fname)
#         dst_img = os.path.join(images_out, fname)
#         if os.path.exists(src_img):
#             shutil.copy(src_img, dst_img)
#         else:
#             print(f"[WARNING] 图像不存在: {src_img}")
#
#         # 写 YOLO 标签
#         label_path = os.path.join(labels_out, fname.replace(".jpg", ".txt").replace(".JPG", ".txt"))
#         with open(label_path, "w") as f_out:
#             for ann in image_to_anno[img_id]:
#                 cls = ann["category_id"]  # UAVVaste category_id=0
#                 bbox = ann["bbox"]
#                 yolo_bbox = convert_bbox_to_yolo(bbox, img_w, img_h)
#                 f_out.write(f"{cls} {' '.join([str(round(a, 6)) for a in yolo_bbox])}\n")
#
#     print(f"✔ 完成 {split_name} 集转换，共 {len(file_list)} 张图片")
#
#
# # ========== 运行转换 ==========
# process_split("train", split["train"])
# process_split("val",   split["val"])
# process_split("test",  split["test"])
#
# print("🎉 YOLO 数据集转换完成！")

#---------------------------------------------------------------------
import os

ROOT = "/home/shuoxing/XQK/UAV-DETR-main/dataset/dataset_uavvaste/uavvaste"
#ROOT = "/home/shuoxing/XQK/UAV-DETR-main/dataset/dataset_visdrone"

splits = ["train", "val", "test"]

for split in splits:
    img_dir = os.path.join(ROOT, split, "images")
    label_dir = os.path.join(ROOT, split, "labels")

    # 统计图片数量
    images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    labels = [f for f in os.listdir(label_dir) if f.lower().endswith('.txt')]

    print(f"\n[{split.upper()}]")
    print(f"Images: {len(images)}")
    print(f"Labels: {len(labels)}")

