"""
训练诊断脚本 - 分析训练质量和检测问题
"""
import os
import cv2
import numpy as np
from pathlib import Path


class DiagnoseTraining:

    def check_training_images(self, path: "training/warZ/dataset/images/train"):
        """检查训练图像的红色框标注"""
        print("=" * 60)
        print("1. 检查训练图像的红色框标注")
        print("=" * 60)

        train_dir = path
        if not train_dir.exists():
            print(f"错误：训练目录不存在 {train_dir}")
            return False

        image_files = list(train_dir.glob("*.png")) + list(train_dir.glob("*.jpg"))
        print(f"\n找到 {len(image_files)} 张训练图像")

        if len(image_files) == 0:
            print("错误：没有训练图像！")
            return False

        # 检查前3张图像的红色框
        for i, img_path in enumerate(image_files[:3]):
            print(f"\n检查图像 {i + 1}: {img_path.name}")
            img = cv2.imread(str(img_path))

            if img is None:
                print(f"  错误：无法读取图像")
                continue

            # 检测红色框
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 100, 100])
            upper_red2 = np.array([180, 255, 255])

            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(mask1, mask2)

            red_pixels = np.sum(red_mask > 0)
            total_pixels = img.shape[0] * img.shape[1]
            red_ratio = red_pixels / total_pixels

            print(f"  图像尺寸: {img.shape[1]}x{img.shape[0]}")
            print(f"  红色像素: {red_pixels} ({red_ratio * 100:.2f}%)")

            if red_ratio < 0.001:
                print(f"  ⚠️ 警告：红色像素太少，可能没有红色框标注！")
            else:
                print(f"  ✓ 检测到红色标注")

        return True

    def check_labels(self, path: "training/warZ/dataset/labels/train"):
        """检查生成的标签文件"""
        print("\n" + "=" * 60)
        print("2. 检查生成的YOLO标签")
        print("=" * 60)

        label_dir = path
        if not label_dir.exists():
            print(f"错误：标签目录不存在 {label_dir}")
            return False

        label_files = list(label_dir.glob("*.txt"))
        print(f"\n找到 {len(label_files)} 个标签文件")

        if len(label_files) == 0:
            print("错误：没有标签文件！")
            return False

        total_boxes = 0
        for i, label_path in enumerate(label_files[:3]):
            print(f"\n标签文件 {i + 1}: {label_path.name}")

            with open(label_path, 'r') as f:
                lines = f.readlines()

            print(f"  标注框数量: {len(lines)}")
            total_boxes += len(lines)

            if len(lines) > 0:
                print(f"  第一个框: {lines[0].strip()}")

        print(f"\n总标注框数: {total_boxes}")

        if total_boxes == 0:
            print("⚠️ 警告：没有任何标注框！")
            return False

        return True
