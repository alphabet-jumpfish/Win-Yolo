"""
YOLO 自定义模型训练类
用于训练专门识别特定人物的精准模型
"""
import os
from pathlib import Path
from ultralytics import YOLO
import yaml
import shutil
import numpy as np
import cv2


class YOLOTrainer:
    """
    YOLO 模型训练器
    用于训练自定义的人物检测模型
    """

    def __init__(self, project_name='person_detection', base_model='yolov8n.pt', model_dir='./models',
                 path='./training'):
        """
        初始化训练器
        :param project_name: 项目名称
        :param base_model: 基础模型（yolov8n.pt, yolov8s.pt, yolov8m.pt等）
        """
        self.project_name = project_name
        self.base_model = base_model
        self.model_dir = model_dir

        # 设置模型路径
        load_model_path = Path(model_dir) / base_model
        self.model_path = load_model_path
        # 加载基础模型以获取原有类别
        print(f"正在加载基础模型: {self.model_path}")
        base_model_obj = YOLO(str(load_model_path))
        self.base_model_classes = list(base_model_obj.names.values())
        print(f"基础模型包含 {len(self.base_model_classes)} 个类别")

        # 创建项目目录结构
        self.project_dir = Path(path + f'/{project_name}')
        self.dataset_dir = self.project_dir / 'dataset'
        self.images_dir = self.dataset_dir / 'images'
        self.labels_dir = self.dataset_dir / 'labels'

        # 创建训练集和验证集目录
        self.train_images_dir = self.images_dir / 'train'
        self.val_images_dir = self.images_dir / 'val'
        self.train_labels_dir = self.labels_dir / 'train'
        self.val_labels_dir = self.labels_dir / 'val'

        self._create_directories()

        print(f"训练项目已创建: {self.project_dir.absolute()}")

    def _create_directories(self):
        """创建所有必要的目录"""
        directories = [
            self.train_images_dir,
            self.val_images_dir,
            self.train_labels_dir,
            self.val_labels_dir
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def merge_class_names(self, new_class_names):
        """
        合并基础模型的类别和新类别
        :param new_class_names: 新增的类别名称列表
        :return: 合并后的完整类别列表
        """
        # 合并类别：保留原有类别 + 添加新类别
        merged_classes = self.base_model_classes.copy()

        for new_class in new_class_names:
            if new_class not in merged_classes:
                merged_classes.append(new_class)

        print(f"\n类别合并完成:")
        print(f"  原有类别数: {len(self.base_model_classes)}")
        print(f"  新增类别数: {len(new_class_names)}")
        print(f"  合并后总数: {len(merged_classes)}")
        print(f"  新增类别: {new_class_names}")

        return merged_classes

    def create_dataset_yaml(self, class_names=['person']):
        """
        创建数据集配置文件
        :param class_names: 类别名称列表
        """
        dataset_config = {
            'path': str(self.dataset_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(class_names),
            'names': class_names
        }

        yaml_path = self.dataset_dir / 'data.yaml'
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)

        print(f"数据集配置文件已创建: {yaml_path}")
        return yaml_path

    def split_dataset(self, val_ratio=0.2):
        """
        将数据集分割为训练集和验证集
        :param val_ratio: 验证集比例
        """
        import random

        print(f"开始分割数据集（验证集比例: {val_ratio}）...")

        # 获取所有训练图像
        image_files = list(self.train_images_dir.glob('*.jpg')) + \
                      list(self.train_images_dir.glob('*.png'))

        # 随机打乱
        random.shuffle(image_files)

        # 计算分割点
        split_idx = int(len(image_files) * (1 - val_ratio))
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]

        # 移动验证集文件
        for img_path in val_files:
            # 移动图像
            shutil.move(str(img_path), str(self.val_images_dir / img_path.name))

            # 移动对应的标签
            label_path = self.train_labels_dir / f'{img_path.stem}.txt'
            if label_path.exists():
                shutil.move(str(label_path), str(self.val_labels_dir / label_path.name))

        print(f"数据集分割完成！")
        print(f"训练集: {len(train_files)} 张")
        print(f"验证集: {len(val_files)} 张")

    def train(self, epochs=50, imgsz=640, batch=16, device='cpu', class_names=None):
        """
        训练模型
        :param epochs: 训练轮数
        :param imgsz: 图像大小
        :param batch: 批次大小
        :param device: 设备 ('cpu' 或 'cuda')
        :param class_names: 类别名称列表
        """
        print("=" * 50)
        print("开始训练模型...")
        print("=" * 50)

        # 创建数据集配置文件（如果提供了class_names则使用，否则使用默认值）
        if class_names is not None:
            yaml_path = self.create_dataset_yaml(class_names=class_names)
        else:
            yaml_path = self.create_dataset_yaml()

        # 加载基础模型
        model = YOLO(str(self.model_path))

        # 开始训练
        results = model.train(
            data=str(yaml_path),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=str(self.project_dir / 'runs'),
            name='train',
            exist_ok=True,
            amp=False  # 禁用自动混合精度训练，避免AMP检查下载模型
        )

        print("=" * 50)
        print("训练完成！")
        print("=" * 50)

        return results

    def detect_red_boxes(self, image_path):
        """
        检测图像中的红色框框（空心矩形框）
        :param image_path: 图像路径
        :return: 检测到的边界框列表 [(x1, y1, x2, y2), ...]
        """
        # 读取图像
        img = cv2.imread(str(image_path))
        if img is None:
            return []

        height, width = img.shape[:2]

        # 转换到HSV色彩空间
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 定义红色的HSV范围（红色在HSV中有两个范围）
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        # 创建红色掩码
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # 形态学操作，去除噪声
        kernel = np.ones((3, 3), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

        # 查找轮廓
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for contour in contours:
            # 获取边界框
            x, y, w, h = cv2.boundingRect(contour)

            # 过滤太小的框（可能是噪声）
            area = w * h
            if area < 2000 or w < 30 or h < 30:
                continue

            # 检查是否为近似矩形
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # 矩形应该有4个顶点左右
            if 4 <= len(approx) <= 8:
                boxes.append((x, y, x + w, y + h))

        return boxes

    def convert_red_boxes_to_yolo_labels(self, min_box_size=20, target_class_id=0):
        """
        将图像中的红色框框转换为YOLO格式标签
        :param min_box_size: 最小框尺寸（过滤噪声）
        :param target_class_id: 目标类别ID（红色框对应的类别）
        """
        print(f"开始从红色框框生成YOLO标签（目标类别ID: {target_class_id}）...")

        # 获取所有训练图像
        image_files = list(self.train_images_dir.glob('*.jpg')) + \
                      list(self.train_images_dir.glob('*.png'))

        converted_count = 0
        total_boxes = 0

        for img_path in image_files:
            # 读取图像获取尺寸
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            height, width = img.shape[:2]

            # 检测红色框框
            boxes = self.detect_red_boxes(img_path)

            if len(boxes) == 0:
                continue

            # 创建对应的标签文件
            label_path = self.train_labels_dir / f'{img_path.stem}.txt'

            with open(label_path, 'w') as f:
                for x1, y1, x2, y2 in boxes:
                    # 转换为YOLO格式（归一化的中心点坐标和宽高）
                    x_center = ((x1 + x2) / 2) / width
                    y_center = ((y1 + y2) / 2) / height
                    box_width = (x2 - x1) / width
                    box_height = (y2 - y1) / height

                    # 写入YOLO格式标签（class_id x_center y_center width height）
                    f.write(f"{target_class_id} {x_center} {y_center} {box_width} {box_height}\n")
                    total_boxes += 1

            converted_count += 1
            print(f"已处理: {converted_count}/{len(image_files)}")

        print(f"转换完成！共处理 {converted_count} 张图像，生成 {total_boxes} 个标签")
        print(f"标签保存在: {self.train_labels_dir}")

    def train_with_red_box_annotations(self, epochs=50, val_ratio=0.2, imgsz=640, batch=16, device='cpu',
                                       class_names=None, keep_base_classes=False):
        """
        使用红色框框标注的数据进行完整训练流程
        :param epochs: 训练轮数
        :param val_ratio: 验证集比例
        :param imgsz: 图像大小
        :param batch: 批次大小
        :param device: 设备 ('cpu' 或 'cuda')
        :param class_names: 类别名称列表，例如 ['person'] 或 ['enemy', 'ally']
        :param keep_base_classes: 是否保留基础模型的原有类别（默认False）
        """
        print("=" * 60)
        print("开始使用红色框框标注数据的训练流程")
        print("=" * 60)

        # 如果没有指定类别，使用默认的 person
        if class_names is None:
            class_names = ['person']

        print(f"\n新增训练类别: {class_names}")

        # 根据参数决定是否保留原有类别
        if keep_base_classes:
            # 保留原有类别并添加新类别
            merged_class_names = self.merge_class_names(class_names)
            new_class_start_id = len(self.base_model_classes)
            print(f"模式: 保留原有类别 + 新增类别")
            print(f"新类别起始ID: {new_class_start_id}")
        else:
            # 只使用新类别
            merged_class_names = class_names
            new_class_start_id = 0
            print(f"模式: 仅训练新类别")
            print(f"新类别ID: 0")

        # 步骤1: 从红色框框生成YOLO标签
        print("\n步骤 1/4: 从红色框框生成YOLO标签")
        self.convert_red_boxes_to_yolo_labels(target_class_id=new_class_start_id)

        # 步骤2: 分割数据集
        print("\n步骤 2/4: 分割数据集")
        self.split_dataset(val_ratio=val_ratio)

        # 步骤3: 创建数据集配置
        print("\n步骤 3/4: 创建数据集配置")
        self.create_dataset_yaml(class_names=merged_class_names)

        # 步骤4: 训练模型
        print("\n步骤 4/4: 训练模型")
        results = self.train(epochs=epochs, imgsz=imgsz, batch=batch, device=device, class_names=merged_class_names)

        print("\n" + "=" * 60)
        print("完整训练流程完成！")
        print("=" * 60)

        return results
