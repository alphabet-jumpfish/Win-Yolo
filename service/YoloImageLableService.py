import cv2
import numpy as np
import shutil


class YoloImageLableService:

    def __init__(self, train_images_dir, train_labels_dir, val_images_dir, val_labels_dir):
        """
        初始化YoloImageLableService对象
        :param train_images_dir: 训练图像目录
        :param train_labels_dir: 训练标签目录
        """
        self.train_images_dir = train_images_dir
        self.train_labels_dir = train_labels_dir
        self.val_images_dir = val_images_dir
        self.val_labels_dir = val_labels_dir

    @staticmethod
    def imread_chinese(image_path):
        """
        读取包含中文路径的图片
        :param image_path: 图片路径
        :return: 图片数组，如果读取失败返回None
        """
        try:
            # 使用numpy读取文件，避免中文路径问题
            with open(str(image_path), 'rb') as f:
                data = np.frombuffer(f.read(), dtype=np.uint8)
                img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                return img
        except Exception as e:
            print(f"读取图片失败: {image_path}, 错误: {e}")
            return None

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
            img = self.imread_chinese(img_path)
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

    def detect_red_boxes(self, image_path):
        """
        检测图像中的红色框框（空心矩形框）
        :param image_path: 图像路径
        :return: 检测到的边界框列表 [(x1, y1, x2, y2), ...]
        """
        # 读取图像
        img = self.imread_chinese(image_path)
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
