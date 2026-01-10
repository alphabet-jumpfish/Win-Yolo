"""
YOLO 自定义模型训练类
用于训练专门识别特定人物的精准模型
"""
from pathlib import Path
from ultralytics import YOLO
import yaml

from service.YoloImageLableService import YoloImageLableService


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

        self.images_label_service = YoloImageLableService(
            train_images_dir=self.train_images_dir,
            train_labels_dir=self.train_labels_dir,
            val_images_dir=self.val_images_dir,
            val_labels_dir=self.val_labels_dir)

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

    def train_with_red_box_annotations(self, epochs=50, val_ratio=0.2, imgsz=640, batch=16, device='cpu',
                                       class_names=None, keep_base_classes=False, label_img_target=True):
        """
        使用红色框框标注的数据进行完整训练流程
        :param epochs: 训练轮数
        :param val_ratio: 验证集比例
        :param imgsz: 图像大小
        :param batch: 批次大小
        :param device: 设备 ('cpu' 或 'cuda')
        :param class_names: 类别名称列表，例如 ['person'] 或 ['enemy', 'ally']
        :param keep_base_classes: 是否保留基础模型的原有类别（默认False）
        :param label_img_target : false 使用自有方法 true 使用labelImg工具生成
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
        if label_img_target:
            input("\n按 Enter 键继续（确保已放入通过labelImg工具完成标注好的图像）...")
        else:
            print("\n步骤 1/4: images_label_service 红色框框生成YOLO标签")
            self.images_label_service.convert_red_boxes_to_yolo_labels(target_class_id=new_class_start_id)

        # 步骤2: 分割数据集
        print("\n步骤 2/4: 分割数据集")
        self.images_label_service.split_dataset(val_ratio=val_ratio)

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
