"""
YOLO 模型类别查询工具
用于查询 .pt 模型文件中的类别信息
"""
from ultralytics import YOLO
from pathlib import Path


class ModelClassInspector:
    """
    YOLO 模型类别检查器
    用于查询模型中的类别名称和ID
    """

    def __init__(self, model_path):
        """
        初始化检查器
        :param model_path: 模型文件路径（.pt文件）
        """
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        print(f"正在加载模型: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        print("模型加载完成！")

    def get_class_names(self):
        """
        获取模型的所有类别名称
        :return: 类别名称字典 {id: name}
        """
        return self.model.names

    def get_num_classes(self):
        """
        获取模型的类别数量
        :return: 类别数量
        """
        return len(self.model.names)

    def get_class_id_by_name(self, class_name):
        """
        根据类别名称获取类别ID
        :param class_name: 类别名称
        :return: 类别ID，如果不存在返回None
        """
        for class_id, name in self.model.names.items():
            if name.lower() == class_name.lower():
                return class_id
        return None

    def get_class_name_by_id(self, class_id):
        """
        根据类别ID获取类别名称
        :param class_id: 类别ID
        :return: 类别名称，如果不存在返回None
        """
        return self.model.names.get(class_id, None)

    def search_classes(self, keyword):
        """
        搜索包含关键词的类别
        :param keyword: 搜索关键词
        :return: 匹配的类别列表 [(id, name), ...]
        """
        results = []
        for class_id, name in self.model.names.items():
            if keyword.lower() in name.lower():
                results.append((class_id, name))
        return results

    def print_all_classes(self):
        """
        打印所有类别信息
        """
        print("\n" + "=" * 60)
        print(f"模型: {self.model_path.name}")
        print(f"类别总数: {self.get_num_classes()}")
        print("=" * 60)
        print(f"{'ID':<5} {'类别名称'}")
        print("-" * 60)

        for class_id, name in sorted(self.model.names.items()):
            print(f"{class_id:<5} {name}")

        print("=" * 60)

    def get_model_info(self):
        """
        获取模型的详细信息
        :return: 模型信息字典
        """
        info = {
            'model_path': str(self.model_path),
            'model_name': self.model_path.name,
            'num_classes': self.get_num_classes(),
            'class_names': self.model.names
        }
        return info
