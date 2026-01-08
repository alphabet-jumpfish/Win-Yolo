"""
双模型检测示例
同时使用预训练模型和自定义模型进行检测
"""
from service.DualModelDetector import DualModelDetector
import os


def main():
    print("=" * 60)
    print("YOLO 双模型实时检测程序")
    print("=" * 60)

    # 配置模型路径
    base_model_path = os.path.dirname(__file__) + '/models/yolov8n.pt'
    custom_model_path = os.path.dirname(__file__) + '/training/warZ/runs/train/weights/best.pt'

    print(f"\n基础模型: {base_model_path}")
    print(f"自定义模型: {custom_model_path}")

    # 配置检测类别
    # 基础模型检测的类别（COCO 80类）
    base_classes = [0]  # 只检测 person
    # base_classes = None  # 检测所有80个类别

    # 自定义模型检测的类别
    custom_classes = [0]  # 检测 warZ（ID 0）
    # custom_classes = None  # 检测所有自定义类别

    print(f"\n基础模型检测类别: {base_classes}")
    print(f"自定义模型检测类别: {custom_classes}")

    # 创建双模型检测器
    detector = DualModelDetector(
        base_model_path=base_model_path,
        custom_model_path=custom_model_path
    )

    # 运行检测
    detector.run(
        base_conf=0.5,
        custom_conf=0.5,
        base_classes=base_classes,
        custom_classes=custom_classes,
        display_scale=0.6
    )


if __name__ == '__main__':
    main()
