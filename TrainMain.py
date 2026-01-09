from service.YoloTrainerService import YOLOTrainer
import os


def main():
    """
    训练示例
    """
    print("=" * 60)
    print("YOLO 自定义目标检测模型训练")
    print("=" * 60)

    # 配置训练类别
    # 单类别训练：['person']
    # 多类别训练：['person', 'car', 'dog']
    # 自定义类别：['enemy', 'ally', 'item']
    class_names = ['warZ']  # 自定义训练类别

    # 自定义项目名称（训练结果会保存在 training/{project_name} 目录下）
    project_name = 'warZ'  # 可以自定义为任意名称

    # 是否保留基础模型的原有类别
    # False: 只训练新类别（推荐，新类别ID从0开始）
    # True: 保留原有80个类别 + 新类别（需要COCO完整数据集，不推荐）
    keep_base_classes = False

    print(f"\n项目名称: {project_name}")
    print(f"训练类别: {class_names}")
    print(f"类别数量: {len(class_names)}")
    print(f"保留原有类别: {keep_base_classes}")

    # 步骤1: 创建训练器
    print("\n创建训练器...")
    path = os.path.dirname(__file__) + '/training'
    base_model_path = os.path.dirname(__file__) + '/models/'
    trainer = YOLOTrainer(
        project_name=project_name,  # 使用自定义项目名称
        base_model=base_model_path + 'yolo11n.pt',
        path=path
    )

    print(f"\n训练项目目录: {trainer.project_dir}")
    print(f"训练图像目录: {trainer.train_images_dir}")
    print(f"训练标签目录: {trainer.train_labels_dir}")

    # 步骤2: 准备训练数据
    print("\n" + "=" * 60)
    print("准备训练数据")
    print("=" * 60)
    print("\n请将带有红色框框标注的图像放入以下目录：")
    print(f"  {trainer.train_images_dir}")
    print("\n红色框框标注说明：")
    print("  - 在图像中用红色矩形框标注出目标区域")
    print("  - 红色框会被自动检测并转换为YOLO格式标签")
    print("  - 支持 .jpg 和 .png 格式")
    print(f"  - 当前训练类别: {class_names}")
    if len(class_names) > 1:
        print("  - 多类别训练：所有红色框都会被标注为第一个类别")
        print("  - 如需区分不同类别，请使用不同颜色的框或分别训练")

    input("\n按 Enter 键继续（确保已放入标注好的图像）...")

    # 步骤3: 开始训练
    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)

    # 使用红色框框标注数据进行训练
    results = trainer.train_with_red_box_annotations(
        epochs=100,  # 训练轮数
        val_ratio=0.2,  # 验证集比例（20%）
        imgsz=640,  # 图像大小
        batch=16,  # 批次大小
        device='cuda',  # 使用CPU训练（如果有GPU，改为'cuda'）
        class_names=class_names,  # 传入自定义类别
        keep_base_classes=keep_base_classes  # 是否保留原有类别
    )

    # 步骤4: 训练完成
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    print(f"\n训练好的模型保存在: {trainer.project_dir / 'runs' / 'train' / 'weights'}")
    print("  - best.pt: 最佳模型")
    print("  - last.pt: 最后一轮模型")

    print("\n使用训练好的模型：")
    print("  1. 在 main.py 中修改 model_name 为训练好的模型路径")
    print("  2. 例如: model_name = 'training/my_person_detection/runs/train/weights/best.pt'")


if __name__ == '__main__':
    main()
