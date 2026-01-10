from service.YoloImageLableService import YoloImageLableService
from pathlib import Path
import os

if __name__ == '__main__':
    path = os.path.dirname(__file__) + '/training'
    project_name = 'warZ'  # 可以自定义为任意名称

    project_dir = Path(path + f'/{project_name}')
    dataset_dir = project_dir / 'dataset'
    images_dir = dataset_dir / 'images'
    labels_dir = dataset_dir / 'labels'

    # 创建训练集和验证集目录
    train_images_dir = images_dir / 'train'
    val_images_dir = images_dir / 'val'
    train_labels_dir = labels_dir / 'train'
    val_labels_dir = labels_dir / 'val'

    # 初始化
    images_label_service = YoloImageLableService(
        train_images_dir=train_images_dir,
        train_labels_dir=train_labels_dir,
        val_images_dir=val_images_dir,
        val_labels_dir=val_labels_dir)

    images_label_service.convert_red_boxes_to_yolo_labels(target_class_id=0)
