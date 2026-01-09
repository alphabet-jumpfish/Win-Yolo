
import os

def main():
    """主函数"""
    print("=" * 50)
    print("YOLO 实时目标检测程序")
    print("=" * 50)

    # ========== 模型配置 ==========
    # 选项1: 使用预训练模型（COCO数据集，80个类别）
    model_name = 'best.pt'

    # ========== 检测类别配置 ==========
    # 预训练模型（COCO数据集）：
    #   None: 检测所有类别
    #   [0]: 只检测 person（人物）
    #   [0, 2, 3]: 检测多个类别（person, car, motorcycle）

    # 自定义训练模型：
    #   None: 检测所有训练的类别
    #   [0]: 只检测第一个类别（如训练时的 'ally'）
    #   [0, 1]: 检测前两个类别（如 'ally', 'enemy'）

    detect_classes = [0]  # 默认只检测第一个类别

    # 初始化检测器
    from service.ScreenDetectorService import ScreenDetector
    model_dirs = os.path.dirname(__file__) + '/models'
    detector = ScreenDetector(
        model_name=model_name,
        model_dir=model_dirs,
        detect_classes=detect_classes  # 传入要检测的类别
    )
    # 运行检测
    # conf_threshold: 置信度阈值 (0.0-1.0)，越高越严格
    # display_scale: 显示窗口缩放比例，避免窗口过大
    detector.run(conf_threshold=0.5, display_scale=0.6)


# COCO 数据集类别列表（YOLO 预训练模型支持的 80 个类别）
# 使用方法：detect_classes = [类别ID]
# 例如：detect_classes = [0, 2, 3] 表示检测 person, car, motorcycle
"""
常用类别：
0: person (人)
1: bicycle (自行车)
2: car (汽车)
3: motorcycle (摩托车)
5: bus (公交车)
7: truck (卡车)
16: dog (狗)
17: cat (猫)
"""


if __name__ == '__main__':
    main()
