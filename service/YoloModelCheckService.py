from ultralytics import YOLO
import cv2
import os


class YoloModelCheck:
    """
    YOLO 模型检查类
    用于检查训练好的模型是否能正确检测目标
    """

    def model_detect_pictures(self, model_path, test_image_dir):
        print("=" * 60)
        print("模型测试诊断程序")
        print("=" * 60)

        if not os.path.exists(model_path):
            print(f"\n错误：模型文件不存在: {model_path}")
            return

        print(f"\n正在加载模型: {model_path}")
        model = YOLO(model_path)

        # 显示模型信息
        print(f"\n模型类别: {model.names}")
        print(f"类别数量: {len(model.names)}")

        if not os.path.exists(test_image_dir):
            print(f"\n错误：测试图像目录不存在: {test_image_dir}")
            return

        # 获取测试图像
        image_files = []
        for ext in ['*.jpg', '*.png', '*.jpeg']:
            import glob
            image_files.extend(glob.glob(os.path.join(test_image_dir, ext)))

        if not image_files:
            print(f"\n错误：在 {test_image_dir} 中没有找到图像文件")
            return

        print(f"\n找到 {len(image_files)} 张测试图像")
        print("\n开始测试检测...")

        # 测试前3张图像
        for i, img_path in enumerate(image_files[:3]):
            print(f"\n测试图像 {i + 1}: {os.path.basename(img_path)}")

            # 进行检测
            results = model(img_path, conf=0.1, verbose=False)  # 降低置信度阈值

            detections = len(results[0].boxes)
            print(f"  检测到 {detections} 个目标")

            if detections > 0:
                for box in results[0].boxes:
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    class_name = model.names[cls]
                    print(f"    - {class_name}: 置信度 {conf:.3f}")
            else:
                print("    - 未检测到任何目标")

        print("\n" + "=" * 60)
        print("测试完成")
        print("=" * 60)
