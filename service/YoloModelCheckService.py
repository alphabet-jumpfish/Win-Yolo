from ultralytics import YOLO
import cv2
import os
from pathlib import Path
import torch


class YoloModelCheck:
    """
    YOLO 模型检查类
    用于检查训练好的模型是否能正确检测目标
    """

    def show_model_metrics_from_pt(self, model_path):
        """
        直接从 .pt 模型文件中读取训练指标
        :param model_path: 模型路径，如 'models/yolov8n.pt' 或 'training/warZ/runs/train/weights/best.pt'
        """
        print("=" * 60)
        print("模型训练指标分析（从.pt文件读取）")
        print("=" * 60)

        if not os.path.exists(model_path):
            print(f"\n错误：模型文件不存在: {model_path}")
            return

        try:
            # 加载模型
            model = YOLO(model_path)

            # 尝试从模型中获取训练器信息
            # PyTorch 2.6+ 需要设置 weights_only=False
            checkpoint = torch.load(model_path, map_location='cuda', weights_only=False)

            print(f"\n模型文件: {model_path}")
            print(f"模型类别: {model.names}")
            print(f"类别数量: {len(model.names)}")

            # 检查是否有训练指标
            if 'train_metrics' in checkpoint:
                metrics = checkpoint['train_metrics']
                print("\n最终训练指标:")
                print("-" * 60)

                # 提取关键指标
                precision = metrics.get('metrics/precision(B)', 0)
                recall = metrics.get('metrics/recall(B)', 0)
                mAP50 = metrics.get('metrics/mAP50(B)', 0)
                mAP50_95 = metrics.get('metrics/mAP50-95(B)', 0)

                print(f"Precision (精确度):  {precision:.5f} ({precision*100:.2f}%)")
                print(f"Recall (召回率):     {recall:.5f} ({recall*100:.2f}%)")
                print(f"mAP50:               {mAP50:.5f} ({mAP50*100:.2f}%)")
                print(f"mAP50-95:            {mAP50_95:.5f} ({mAP50_95*100:.2f}%)")

                print("\n" + "-" * 60)
                print("指标说明:")
                print("  Precision: 检测出的目标中有多少是正确的")
                print("  Recall:    实际目标中有多少被检测出来")
                print("  mAP50:     在IoU=0.5时的平均精度")
                print("  mAP50-95:  在IoU=0.5-0.95时的平均精度")
                print("\n" + "-" * 60)
                print("质量评估:")
                if precision < 0.3:
                    print("  Precision 过低 (<30%) - 模型误检太多")
                elif precision < 0.6:
                    print("  Precision 一般 (30-60%) - 需要改进")
                else:
                    print("  Precision 良好 (>60%)")
                if recall < 0.3:
                    print("  Recall 过低 (<30%) - 模型漏检太多")
                elif recall < 0.6:
                    print("  Recall 一般 (30-60%) - 需要改进")
                else:
                    print("  Recall 良好 (>60%)")
                if mAP50 < 0.3:
                    print("  mAP50 过低 (<30%) - 模型效果很差")
                elif mAP50 < 0.6:
                    print("  mAP50 一般 (30-60%) - 需要改进")
                else:
                    print("  mAP50 良好 (>60%)")

            elif 'best_fitness' in checkpoint:
                # 尝试从 best_fitness 获取
                print("\n模型包含 best_fitness 指标")
                print(f"Best Fitness: {checkpoint['best_fitness']}")

            else:
                print("\n该模型文件中没有保存训练指标")
                print("这可能是预训练模型或训练时未保存指标")

        except Exception as e:
            print(f"\n读取模型文件时出错: {e}")

        print("\n" + "=" * 60)

    def show_training_metrics(self, model_path):
        """
        显示模型的训练指标
        :param model_path: 模型路径，如 'training/warZ/runs/train/weights/best.pt'
        """
        print("=" * 60)
        print("模型训练指标分析")
        print("=" * 60)

        # 从模型路径推断训练结果目录
        model_path = Path(model_path)
        train_dir = model_path.parent.parent  # 从 weights/best.pt 回到 train 目录

        results_csv = train_dir / 'results.csv'

        if not results_csv.exists():
            print(f"\n错误：找不到训练结果文件: {results_csv}")
            return

        # 读取 CSV 文件
        with open(results_csv, 'r') as f:
            lines = f.readlines()

        if len(lines) < 2:
            print("\n错误：训练结果文件为空")
            return

        # 解析表头
        headers = [h.strip() for h in lines[0].split(',')]
        # 解析最后一行数据
        last_line = lines[-1].strip().split(',')

        # 创建字典
        metrics = {}
        for i, header in enumerate(headers):
            if i < len(last_line):
                try:
                    metrics[header] = float(last_line[i])
                except:
                    metrics[header] = last_line[i]

        print(f"\n训练轮数: {int(metrics.get('epoch', 0)) + 1}")
        print("\n最终训练指标:")
        print("-" * 60)

        # 显示关键指标
        precision = metrics.get('metrics/precision(B)', 0)
        recall = metrics.get('metrics/recall(B)', 0)
        mAP50 = metrics.get('metrics/mAP50(B)', 0)
        mAP50_95 = metrics.get('metrics/mAP50-95(B)', 0)

        print(f"Precision (精确度):  {precision:.5f} ({precision*100:.2f}%)")
        print(f"Recall (召回率):     {recall:.5f} ({recall*100:.2f}%)")
        print(f"mAP50:               {mAP50:.5f} ({mAP50*100:.2f}%)")
        print(f"mAP50-95:            {mAP50_95:.5f} ({mAP50_95*100:.2f}%)")

        print("\n" + "-" * 60)
        print("指标说明:")
        print("  Precision: 检测出的目标中有多少是正确的")
        print("  Recall:    实际目标中有多少被检测出来")
        print("  mAP50:     在IoU=0.5时的平均精度")
        print("  mAP50-95:  在IoU=0.5-0.95时的平均精度")

        print("\n" + "-" * 60)
        print("质量评估:")

        if precision < 0.3:
            print("  Precision 过低 (<30%) - 模型误检太多")
        elif precision < 0.6:
            print("  Precision 一般 (30-60%) - 需要改进")
        else:
            print("  Precision 良好 (>60%)")

        if recall < 0.3:
            print("  Recall 过低 (<30%) - 模型漏检太多")
        elif recall < 0.6:
            print("  Recall 一般 (30-60%) - 需要改进")
        else:
            print("  Recall 良好 (>60%)")

        if mAP50 < 0.3:
            print("  mAP50 过低 (<30%) - 模型效果很差")
        elif mAP50 < 0.6:
            print("  mAP50 一般 (30-60%) - 需要改进")
        else:
            print("  mAP50 良好 (>60%)")

        print("\n" + "=" * 60)

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
