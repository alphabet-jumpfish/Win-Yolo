"""
双模型检测器
同时使用预训练模型和自定义模型进行检测
"""
import cv2
import numpy as np
from mss import mss
from ultralytics import YOLO
import time
import os
from pathlib import Path
import pyautogui


class DualModelDetector:
    """
    双模型检测器
    同时使用两个YOLO模型进行检测
    """

    def __init__(self, base_model_path, custom_model_path, model_dir='./models'):
        """
        初始化双模型检测器
        :param base_model_path: 基础模型路径（如 yolov8n.pt）
        :param custom_model_path: 自定义模型路径（如 best.pt）
        :param model_dir: 模型目录
        """
        print("正在加载双模型检测器...")

        # 加载基础模型（COCO 80类）
        print(f"加载基础模型: {base_model_path}")
        self.base_model = YOLO(base_model_path)
        self.base_model_classes = self.base_model.names
        print(f"基础模型类别数: {len(self.base_model_classes)}")

        # 加载自定义模型
        print(f"加载自定义模型: {custom_model_path}")
        self.custom_model = YOLO(custom_model_path)
        self.custom_model_classes = self.custom_model.names
        print(f"自定义模型类别数: {len(self.custom_model_classes)}")

        # 屏幕捕获
        self.sct = mss()
        self.monitor = self.sct.monitors[1]

        # 鼠标控制
        self.processed_persons = set()
        pyautogui.PAUSE = 0
        pyautogui.FAILSAFE = False

        print("双模型检测器加载完成！")

    def capture_screen(self):
        """捕获屏幕截图"""
        screenshot = self.sct.grab(self.monitor)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def detect_dual_models(self, img, base_conf=0.5, custom_conf=0.5, base_classes=None, custom_classes=None):
        """
        使用双模型进行检测
        :param img: 输入图像
        :param base_conf: 基础模型置信度阈值
        :param custom_conf: 自定义模型置信度阈值
        :param base_classes: 基础模型要检测的类别列表
        :param custom_classes: 自定义模型要检测的类别列表
        :return: 合并后的检测结果
        """
        # 使用基础模型检测
        if base_classes is None:
            base_results = self.base_model(img, conf=base_conf, verbose=False)
        else:
            base_results = self.base_model(img, conf=base_conf, classes=base_classes, verbose=False)

        # 使用自定义模型检测
        if custom_classes is None:
            custom_results = self.custom_model(img, conf=custom_conf, verbose=False)
        else:
            custom_results = self.custom_model(img, conf=custom_conf, classes=custom_classes, verbose=False)

        return base_results[0], custom_results[0]

    def draw_detections(self, img, base_results, custom_results):
        """
        在图像上绘制双模型的检测结果
        :param img: 原始图像
        :param base_results: 基础模型检测结果
        :param custom_results: 自定义模型检测结果
        :return: 标注后的图像
        """
        annotated_img = img.copy()

        # 绘制基础模型的检测结果（蓝色框）
        for box in base_results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            label = f"{self.base_model_classes[cls]} {conf:.2f}"

            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(annotated_img, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # 绘制自定义模型的检测结果（红色框）
        for box in custom_results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            label = f"{self.custom_model_classes[cls]} {conf:.2f}"

            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(annotated_img, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return annotated_img

    def move_mouse_to_target(self, base_results, custom_results):
        """
        将鼠标移动到检测目标的头部位置
        :param base_results: 基础模型检测结果
        :param custom_results: 自定义模型检测结果
        """
        # 优先移动到自定义模型检测的目标
        for box in custom_results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            head_x = int((x1 + x2) / 2)
            head_y = int(y1 + (y2 - y1) * 0.2)

            person_id = (head_x // 100, head_y // 100)
            if person_id not in self.processed_persons:
                pyautogui.moveTo(head_x, head_y, duration=0.2)
                self.processed_persons.add(person_id)
                return

    def run(self, base_conf=0.5, custom_conf=0.5, base_classes=None, custom_classes=None, display_scale=0.6):
        """
        运行双模型实时检测
        :param base_conf: 基础模型置信度阈值
        :param custom_conf: 自定义模型置信度阈值
        :param base_classes: 基础模型要检测的类别
        :param custom_classes: 自定义模型要检测的类别
        :param display_scale: 显示窗口缩放比例
        """
        print("开始双模型实时检测...")
        print("按 'q' 键暂停/恢复程序")
        print("按 'ESC' 键退出程序")

        fps_time = time.time()
        fps_counter = 0
        fps = 0
        is_paused = False
        last_frame = None

        try:
            while True:
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    is_paused = not is_paused
                    print("程序已暂停" if is_paused else "程序已恢复")
                elif key == 27:
                    break

                if is_paused:
                    if last_frame is not None:
                        cv2.imshow('Dual Model Detection - PAUSED', last_frame)
                    continue

                # 捕获屏幕
                screen = self.capture_screen()

                # 双模型检测
                base_results, custom_results = self.detect_dual_models(
                    screen, base_conf, custom_conf, base_classes, custom_classes
                )

                # 移动鼠标到目标
                self.move_mouse_to_target(base_results, custom_results)

                # 计算FPS
                fps_counter += 1
                if time.time() - fps_time > 1:
                    fps = fps_counter
                    fps_counter = 0
                    fps_time = time.time()

                # 绘制检测结果
                annotated_img = self.draw_detections(screen, base_results, custom_results)

                # 显示统计信息
                base_count = len(base_results.boxes)
                custom_count = len(custom_results.boxes)
                cv2.putText(annotated_img, f'FPS: {fps}', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(annotated_img, f'Base: {base_count}', (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(annotated_img, f'Custom: {custom_count}', (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # 缩放显示
                height, width = annotated_img.shape[:2]
                new_width = int(width * display_scale)
                new_height = int(height * display_scale)
                display_img = cv2.resize(annotated_img, (new_width, new_height))

                last_frame = display_img.copy()
                cv2.imshow('Dual Model Detection - RUNNING', display_img)

        except KeyboardInterrupt:
            print("\n程序被用户中断")
        except Exception as e:
            print(f"发生错误: {e}")
        finally:
            cv2.destroyAllWindows()
            print("程序已退出")
