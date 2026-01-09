import cv2
import numpy as np
from mss import mss
from ultralytics import YOLO
import time
import os
from pathlib import Path
import pyautogui


class ScreenDetector:

    def __init__(self, model_name='yolov8n.pt', model_dir='./models', detect_classes=None):
        """
        初始化屏幕检测器
        :param model_name: YOLO模型名称，默认使用yolov8n（最快的模型）
        :param model_dir: 模型保存目录，默认为 ./models
        :param detect_classes: 要检测的类别列表，None表示检测所有类别，[0]表示只检测person
        """
        # 创建模型目录
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # 设置模型路径
        model_path = self.model_dir / model_name

        print(f"模型保存路径: {model_path.absolute()}")
        print("正在加载YOLO模型...")

        # 如果模型文件已存在，直接加载；否则会自动下载到指定路径
        self.model = YOLO(str(model_path))
        print(f"模型 {model_name} 加载完成！")

        # 保存要检测的类别
        self.detect_classes = detect_classes
        if detect_classes is None:
            print("检测模式: 检测所有类别")
        else:
            print(f"检测模式: 只检测类别 {detect_classes}")

        self.sct = mss()
        # 获取主显示器的尺寸
        self.monitor = self.sct.monitors[1]

        # 用于跟踪已移动鼠标的人物位置，避免重复移动
        self.processed_persons = set()

        # 禁用 pyautogui 的安全暂停
        pyautogui.PAUSE = 0
        pyautogui.FAILSAFE = False

    def capture_screen(self):
        """捕获屏幕截图"""
        screenshot = self.sct.grab(self.monitor)
        # 转换为numpy数组
        img = np.array(screenshot)
        # 转换颜色格式 BGRA -> BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def detect_objects(self, img, conf_threshold=0.5):
        """
        使用YOLO检测图像中的目标
        :param img: 输入图像
        :param conf_threshold: 置信度阈值
        :return: 带有检测框的图像
        """
        # 使用配置的类别进行检测
        if self.detect_classes is None:
            # 检测所有类别
            results = self.model(img, conf=conf_threshold, verbose=False)
        else:
            # 只检测指定类别
            results = self.model(img, conf=conf_threshold, classes=self.detect_classes, verbose=False)

        annotated_img = results[0].plot()
        return annotated_img, results[0]

    def move_mouse_to_person(self, results):
        """
        将鼠标移动到检测到的人物头部位置
        每个人物只移动一次
        :param results: YOLO检测结果
        """
        if len(results.boxes) == 0:
            return

        for box in results.boxes:
            # 获取边界框坐标 (x1, y1, x2, y2)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            # 计算头部位置（水平居中，垂直位置在上部20%处）
            head_x = int((x1 + x2) / 2)
            head_y = int(y1 + (y2 - y1) * 0.2)

            # 创建唯一标识符（基于位置的粗略区域）
            # 使用100像素的网格来判断是否是同一个人物
            person_id = (head_x // 100, head_y // 100)

            # 如果这个人物还没有被处理过，移动鼠标
            if person_id not in self.processed_persons:
                # 获取屏幕尺寸并限制坐标范围
                screen_width = self.monitor['width']
                screen_height = self.monitor['height']

                # 确保坐标在屏幕范围内
                head_x = max(0, min(head_x, screen_width - 1))
                head_y = max(0, min(head_y, screen_height - 1))

                try:
                    pyautogui.moveTo(head_x, head_y, duration=0.2)
                    self.processed_persons.add(person_id)
                    print(f"鼠标移动到人物头部位置: ({head_x}, {head_y})")
                except Exception as e:
                    print(f"鼠标移动失败: {e}")
                break  # 每次只移动到一个新人物

    def run(self, conf_threshold=0.5, display_scale=0.6):
        """
        运行实时屏幕检测
        :param conf_threshold: 检测置信度阈值
        :param display_scale: 显示窗口缩放比例（避免窗口过大）
        """
        print("开始实时人物检测...")
        print("按 'q' 键暂停/恢复程序")
        print("按 'ESC' 键退出程序")

        fps_time = time.time()
        fps_counter = 0
        fps = 0
        is_paused = False  # 暂停状态标志
        last_frame = None  # 保存最后一帧用于暂停时显示

        try:
            while True:
                # 处理键盘输入
                key = cv2.waitKey(1) & 0xFF

                # 按 'q' 键切换暂停/恢复状态
                if key == ord('q'):
                    is_paused = not is_paused
                    if is_paused:
                        print("程序已暂停，按 'q' 键恢复")
                    else:
                        print("程序已恢复")

                # 按 ESC 键退出
                elif key == 27:  # ESC键
                    break

                # 如果暂停，显示最后一帧并跳过检测
                if is_paused:
                    if last_frame is not None:
                        status = "PAUSED"
                        cv2.imshow(f'Person Detection - {status} (Q:Pause/Resume, ESC:Quit)', last_frame)
                    continue

                # 捕获屏幕
                screen = self.capture_screen()

                # 进行目标检测
                annotated_img, results = self.detect_objects(screen, conf_threshold)

                # 将鼠标移动到检测到的新人物位置
                self.move_mouse_to_person(results)

                # 计算FPS
                fps_counter += 1
                if time.time() - fps_time > 1:
                    fps = fps_counter
                    fps_counter = 0
                    fps_time = time.time()

                # 在图像上显示FPS和检测到的目标数量
                num_detections = len(results.boxes)

                # 获取检测到的类别名称（如果有检测结果）
                if num_detections > 0:
                    # 获取第一个检测到的类别ID
                    first_class_id = int(results.boxes[0].cls[0].cpu().numpy())
                    detection_label = self.model.names[first_class_id]
                else:
                    detection_label = "Objects"

                cv2.putText(annotated_img, f'FPS: {fps}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(annotated_img, f'{detection_label}: {num_detections}', (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # 缩放图像以适应显示
                height, width = annotated_img.shape[:2]
                new_width = int(width * display_scale)
                new_height = int(height * display_scale)
                display_img = cv2.resize(annotated_img, (new_width, new_height))

                # 保存当前帧
                last_frame = display_img.copy()

                # 显示结果
                status = "RUNNING"
                cv2.imshow(f'Person Detection - {status} (Q:Pause/Resume, ESC:Quit)', display_img)

        except KeyboardInterrupt:
            print("\n程序被用户中断")
        except Exception as e:
            print(f"发生错误: {e}")
        finally:
            cv2.destroyAllWindows()
            print("程序已退出")
