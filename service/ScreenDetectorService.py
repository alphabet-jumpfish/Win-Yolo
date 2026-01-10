import cv2
import numpy as np
from mss import mss
from ultralytics import YOLO
import time
import os
from pathlib import Path
import pyautogui
import ctypes


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

        # 循环锁定相关变量
        self.current_target_index = 0  # 当前锁定的目标索引
        self.lock_interval = 0.5  # 锁定间隔时间（秒）
        self.last_lock_time = 0  # 上次锁定时间

        # 自动开火相关变量
        self.auto_fire_enabled = False  # 自动开火开关（同时控制自动移动）
        self.fire_rate = 1  # 每秒开火次数（改为1次）
        self.last_fire_time = 0  # 上次开火时间
        self.is_locked_on_target = False  # 是否锁定在目标上
        self.mouse_in_target_range = False  # 鼠标是否在目标范围内
        self.target_range_threshold = 50  # 目标范围阈值（像素）

        # 禁用 pyautogui 的安全暂停
        pyautogui.PAUSE = 0
        pyautogui.FAILSAFE = False

    def mouse_click(self):
        """
        使用ctypes执行鼠标左键点击
        """
        try:
            # 定义鼠标事件常量
            MOUSEEVENTF_LEFTDOWN = 0x0002
            MOUSEEVENTF_LEFTUP = 0x0004

            # 按下左键
            ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            # 释放左键
            ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

            print("[鼠标点击] 使用ctypes执行点击")
        except Exception as e:
            print(f"[鼠标点击] 点击失败: {e}")

    def fire(self):
        """
        开火方法 - 执行鼠标点击
        """
        try:
            self.mouse_click()
            print("[开火] ✓ 开火！")
        except Exception as e:
            print(f"[开火] ✗ 开火失败: {e}")

    def capture_screen(self):
        """捕获屏幕截图"""
        screenshot = self.sct.grab(self.monitor)
        # 转换为numpy数组
        img = np.array(screenshot)
        # 转换颜色格式 BGRA -> BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def smooth_move_mouse(self, target_x, target_y, duration=0.3):
        """
        平滑移动鼠标到目标位置
        使用ctypes分5次移动，配合微秒级延迟
        :param target_x: 目标X坐标
        :param target_y: 目标Y坐标
        :param duration: 移动持续时间（秒）
        """
        try:
            # 获取当前鼠标位置
            current_x, current_y = pyautogui.position()

            # 计算移动距离
            delta_x = target_x - current_x
            delta_y = target_y - current_y

            # 分5步移动
            steps = 5
            step_delay = duration / steps  # 每步之间的延迟

            for i in range(1, steps + 1):
                # 计算当前步骤的目标位置
                step_x = int(current_x + (delta_x * i / steps))
                step_y = int(current_y + (delta_y * i / steps))

                # 使用ctypes设置鼠标位置
                ctypes.windll.user32.SetCursorPos(step_x, step_y)

                # 微秒级延迟
                time.sleep(step_delay)

            print(f"[鼠标移动检查] 使用ctypes分5步移动到 ({target_x}, {target_y})")

        except Exception as e:
            print(f"鼠标移动失败: {e}")

    def auto_fire(self):
        """
        自动开火功能
        只有在鼠标在目标范围内且自动开火开启时才开火
        """
        # 检查是否有目标
        if not self.is_locked_on_target:
            # print("[开火检查] 未检测到目标，开火未触发")
            return

        # 检查鼠标是否在目标范围内
        if not self.mouse_in_target_range:
            print("[开火检查] 鼠标不在目标范围内，开火未触发")
            return

        # 检查自动开火是否开启
        if not self.auto_fire_enabled:
            print("[开火检查] 鼠标在目标范围内，但自动开火未开启，开火未触发")
            return

        # 检查开火间隔
        current_time = time.time()
        fire_interval = 1.0 / self.fire_rate  # 计算开火间隔

        if current_time - self.last_fire_time >= fire_interval:
            # 调用公共的开火方法
            self.fire()
            self.last_fire_time = current_time

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
        检测目标并判断鼠标是否在目标范围内
        :param results: YOLO检测结果
        """
        # 没有检测到目标
        if len(results.boxes) == 0:
            self.is_locked_on_target = False
            self.mouse_in_target_range = False
            return

        # 获取屏幕中心点
        screen_center_x = self.monitor['width'] // 2
        screen_center_y = self.monitor['height'] // 2

        # 找到距离屏幕中心最近的目标
        closest_target = None
        closest_box = None
        min_distance = float('inf')

        for box in results.boxes:
            # 获取边界框坐标 (x1, y1, x2, y2)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            # 计算头部位置（水平居中，垂直位置在上部20%处）
            head_x = int((x1 + x2) / 2)
            head_y = int(y1 + (y2 - y1) * 0.2)

            # 计算到屏幕中心的距离
            distance = ((head_x - screen_center_x) ** 2 + (head_y - screen_center_y) ** 2) ** 0.5

            if distance < min_distance:
                min_distance = distance
                closest_target = (head_x, head_y)
                closest_box = (int(x1), int(y1), int(x2), int(y2))

        # 如果找到了最近的目标
        if closest_target is not None and closest_box is not None:
            target_x, target_y = closest_target
            box_x1, box_y1, box_x2, box_y2 = closest_box

            # 打印目标信息
            print(f"\n[目标检测] 检测到目标")
            print(f"  目标位置: ({target_x}, {target_y})")
            print(f"  目标框范围: ({box_x1}, {box_y1}) -> ({box_x2}, {box_y2})")

            # 获取当前鼠标位置
            current_mouse_x, current_mouse_y = pyautogui.position()
            print(f"  当前鼠标位置: ({current_mouse_x}, {current_mouse_y})")

            # 判断鼠标是否在目标检测框范围内
            # 使用扩展的范围（目标框 + 阈值）
            extended_x1 = box_x1 - self.target_range_threshold
            extended_y1 = box_y1 - self.target_range_threshold
            extended_x2 = box_x2 + self.target_range_threshold
            extended_y2 = box_y2 + self.target_range_threshold

            if (extended_x1 <= current_mouse_x <= extended_x2 and
                extended_y1 <= current_mouse_y <= extended_y2):
                self.mouse_in_target_range = True
                print(f"  [判断] ✓ 鼠标在目标范围内")
            else:
                self.mouse_in_target_range = False
                print(f"  [判断] ✗ 鼠标不在目标范围内")

                # 检查自动移动是否开启（由K键控制）
                if self.auto_fire_enabled:
                    print(f"  [自动移动] 正在移动鼠标到目标位置...")

                    # 自动移动鼠标到目标位置
                    self.smooth_move_mouse(target_x, target_y, duration=0.2)

                    # 移动后重新检查
                    current_mouse_x, current_mouse_y = pyautogui.position()
                    if (extended_x1 <= current_mouse_x <= extended_x2 and
                        extended_y1 <= current_mouse_y <= extended_y2):
                        self.mouse_in_target_range = True
                        print(f"  [自动移动] ✓ 鼠标已移动到目标范围内")
                    else:
                        print(f"  [自动移动] ✗ 移动后仍不在目标范围内")
                else:
                    print(f"  [自动移动] 自动移动未开启（按K键开启）")

            self.is_locked_on_target = True

        else:
            self.is_locked_on_target = False
            self.mouse_in_target_range = False

    def run(self, conf_threshold=0.5, display_scale=0.6):
        """
        运行实时屏幕检测
        :param conf_threshold: 检测置信度阈值
        :param display_scale: 显示窗口缩放比例（避免窗口过大）
        """
        print("开始实时人物检测...")
        print("按 'q' 键暂停/恢复程序")
        print("按 'k' 键开启/关闭自动瞄准+自动开火")
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

                # 按 'k' 键切换自动瞄准+自动开火
                elif key == ord('k'):
                    self.auto_fire_enabled = not self.auto_fire_enabled
                    if self.auto_fire_enabled:
                        print("✓ 自动瞄准+自动开火已开启 [每秒1次]")
                    else:
                        print("✗ 自动瞄准+自动开火已关闭")

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

                # 自动开火（如果已开启）
                self.auto_fire()

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

                # 显示自动瞄准+开火状态
                fire_status = "ON" if self.auto_fire_enabled else "OFF"
                fire_color = (0, 255, 0) if self.auto_fire_enabled else (0, 0, 255)
                cv2.putText(annotated_img, f'Auto Aim+Fire [K]: {fire_status}', (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, fire_color, 2)

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
