"""
游戏灵敏度校准脚本 - 基于FOV的角度校准版本
使用倒计时代替按键监听，适用于全屏游戏

校准原理（基于视场角）：
1. 通过FOV将屏幕像素距离转换为视角角度
2. 在游戏中找一个明显的参考点（如建筑物边缘、门框等）
3. 将准星对准参考点
4. 倒计时后脚本自动移动鼠标
5. 你手动移动鼠标，将准星重新对准参考点
6. 脚本计算角度偏移，得出灵敏度系数（与距离无关）

公式说明：
- 焦距系数: f = (W/2) / tan(HFOV/2)
- 偏差角度: θ = arctan(Δpixel / f)
- 鼠标位移: dx = θ / (m_yaw × Sensitivity)

使用方法：
1. 启动游戏并进入场景
2. 运行脚本，按照倒计时提示操作
3. 无需按键，只需在倒计时结束前完成操作
"""

import ctypes
import time
import pyautogui
import json
import math
from pathlib import Path


class SensitivityCalibrator:
    def __init__(self, config_file_path, fov=90, m_yaw=0.022, previous_results=None):
        # 禁用 pyautogui 的安全暂停
        pyautogui.PAUSE = 0
        pyautogui.FAILSAFE = False

        # 获取屏幕尺寸
        self.screen_width = ctypes.windll.user32.GetSystemMetrics(0)
        self.screen_height = ctypes.windll.user32.GetSystemMetrics(1)

        # FOV参数（视场角）
        self.fov = fov  # 水平视场角（度）
        self.m_yaw = m_yaw  # 游戏引擎常数（Source引擎默认0.022）

        # 计算焦距系数 f = (W/2) / tan(HFOV/2)
        hfov_rad = math.radians(self.fov / 2)
        self.focal_length = (self.screen_width / 2) / math.tan(hfov_rad)

        # 测试参数
        self.test_distances = [30, 50, 100]  # 测试的像素距离
        self.calibration_results = []

        # 历史测试结果（用于综合测试）
        self.previous_results = previous_results if previous_results else []

        # 配置文件路径
        self.config_file = config_file_path

    def pixel_to_angle(self, pixel_distance):
        """
        将像素距离转换为视角角度（度）
        公式: θ = arctan(Δpixel / f)
        """
        angle_rad = math.atan(pixel_distance / self.focal_length)
        angle_deg = math.degrees(angle_rad)
        return angle_deg

    def countdown(self, seconds, message=""):
        """
        倒计时显示
        """
        print(f"\n{message}")
        for i in range(seconds, 0, -1):
            print(f"  {i} 秒...", end='\r')
            time.sleep(1)
        print("  开始！    ")

    def smooth_move_mouse(self, target_x, target_y, duration=0.3):
        """平滑移动鼠标"""
        try:
            MOUSEEVENTF_MOVE = 0x0001
            MOUSEEVENTF_ABSOLUTE = 0x8000

            current_x, current_y = pyautogui.position()
            delta_x = target_x - current_x
            delta_y = target_y - current_y

            steps = 50
            step_delay = duration / steps

            for i in range(1, steps + 1):
                step_x = int(current_x + (delta_x * i / steps))
                step_y = int(current_y + (delta_y * i / steps))

                abs_x = int(step_x * 65535 / self.screen_width)
                abs_y = int(step_y * 65535 / self.screen_height)

                ctypes.windll.user32.mouse_event(
                    MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE,
                    abs_x, abs_y, 0, 0
                )
                time.sleep(step_delay)

            return True
        except Exception as e:
            print(f"✗ 移动失败: {e}")
            return False

    def test_with_countdown(self, pixel_distance):
        """使用倒计时的方式测试（基于角度计算）"""
        # 计算这个像素距离对应的角度
        angle_deg = self.pixel_to_angle(pixel_distance)

        print(f"\n{'=' * 60}")
        print(f"测试 {pixel_distance} 像素移动 (对应 {angle_deg:.2f}°)")
        print(f"{'=' * 60}")

        # 步骤1：准备阶段
        print("\n步骤 1: 将准星对准一个明显的参考点")
        print("        (如：建筑边缘、门框、树等)")
        self.countdown(5, "准备时间：")

        # 记录起始位置
        start_x, start_y = pyautogui.position()
        print(f"✓ 已记录起始位置: ({start_x}, {start_y})")

        # 步骤2：脚本移动鼠标
        print(f"\n步骤 2: 脚本将向右移动 {pixel_distance} 像素")
        self.countdown(3, "即将移动：")

        target_x = start_x + pixel_distance
        success = self.smooth_move_mouse(target_x, start_y, duration=0.3)

        if not success:
            return None

        time.sleep(0.5)
        print(f"✓ 移动完成")

        # 步骤3：用户手动移回
        print(f"\n步骤 3: 请手动移动鼠标，将准星重新对准参考点")
        self.countdown(10, "操作时间：")

        # 记录结束位置
        end_x, end_y = pyautogui.position()
        print(f"✓ 已记录结束位置: ({end_x}, {end_y})")

        return pixel_distance, start_x, end_x, angle_deg

    def run_calibration(self):
        """运行完整的校准流程（基于FOV角度计算）"""
        print("\n" + "=" * 60)
        print("游戏灵敏度校准工具 - 基于FOV角度校准版本")
        print("=" * 60)
        print(f"\n配置参数:")
        print(f"  屏幕宽度: {self.screen_width}px")
        print(f"  水平FOV: {self.fov}°")
        print(f"  m_yaw: {self.m_yaw}")
        print(f"  焦距系数: {self.focal_length:.2f}")
        print("\n特点：基于视场角计算，与目标距离无关")
        print("适用于全屏游戏")
        print("=" * 60)

        self.countdown(10, "准备开始校准，请切换到游戏窗口：")

        for distance in self.test_distances:
            result = self.test_with_countdown(distance)

            if result is None:
                continue

            pixel_distance, start_x, end_x, angle_deg = result

            # 计算实际移动距离（像素）
            actual_pixel_move = abs(end_x - start_x)

            # 将实际移动的像素转换为角度
            actual_angle_move = self.pixel_to_angle(actual_pixel_move)

            print(f"\n分析:")
            print(f"  脚本移动: {pixel_distance} 像素 ({angle_deg:.2f}°)")
            print(f"  你移回了: {actual_pixel_move} 像素 ({actual_angle_move:.2f}°)")

            # 计算灵敏度系数（基于角度）
            # 理论上：actual_angle = angle_deg
            # 实际上：由于游戏灵敏度，actual_angle 可能不同
            if actual_pixel_move > 0:
                # 角度比例系数
                angle_ratio = actual_angle_move / angle_deg

                self.calibration_results.append({
                    'screen_pixels': pixel_distance,
                    'screen_angle': angle_deg,
                    'actual_pixels': actual_pixel_move,
                    'actual_angle': actual_angle_move,
                    'angle_ratio': angle_ratio
                })
                print(f"  角度比例: {angle_ratio:.4f}")

            # 等待下一次测试
            if distance != self.test_distances[-1]:
                time.sleep(3)

        # 计算最终结果
        self.calculate_final_sensitivity()

    def calculate_final_sensitivity(self):
        """计算最终的灵敏度系数并保存（基于角度，支持综合历史数据）"""
        if not self.calibration_results:
            print("\n没有校准数据")
            return

        print("\n" + "=" * 60)
        print("本次测试结果")
        print("=" * 60)

        for i, result in enumerate(self.calibration_results, 1):
            print(f"{i}. 脚本移动: {result['screen_pixels']}px ({result['screen_angle']:.2f}°) -> "
                  f"你移回: {result['actual_pixels']}px ({result['actual_angle']:.2f}°) -> "
                  f"角度比例: {result['angle_ratio']:.4f}")

        # 计算本次平均角度比例
        current_avg = sum(r['angle_ratio'] for r in self.calibration_results) / len(self.calibration_results)
        print(f"\n本次平均角度比例 = {current_avg:.4f}")

        # 如果有历史数据，进行综合计算
        all_results = self.previous_results + self.calibration_results

        if self.previous_results:
            print("\n" + "=" * 60)
            print("综合所有历史数据")
            print("=" * 60)
            print(f"历史测试次数: {len(self.previous_results)}")
            print(f"本次测试次数: {len(self.calibration_results)}")
            print(f"总测试次数: {len(all_results)}")

        # 计算综合平均角度比例
        avg_angle_ratio = sum(r['angle_ratio'] for r in all_results) / len(all_results)

        # 计算标准差，检查一致性
        variance = sum((r['angle_ratio'] - avg_angle_ratio) ** 2 for r in all_results) / len(all_results)
        std_dev = variance ** 0.5

        print(f"\n综合平均角度比例 = {avg_angle_ratio:.4f}")
        print(f"标准差 = {std_dev:.4f}")

        if std_dev < 0.1:
            print("✓ 校准结果一致性良好！")
        else:
            print("⚠ 校准结果波动较大，建议重新校准")

        self.save_config(avg_angle_ratio, all_results)

    def save_config(self, angle_ratio, all_results):
        """保存灵敏度配置到文件（包含FOV参数和所有历史数据）"""
        config = {
            'angle_ratio': angle_ratio,
            'fov': self.fov,
            'm_yaw': self.m_yaw,
            'screen_width': self.screen_width,
            'screen_height': self.screen_height,
            'focal_length': self.focal_length,
            'calibration_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'test_results': all_results,
            'total_tests': len(all_results),
            'description': '基于FOV的角度校准，与目标距离无关'
        }

        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            print(f"\n✓ 配置已保存到: {self.config_file.absolute()}")
        except Exception as e:
            print(f"\n✗ 保存配置失败: {e}")
