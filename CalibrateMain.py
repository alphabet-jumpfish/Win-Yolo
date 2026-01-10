import os
import json
from pathlib import Path

from service.CalibrateSensitivityService import SensitivityCalibrator


def show_fov_guide():
    """显示FOV检测指南"""
    print("\n" + "=" * 60)
    print("如何获取游戏的FOV（视场角）？")
    print("=" * 60)
    print("\n方法1: 查看游戏设置")
    print("  - 进入游戏设置 -> 视频/显示设置")
    print("  - 查找 'FOV'、'视场角'、'Field of View' 等选项")
    print("  - 记录数值（通常在60-120之间）")

    print("\n方法2: 常见游戏FOV参考")
    print("  - CS:GO/CS2: 默认90° (可调整60-90)")
    print("  - Apex Legends: 默认90° (可调整70-110)")
    print("  - Valorant: 固定103° (水平FOV)")
    print("  - Call of Duty: 默认80° (可调整60-120)")
    print("  - Overwatch: 默认103° (水平FOV)")
    print("  - PUBG: 默认80° (可调整80-103)")
    print("  - Fortnite: 默认80°")

    print("\n方法3: 在线FOV计算器")
    print("  - 访问: https://themetalmuncher.github.io/fov-calc/")
    print("  - 或搜索: 'FOV calculator' + 游戏名称")

    print("\n注意事项:")
    print("  ⚠ 有些游戏显示的是垂直FOV，需要转换为水平FOV")
    print("  ⚠ 16:9屏幕下，垂直FOV 75° ≈ 水平FOV 106°")
    print("  ⚠ 如果不确定，使用默认值90°即可")
    print("=" * 60)


def load_previous_config(config_path):
    """加载上次的配置"""
    if not config_path.exists():
        return None

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"读取配置文件失败: {e}")
        return None


def display_previous_config(config):
    """显示上次的配置信息"""
    print("\n" + "=" * 60)
    print("上次校准结果:")
    print("=" * 60)
    print(f"  校准时间: {config.get('calibration_date', '未知')}")
    print(f"  角度比例: {config.get('angle_ratio', 0):.4f}")
    print(f"  FOV: {config.get('fov', 90)}°")
    print(f"  m_yaw: {config.get('m_yaw', 0.022)}")
    print(f"  屏幕分辨率: {config.get('screen_width', 0)}x{config.get('screen_height', 0)}")

    if 'test_results' in config and config['test_results']:
        print(f"\n  测试结果数量: {len(config['test_results'])} 次")
        for i, result in enumerate(config['test_results'], 1):
            print(f"    {i}. 角度比例: {result.get('angle_ratio', 0):.4f}")
    print("=" * 60)


def main():
    """主函数 - 基于FOV的灵敏度校准"""
    print("=" * 60)
    print("游戏灵敏度校准工具 - FOV角度校准版本")
    print("=" * 60)

    config_path = Path(os.path.dirname(__file__) + '/sensitivity_config.json')

    # 尝试加载上次的配置
    previous_config = load_previous_config(config_path)

    if previous_config:
        display_previous_config(previous_config)
        print("\n请选择模式:")
        print("  1. 新建校准 (重新测试)")
        print("  2. 综合测试 (使用上次配置继续测试)")
        mode = input("请输入选择 (1/2, 默认1): ").strip()
    else:
        print("\n未找到历史配置，将进行新建校准")
        mode = "1"

    # 模式1: 新建校准
    if mode != "2":
        print("\n请输入游戏参数:")

        # 询问是否需要查看FOV帮助
        show_help = input("  需要查看FOV获取指南吗？(y/n, 默认n): ").strip().lower()
        if show_help == 'y':
            show_fov_guide()

        try:
            fov_input = input("\n  水平视场角 FOV (默认90°，输入?查看帮助): ").strip()

            # 如果用户输入?，显示帮助
            if fov_input == '?':
                show_fov_guide()
                fov_input = input("\n  水平视场角 FOV (默认90°): ").strip()

            fov = float(fov_input) if fov_input else 90.0

            m_yaw_input = input("  m_yaw 参数 (默认0.022，适用于Source引擎游戏如CS/APEX): ").strip()
            m_yaw = float(m_yaw_input) if m_yaw_input else 0.022

        except ValueError:
            print("输入无效，使用默认值: FOV=90°, m_yaw=0.022")
            fov = 90.0
            m_yaw = 0.022

        calibrator = SensitivityCalibrator(config_file_path=config_path, fov=fov, m_yaw=m_yaw)

        try:
            calibrator.run_calibration()
        except KeyboardInterrupt:
            print("\n\n校准已取消")
        except Exception as e:
            print(f"\n发生错误: {e}")

        print("\n校准完成！")

    # 模式2: 综合测试
    else:
        print("\n使用上次配置进行综合测试...")
        fov = previous_config.get('fov', 90.0)
        m_yaw = previous_config.get('m_yaw', 0.022)

        calibrator = SensitivityCalibrator(
            config_file_path=config_path,
            fov=fov,
            m_yaw=m_yaw,
            previous_results=previous_config.get('test_results', [])
        )

        try:
            calibrator.run_calibration()
        except KeyboardInterrupt:
            print("\n\n校准已取消")
        except Exception as e:
            print(f"\n发生错误: {e}")

        print("\n综合测试完成！")


if __name__ == '__main__':
    main()
