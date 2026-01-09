"""
测试训练好的模型
用于诊断模型是否能正确检测
"""
from service.YoloModelCheckService import YoloModelCheck

if __name__ == '__main__':
    model_path = 'training/warZ/runs/train/weights/best.pt'
    # 测试图像目录
    test_image_dir = r"C:\Users\wangjiawen\Pictures\Screenshots"
    check = YoloModelCheck()
    check.model_detect_pictures(model_path=model_path, test_image_dir=test_image_dir)
