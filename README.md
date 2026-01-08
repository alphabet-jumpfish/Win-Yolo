# YOLO 实时人物检测程序

这是一个使用 YOLOv8 实时检测电脑屏幕中人物的 Python 程序。

## 功能特点

- 实时捕获电脑屏幕
- 使用 YOLOv8 进行人物检测（仅检测人物，过滤其他物体）
- 显示检测框和标签
- 实时显示 FPS 和检测到的人物数量
- 支持多种 YOLO 模型
- **自动鼠标控制**：检测到新人物时，自动将鼠标移动到人物中心位置
- 智能跟踪：每个人物只移动一次鼠标，避免重复移动
- **暂停/恢复功能**：按 'q' 键可以暂停或恢复程序运行

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

运行程序：

```bash
python main.py
```

### 键盘控制

- 按 `q` 键：暂停/恢复程序
- 按 `ESC` 键：退出程序

## 配置选项

在 `main.py` 的 `main()` 函数中，你可以修改以下参数：

### 模型选择
```python
model_name = 'yolov8n.pt'  # 可选：yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
```

- `yolov8n.pt` - 最快，适合实时检测
- `yolov8s.pt` - 速度和精度平衡
- `yolov8m.pt` - 中等精度
- `yolov8l.pt` - 高精度
- `yolov8x.pt` - 最高精度，速度最慢

### 模型保存路径
```python
model_dir = './models'  # 默认保存在当前目录的 models 文件夹
# model_dir = 'D:/AI_Models/YOLO'  # 可以设置为任意自定义路径
```

模型文件会自动下载并保存到指定的目录中。如果模型已存在，则直接加载，不会重复下载。

### 检测参数
```python
detector.run(
    conf_threshold=0.5,  # 置信度阈值 (0.0-1.0)，越高越严格
    display_scale=0.6    # 显示窗口缩放比例
)
```

## 注意事项

- 首次运行时，程序会自动下载 YOLO 模型文件
- 建议使用 `yolov8n.pt` 以获得最佳实时性能
- 如果显示窗口太大或太小，可以调整 `display_scale` 参数
- 程序会检测 COCO 数据集中的 80 种常见物体（人、车、动物等）

## 系统要求

- Python 3.8+
- Windows/Linux/macOS
- 建议使用 GPU 以获得更好的性能（可选）
