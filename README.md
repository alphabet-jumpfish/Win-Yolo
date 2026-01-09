# YOLO 实时目标检测与自定义训练系统

这是一个功能完整的 YOLOv8 目标检测系统，支持实时屏幕检测、自定义模型训练、模型质量分析等功能。

## 核心功能

### 1. 实时屏幕检测
- 实时捕获电脑屏幕并进行目标检测
- 支持检测 COCO 数据集的 80 个类别（人、车、动物等）
- 支持自定义训练模型检测
- 实时显示 FPS 和检测结果
- **自动鼠标控制**：检测到目标时自动移动鼠标到目标位置
- **智能跟踪**：每个目标只移动一次鼠标，避免重复移动
- **暂停/恢复功能**：按 'q' 键暂停或恢复程序

### 2. 自定义模型训练
- 支持红色框标注方式快速标注训练数据
- 自动生成 YOLO 格式标签文件
- 自动分割训练集和验证集
- 支持自定义类别名称
- 支持多类别训练
- 完整的训练流程自动化

### 3. 模型质量分析
- 从 .pt 文件直接读取训练指标
- 显示 Precision（精确度）、Recall（召回率）、mAP50、mAP50-95
- 自动评估模型质量（优秀/良好/一般/较差）
- 支持测试图像批量检测
- 详细的检测结果展示

### 4. 双模型检测
- 同时使用预训练模型和自定义模型
- 不同颜色区分检测结果（蓝色=预训练，红色=自定义）
- 灵活配置检测类别和置信度阈值

## 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：
- ultralytics (YOLOv8)
- opencv-python
- numpy
- mss (屏幕捕获)
- pyautogui (鼠标控制)
- torch (PyTorch)

## 使用方法

### 1. 实时屏幕检测

运行实时检测程序：

```bash
python DetectMain.py
```

**键盘控制：**
- 按 `q` 键：暂停/恢复程序
- 按 `ESC` 键：退出程序

**配置选项（在 DetectMain.py 中修改）：**

```python
# 模型选择
model_name = 'yolov8n.pt'  # 预训练模型
# model_name = 'training/warZ/runs/train/weights/best.pt'  # 自定义模型

# 检测类别
detect_classes = [0]  # [0]=只检测person，None=检测所有类别

# 检测参数
detector.run(
    conf_threshold=0.5,  # 置信度阈值 (0.0-1.0)
    display_scale=0.6    # 显示窗口缩放比例
)
```

### 2. 自定义模型训练

运行训练程序：

```bash
python TrainMain.py
```

**训练步骤：**

1. **配置训练参数**（在 TrainMain.py 中）：
```python
class_names = ['warZ']  # 自定义类别名称
project_name = 'warZ_model'  # 项目名称
keep_base_classes = False  # 是否保留原有80个类别
```

2. **准备训练数据**：
   - 将带有红色框标注的图像放入：`training/{project_name}/dataset/images/train/`
   - 红色框会自动转换为 YOLO 格式标签

3. **开始训练**：
   - 程序会自动生成标签、分割数据集、开始训练
   - 训练结果保存在：`training/{project_name}/runs/train/weights/`

**训练参数：**
```python
epochs=100,        # 训练轮数
val_ratio=0.2,     # 验证集比例（20%）
imgsz=640,         # 图像大小
batch=16,          # 批次大小
device='cuda'      # 使用GPU（如果有）
```

### 3. 模型质量分析

运行模型检查程序：

```bash
python YoloModelCheckMain.py
```

**功能：**
- 从 .pt 文件直接读取训练指标
- 显示 Precision、Recall、mAP50、mAP50-95
- 自动评估模型质量
- 测试图像批量检测

**配置（在 YoloModelCheckMain.py 中）：**
```python
model_path = 'training/warZ/runs/train/weights/best.pt'
test_image_dir = r"C:\Users\xxx\Pictures\Screenshots"
```

### 4. 双模型检测

运行双模型检测程序：

```bash
python DualDetectMain.py
```

**功能：**
- 同时使用预训练模型和自定义模型
- 蓝色框：预训练模型检测结果
- 红色框：自定义模型检测结果

**配置（在 DualDetectMain.py 中）：**
```python
base_model_path = 'models/yolov8n.pt'
custom_model_path = 'training/warZ/runs/train/weights/best.pt'
base_classes = [0]      # 预训练模型检测类别
custom_classes = [0]    # 自定义模型检测类别
```

## 项目结构

```
Win-Yolo/
├── DetectMain.py              # 实时屏幕检测主程序
├── TrainMain.py               # 模型训练主程序
├── YoloModelCheckMain.py      # 模型质量分析主程序
├── DualDetectMain.py          # 双模型检测主程序
├── service/
│   ├── ScreenDetectorService.py    # 屏幕检测服务
│   ├── YoloTrainer.py              # 模型训练服务
│   ├── YoloModelCheckService.py    # 模型检查服务
│   └── DualModelDetector.py        # 双模型检测服务
├── models/                    # 预训练模型目录
│   └── yolov8n.pt
└── training/                  # 训练项目目录
    └── {project_name}/
        ├── dataset/           # 数据集
        └── runs/              # 训练结果
```

## 模型选择

### 预训练模型（COCO 80类）
- `yolov8n.pt` - 最快，适合实时检测（推荐）
- `yolov8s.pt` - 速度和精度平衡
- `yolov8m.pt` - 中等精度
- `yolov8l.pt` - 高精度
- `yolov8x.pt` - 最高精度，速度最慢

### COCO 数据集常用类别
```python
0: person (人)
2: car (汽车)
3: motorcycle (摩托车)
5: bus (公交车)
7: truck (卡车)
16: dog (狗)
17: cat (猫)
```

完整的 80 个类别列表请参考 DetectMain.py 文件。

## 注意事项

### 实时检测
- 首次运行时，程序会自动下载 YOLO 模型文件
- 建议使用 `yolov8n.pt` 以获得最佳实时性能
- 如果显示窗口太大或太小，可以调整 `display_scale` 参数
- 鼠标自动移动功能可能影响正常操作，可在代码中禁用

### 自定义训练
- **训练数据量**：建议至少 50-100 张图像
- **图像质量**：确保图像清晰，目标明显
- **标注准确性**：红色框要准确框选目标区域
- **训练时间**：根据数据量和硬件，可能需要几分钟到几小时
- **GPU 加速**：强烈建议使用 GPU 训练（设置 `device='cuda'`）

### 模型质量评估标准
- **Precision > 60%**：良好
- **Recall > 60%**：良好
- **mAP50 > 60%**：良好
- 如果指标过低，需要增加训练数据或改进标注质量

## 训练建议

### 提高模型质量的方法
1. **增加训练数据**：至少 50-100 张高质量图像
2. **改进标注质量**：使用 LabelImg 工具进行精确标注
3. **数据多样性**：包含不同角度、光照、背景的图像
4. **增加训练轮数**：设置 `epochs=200` 或更多
5. **数据增强**：YOLO 会自动进行数据增强

## 系统要求

- Python 3.8+
- Windows/Linux/macOS
- 建议使用 GPU 以获得更好的训练性能（可选）
- 至少 4GB RAM
- 建议 8GB+ RAM 用于训练

## 常见问题

### Q: 训练后的模型检测不到目标？
**A:** 可能的原因：
1. 训练数据太少（少于50张）
2. 训练图像包含标注框（应使用纯净图像）
3. 标注质量差
4. 模型路径配置错误

**解决方法：**
- 运行 `python YoloModelCheckMain.py` 检查模型质量
- 如果 Precision < 30%，需要重新准备训练数据
- 确保 DetectMain.py 中使用正确的模型路径

### Q: 如何提高检测准确率？
**A:**
1. 增加训练数据到 100+ 张
2. 使用 LabelImg 工具精确标注
3. 增加训练轮数到 200+
4. 确保训练数据多样性

### Q: 鼠标自动移动功能如何禁用？
**A:** 在 `service/ScreenDetectorService.py` 中注释掉：
```python
# self.move_mouse_to_person(results)
```

## 许可证

本项目仅供学习和研究使用。

## 作者

Win-Yolo - YOLO 实时目标检测与自定义训练系统

---

**祝你使用愉快！如有问题，请查看常见问题部分或检查代码注释。**
