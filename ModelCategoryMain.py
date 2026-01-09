from service.ModelClassInspectorService import ModelClassInspector
import os


def main():
    print("=" * 60)
    print("YOLO 模型类别查询工具")
    print("=" * 60)

    # 选择要查询的模型
    # 选项1: 预训练模型
    # model_path = os.path.dirname(__file__) + '/models/yolov8n.pt'
    model_path = os.path.dirname(__file__) + '/models/best.pt'

    # 选项2: 自定义训练的模型
    # model_path = 'training/warZ/runs/train/weights/best.pt'

    print(f"\n正在查询模型: {model_path}\n")

    try:
        # 创建检查器
        inspector = ModelClassInspector(model_path)

        # 1. 打印所有类别
        inspector.print_all_classes()
        #
        # # 4. 搜索类别
        # keyword = 'car'
        # results = inspector.search_classes(keyword)
        # print(f"\n搜索包含 '{keyword}' 的类别:")
        # for class_id, class_name in results:
        #     print(f"  ID {class_id}: {class_name}")

        print("\n" + "=" * 60)

    except FileNotFoundError as e:
        print(f"错误: {e}")
    except Exception as e:
        print(f"发生错误: {e}")


if __name__ == '__main__':
    main()