#!/usr/bin/env python3
"""
批量手动标注工具
连续标注多张图片
"""

import cv2
import numpy as np
from pathlib import Path
import json
import argparse


class BatchAnnotator:
    """批量标注工具类"""

    def __init__(self):
        self.image = None
        self.image_copy = None
        self.boxes = []
        self.current_box = None
        self.drawing = False
        self.box_color = (0, 0, 255)  # 红色
        self.box_thickness = 3
        self.window_name = "批量标注工具"
        self.skip_current = False
        self.output_dir = None  # 输出目录

    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.current_box = [x, y, x, y]

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_box[2] = x
                self.current_box[3] = y

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.current_box[2] = x
            self.current_box[3] = y

            x1 = min(self.current_box[0], self.current_box[2])
            y1 = min(self.current_box[1], self.current_box[3])
            x2 = max(self.current_box[0], self.current_box[2])
            y2 = max(self.current_box[1], self.current_box[3])

            if x2 - x1 > 5 and y2 - y1 > 5:
                self.boxes.append([x1, y1, x2, y2])

            self.current_box = None

    def draw_boxes(self):
        """绘制所有标注框"""
        self.image_copy = self.image.copy()

        for i, box in enumerate(self.boxes):
            x1, y1, x2, y2 = box
            cv2.rectangle(self.image_copy, (x1, y1), (x2, y2), self.box_color, self.box_thickness)
            label = f"#{i+1}"
            cv2.putText(self.image_copy, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.box_color, 2)

        if self.drawing and self.current_box:
            x1, y1, x2, y2 = self.current_box
            cv2.rectangle(self.image_copy, (x1, y1), (x2, y2), self.box_color, self.box_thickness)

    def show_help(self, current_index, total):
        """显示帮助信息"""
        help_text = [
            f"Image {current_index}/{total}",
            "Left Click: Draw Box",
            "U: Undo | C: Clear",
            "N: Next | P: Previous",
            "S: Skip | Q: Quit",
            f"Boxes: {len(self.boxes)}"
        ]

        y_offset = 30
        for i, text in enumerate(help_text):
            y = y_offset + i * 25
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(self.image_copy, (10, y - 20), (20 + w, y + 5), (0, 0, 0), -1)
            cv2.putText(self.image_copy, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def annotate_single(self, image_path: Path, current_index: int, total: int):
        """标注单张图片"""
        # 读取图片
        try:
            with open(image_path, 'rb') as f:
                image_data = np.frombuffer(f.read(), np.uint8)
                self.image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"错误: 无法读取图片 {image_path}: {e}")
            return 'next'

        if self.image is None:
            print(f"错误: 无法解码图片 {image_path}")
            return 'next'

        # 加载已有标注
        self.boxes = []
        annotation_file = image_path.parent / f"{image_path.stem}_annotations.json"
        if annotation_file.exists():
            try:
                with open(annotation_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.boxes = data.get('boxes', [])
                    print(f"加载了 {len(self.boxes)} 个已有标注")
            except:
                pass

        print(f"\n[{current_index}/{total}] 标注: {image_path.name}")

        # 主循环
        while True:
            self.draw_boxes()
            self.show_help(current_index, total)
            cv2.imshow(self.window_name, self.image_copy)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('n') or key == ord('N'):
                self.save_annotations(image_path, annotation_file)
                return 'next'
            elif key == ord('p') or key == ord('P'):
                self.save_annotations(image_path, annotation_file)
                return 'prev'
            elif key == ord('s') or key == ord('S'):
                return 'next'
            elif key == ord('q') or key == ord('Q') or key == 27:
                return 'quit'
            elif key == ord('u') or key == ord('U'):
                if self.boxes:
                    self.boxes.pop()
            elif key == ord('c') or key == ord('C'):
                self.boxes = []

    def save_annotations(self, image_path: Path, annotation_file: Path):
        """保存标注"""
        if not self.boxes:
            return

        annotation_data = {
            'image': str(image_path.name),
            'boxes': self.boxes,
            'count': len(self.boxes)
        }

        # 保存到输出目录
        output_annotation_file = self.output_dir / f"{image_path.stem}_annotations.json"

        try:
            with open(output_annotation_file, 'w', encoding='utf-8') as f:
                json.dump(annotation_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存失败: {e}")
            return

        # 保存标注图片到输出目录
        output_path = self.output_dir / f"{image_path.stem}_annotated{image_path.suffix}"
        annotated_image = self.image.copy()
        for box in self.boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), self.box_color, self.box_thickness)

        try:
            _, encoded = cv2.imencode(image_path.suffix, annotated_image)
            with open(output_path, 'wb') as f:
                f.write(encoded.tobytes())
            print(f"已保存 {len(self.boxes)} 个标注到: {self.output_dir.name}/")
        except Exception as e:
            print(f"保存图片失败: {e}")

    def batch_annotate(self, image_dir: Path):
        """批量标注目录中的图片"""
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
        image_files = [f for f in image_dir.iterdir()
                      if f.is_file() and f.suffix.lower() in image_extensions]

        if not image_files:
            print(f"在 {image_dir} 中没有找到图片文件")
            return

        # 创建输出目录
        self.output_dir = image_dir / "annotated_images"
        self.output_dir.mkdir(exist_ok=True)
        print(f"输出目录: {self.output_dir}")

        print(f"找到 {len(image_files)} 张图片")
        print("操作说明:")
        print("  左键拖动: 绘制框")
        print("  U: 撤销 | C: 清除")
        print("  N: 保存并下一张 | P: 保存并上一张")
        print("  S: 跳过当前图片 | Q/ESC: 退出")

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        idx = 0
        while idx < len(image_files):
            result = self.annotate_single(image_files[idx], idx + 1, len(image_files))

            if result == 'next':
                idx += 1
            elif result == 'prev':
                idx = max(0, idx - 1)
            elif result == 'quit':
                break

        cv2.destroyAllWindows()
        print("\n批量标注完成！")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="批量手动标注工具")
    parser.add_argument('-d', '--directory', type=str, default='.', help='图片目录路径')
    args = parser.parse_args()

    image_dir = Path(args.directory)
    if not image_dir.exists() or not image_dir.is_dir():
        print(f"错误: 目录不存在: {image_dir}")
        return

    annotator = BatchAnnotator()
    annotator.batch_annotate(image_dir)


if __name__ == "__main__":
    #  python batch_annotate.py -d .
    main()

