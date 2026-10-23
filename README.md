
滚边毛刺、边缘毛刺、短划痕、表面凹痕、麻点簇、边缘凹痕、粗糙斑块、长划痕、点状麻点、表面斑块共 10 类工业零部件表面缺陷![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e5354ae1ab004d9688d47aa66fd2c61e.png)
，1632张，提供yolo，voc，coco三种标注方式
图像尺寸:620*526
类别数量:10类![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/bb60aebcf1cb424e9348d071f798af50.png)

训练集图像数量:1380; 验证集图像数量:216； 测试集图像数量:36
类别名称: 每一类图像数 ，每一类标注数![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/696c7e6d920545409f82d5ceb6408830.png)

ROLLOVER_BURR 滚边毛刺 92 120
EDGE_BURR 边缘毛刺 304 400
SHORT_SCRATCH 短划痕 214 361
SURFACE_DENT 表面凹痕 227 412
PITS_CLUSTER 麻点簇 264 475
EDGE_DENT 边缘凹痕 547 849
ROUGH_PATCH 粗糙斑块 67 81
LONG_SCRATCH 长划痕 89 115
PITS_DOTS 点状麻点 84 188
SURFACE_PATCH 表面斑块 158 209
image num: 1632![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/bf0b244a930a4ad3a3d6da006d1b1f94.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/757d2998aa874e66bc1736e292d5b5a1.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/52a9d24549df4ea8a60c24b65525b811.png)
# 工业零部件产品缺陷检测数据集
## 数据集总览表
|项目|详情|
| ---- | ---- |
|数据集名称|工业零部件产品缺陷检测数据集|
|图像总数|1632张|
|标注格式|YOLO、VOC、COCO|
|图像尺寸|620*526|
|类别总数|10类|
|训练集|1380张|
|验证集|216张|
|测试集|36张|

## 类别统计表
|英文标签|中文标签|含该类图像数|标注框总数|
| ---- | ---- | ---- | ---- |
|ROLLOVER_BURR|滚边毛刺|92|120|
|EDGE_BURR|边缘毛刺|304|400|
|SHORT_SCRATCH|短划痕|214|361|
|SURFACE_DENT|表面凹痕|227|412|
|PITS_CLUSTER|麻点簇|264|475|
|EDGE_DENT|边缘凹痕|547|849|
|ROUGH_PATCH|粗糙斑块|67|81|
|LONG_SCRATCH|长划痕|89|115|
|PITS_DOTS|点状麻点|84|188|
|SURFACE_PATCH|表面斑块|158|209|

## 标签关键词
工业零部件缺陷检测、金属零件质检、毛刺、划痕、凹痕、麻点、表面斑块、YOLOv11、机器视觉、工业视觉、产品外观检测、工厂流水线质检

## 应用场景
1. 精密金属零部件出厂外观自动化质检，识别毛刺、划痕、凹坑、麻点等表面缺陷，替代人工目视检测，提升质检效率，降低漏检率。
2. 零部件生产流水线在线视觉检测，实时采集工件图像，自动判定产品是否合格。
3. 机器视觉、深度学习目标检测算法研究、毕业设计、工程项目原型开发。
4. 零部件来料抽检、成品入库复检，缺陷分类统计，用于工艺改进，分析缺陷产生原因。

## 构建系统完整代码
### 1. 数据集配置文件 defect.yaml
```yaml
path: "./parts_defect_dataset"
train: images/train
val: images/val
test: images/test

names:
  0: ROLLOVER_BURR
  1: EDGE_BURR
  2: SHORT_SCRATCH
  3: SURFACE_DENT
  4: PITS_CLUSTER
  5: EDGE_DENT
  6: ROUGH_PATCH
  7: LONG_SCRATCH
  8: PITS_DOTS
  9: SURFACE_PATCH
```

### 2. 训练脚本 train_defect.py
```python
from ultralytics import YOLO

if __name__ == '__main__':
    # 加载yolo11n预训练权重
    model = YOLO("yolo11n.pt")
    train_result = model.train(
        data="defect.yaml",
        epochs=80,
        imgsz=640,
        batch=8,
        device=0,
        workers=2,
        project="parts_defect_result",
        name="yolo11_defect",
        conf=0.25,
        augment=True
    )
```

### 3. PyQt5可视化检测界面 defect_gui.py
```python
import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QTableWidget, QTableWidgetItem
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from ultralytics import YOLO

class DefectDetectWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("工业零部件缺陷检测系统 YOLOv11")
        self.setGeometry(100, 100, 1250, 820)
        self.model = YOLO("./parts_defect_result/yolo11_defect/weights/best.pt")
        self.init_ui()

    def init_ui(self):
        #图像显示区域
        self.img_label = QLabel("零部件图像预览", self)
        self.img_label.setGeometry(20, 60, 750, 650)
        self.img_label.setStyleSheet("border:1px solid #666;")
        self.img_label.setAlignment(Qt.AlignCenter)

        self.btn_img = QPushButton("图片缺陷检测", self)
        self.btn_img.setGeometry(800,70,200,45)
        self.btn_img.clicked.connect(self.detect_image)

        #缺陷结果表格
        self.table = QTableWidget(self)
        self.table.setGeometry(20,730,1200,70)
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["序号","缺陷类别","置信度","坐标","判定结果"])

    def detect_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self,"选择零部件图片","","图片文件(*.jpg *.png *.bmp)")
        if not file_path:
            return
        results = self.model(file_path)[0]
        img_result = results.plot()
        h,w,c = img_result.shape
        qt_img = QImage(img_result.data, w, h, c*w, QImage.Format_RGB888).rgbSwapped()
        self.img_label.setPixmap(QPixmap.fromImage(qt_img).scaled(self.img_label.size(),Qt.KeepAspectRatio))

        self.table.setRowCount(0)
        for idx, box in enumerate(results.boxes):
            cls_id = int(box.cls)
            cls_name = self.model.names[cls_id]
            conf = float(box.conf)
            xyxy = box.xyxy[0].tolist()
            self.table.insertRow(idx)
            self.table.setItem(idx,0,QTableWidgetItem(str(idx+1)))
            self.table.setItem(idx,1,QTableWidgetItem(cls_name))
            self.table.setItem(idx,2,QTableWidgetItem(f"{conf:.2f}"))
            self.table.setItem(idx,3,QTableWidgetItem(f"{xyxy}"))
            self.table.setItem(idx,4,QTableWidgetItem("不合格"))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = DefectDetectWindow()
    win.show()
    sys.exit(app.exec_())
```

### 环境安装命令
```bash
pip install ultralytics torch opencv-python pyqt5
```

