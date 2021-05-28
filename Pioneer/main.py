import _thread
import time

from PIL.ImageTk import getimage
from PySide2.QtCore import QFile, QStringListModel, QRect
from PySide2.QtGui import QImage, QPixmap
from PySide2.QtUiTools import QUiLoader
from PySide2.QtWidgets import *
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image
import cv2
from PySide2 import QtWidgets, QtCore
import os
import json
import torch
from torchvision import transforms

from Pioneer.model.model import efficientnet_b0 as create_model
from Pioneer.utils.ui_utils import pie, history_save, history_to_chart


class Child(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("历史记录显示")
        self.resize(960, 700)

        qfile = QFile("ui\child.ui")
        qfile.open(QFile.ReadOnly)
        qfile.close()

        self.ui = QUiLoader().load(qfile)

        slm = QStringListModel()  # 创建mode

        os.chdir("history")
        self.qList = os.listdir()  # 添加的数组数据
        os.chdir("../")

        slm.setStringList(self.qList)  # 将数据设置到model
        self.ui.listView.setModel(slm)  # 绑定 listView 和 model
        self.ui.listView.clicked.connect(self.clickedlist)  # listview 的点击事件

    def clickedlist(self, qModelIndex):
        chart, df = history_to_chart(f"history/{self.qList[qModelIndex.row()]}")
        showImage = QImage(chart.data, chart.shape[1], chart.shape[0], QImage.Format_RGB888)
        self.ui.label.setPixmap(QPixmap.fromImage(showImage))
        self.ui.textEdit.append(str(df))


class Fire:
    def __init__(self):
        super(Fire, self).__init__()

        self.child_window = Child()
        qfile = QFile("ui\main.ui")
        qfile.open(QFile.ReadOnly)
        qfile.close()

        self.ui = QUiLoader().load(qfile)
        self.ui.btn_lead.clicked.connect(self.open_video)
        self.ui.btn_record.clicked.connect(self.show_child)
        self.ui.btn_infer.clicked.connect(self.start_predict)
        self.ui.btn_open.clicked.connect(self.setting)

        self.ui.label.setScaledContents(True)  # 让图片自适应 label 大小
        self.ui.label_2.setScaledContents(True)

        self.file_path: str

    def show_child(self):
        self.child_window.ui.show()

    def open_video(self):
        file_dialog = QFileDialog(self.ui.btn_lead)
        # 设置可以打开任何文件
        file_dialog.setFileMode(QFileDialog.AnyFile)
        # 文件过滤
        self.file_path, _ = file_dialog.getOpenFileName(self.ui.btn_lead, 'open file', './', )
        # 判断是否正确打开文件
        if not self.file_path:
            QMessageBox.warning(self.ui.btn_lead, "警告", "文件错误打开或打开文件失败！", QMessageBox.Yes)
            return

        self.ui.textEdit.append(f"读入文件 {self.file_path} 成功")
        # 设置标签的图片
        self.ui.label.setPixmap(self.file_path)  # 输入为图片路径，比如当前文件内的logo.png图片

    def start_predict(self):
        def in_pre():
            # todo 解耦设置 和 模型初始化
            self.ui.textEdit.append("loading model...")
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            img_size = {"B0": 224,
                        "B1": 240,
                        "B2": 260,
                        "B3": 300,
                        "B4": 380,
                        "B5": 456,
                        "B6": 528,
                        "B7": 600}
            num_model = "B0"

            data_transform = transforms.Compose(
                [transforms.Resize(img_size[num_model]),
                 transforms.CenterCrop(img_size[num_model]),
                 transforms.ToTensor(),
                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

            # read class_indict
            json_path = 'class_indices.json'
            assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

            json_file = open(json_path, "r")
            class_indict = json.load(json_file)

            # create model
            model = create_model(num_classes=3).to(device)
            # load model weights
            model_weight_path = "weights/model-0.pth"
            model.load_state_dict(torch.load(model_weight_path, map_location=device))
            model.eval()

            save_file = "history/" + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + ".csv"
            save_list = []

            # 加载图片或者视频
            self.ui.textEdit.append("start inferring...")
            if self.file_path.endswith(".mp4"):
                vc = cv2.VideoCapture(self.file_path)
                frames = vc.get(cv2.CAP_PROP_FRAME_COUNT)
                for i in range(int(frames)):
                    rval, img = vc.read()
                    if rval:
                        img = Image.fromarray(img)

                        # 展示原图
                        show = np.array(img)
                        showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
                        self.ui.label.setPixmap(QPixmap.fromImage(showImage))

                        # predict
                        with torch.no_grad():
                            # [N, C, H, W]
                            img = data_transform(img)
                            # expand batch dimension
                            img = torch.unsqueeze(img, dim=0)
                            # predict class
                            output = torch.squeeze(model(img.to(device))).cpu()
                            predict_data = torch.softmax(output, dim=0)
                            predict_cla = torch.argmax(predict_data).numpy()

                        # 数据整理
                        print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                                     predict_data[predict_cla].numpy())
                        predict_data = np.array(predict_data).tolist()
                        save_list.append(predict_data)

                        # 文本框追加输出命令行
                        self.ui.textEdit.append(str(predict_data))
                        self.ui.textEdit.append(print_res + "\n")
                        # 饼图
                        show = pie(predict_data)
                        showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
                        self.ui.label_2.setPixmap(QPixmap.fromImage(showImage))

            elif self.file_path.endswith(".jpg"):
                # load image
                img = Image.open(self.file_path)

                # 展示原图
                show = np.array(img)
                showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
                self.ui.label.setPixmap(QPixmap.fromImage(showImage))

                # predict
                with torch.no_grad():
                    # [N, C, H, W]
                    img = data_transform(img)
                    # expand batch dimension
                    img = torch.unsqueeze(img, dim=0)
                    # predict class
                    output = torch.squeeze(model(img.to(device))).cpu()
                    predict_data = torch.softmax(output, dim=0)
                    predict_cla = torch.argmax(predict_data).numpy()

                # 数据整理
                print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                             predict_data[predict_cla].numpy())
                predict_data = np.array(predict_data).tolist()
                save_list.append(predict_data)

                # 文本框追加输出命令行
                self.ui.textEdit.append(str(predict_data))
                self.ui.textEdit.append(print_res + "\n")
                # 饼图
                show = pie(predict_data)
                showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
                self.ui.label_2.setPixmap(QPixmap.fromImage(showImage))

            history_save(save_list, save_file)  # 保存历史记录
            self.ui.textEdit.append("save successful! " + save_file + "\n")

        _thread.start_new_thread(in_pre, ())

    def setting(self):
        pass


if __name__ == '__main__':
    app = QApplication()
    app.setStyle('Windows')
    main = Fire()
    main.ui.show()
    app.exec_()
