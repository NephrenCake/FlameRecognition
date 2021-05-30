import _thread
import time

from PySide2.QtCore import QFile, QStringListModel
from PySide2.QtGui import QImage, QPixmap
from PySide2.QtUiTools import QUiLoader
from PySide2.QtWidgets import *
import numpy as np
from PIL import Image
import cv2
from PySide2 import QtWidgets
import os
import json
import torch
from torchvision import transforms

from Pioneer.model.model import efficientnet_b0 as create_model
from Pioneer.utils.ui_utils import pie, history_save, history_to_chart


# import warnings
#
# warnings.simplefilter("error")


class Child(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("历史记录显示")
        self.resize(960, 700)

        qfile = QFile("ui\child.ui")
        qfile.open(QFile.ReadOnly)
        qfile.close()

        self.ui = QUiLoader().load(qfile)

        self.slm = QStringListModel()  # 创建mode

        # self.qList = os.listdir("history")  # 添加的数组数据

        self.slm.setStringList(os.listdir("history"))  # 将数据设置到model
        self.ui.listView.setModel(self.slm)  # 绑定 listView 和 model
        self.ui.listView.clicked.connect(self.clicked_list)  # listview 的点击事件

    def clicked_list(self, qModelIndex):
        qList = os.listdir("history")
        chart, df = history_to_chart(f"history/{qList[qModelIndex.row()]}")
        show_image = QImage(chart.data, chart.shape[1], chart.shape[0], QImage.Format_RGB888)
        self.ui.label.setPixmap(QPixmap.fromImage(show_image))
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
        self.ui.btn_open.clicked.connect(self.open_dir)

        self.ui.label.setScaledContents(True)  # 让图片自适应 label 大小
        self.ui.label_2.setScaledContents(True)

        self.files = []
        self.choice_type = None

    def show_child(self):
        self.child_window.slm.setStringList(os.listdir("history"))
        self.child_window.ui.show()

    def open_video(self):
        file_dialog = QFileDialog(self.ui.btn_lead)
        # 设置可以打开任何文件
        file_dialog.setFileMode(QFileDialog.AnyFile)
        # 文件过滤
        file, _ = file_dialog.getOpenFileName(self.ui.btn_lead, 'open file', './', )
        # 判断是否正确打开文件
        if not (file and (file.endswith(".mp4") or file.endswith(".jpg"))):
            QMessageBox.warning(self.ui.btn_lead, "警告", "文件错误打开或打开文件失败！", QMessageBox.Yes)
            return

        self.ui.textEdit.append(f"读入文件 {file} 成功")
        # 设置标签的图片
        self.ui.label.setPixmap(file)  # 输入为图片路径，比如当前文件内的logo.png图片
        self.files.clear()
        self.files.append(file)

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
            model_weight_path = "weights/B0-model-4.pth"
            model.load_state_dict(torch.load(model_weight_path, map_location=device))
            model.eval()

            save_file = "history/" + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + ".csv"
            save_list = []

            # 加载图片或者视频
            self.ui.textEdit.append("start inferring...")

            for file in self.files:

                # time.sleep(0.1)  # 等待打印信息

                if file and file.endswith(".mp4"):
                    vc = cv2.VideoCapture(file)
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
                            pro = predict_data[predict_cla].numpy()
                            print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)], pro)
                            predict_data = np.array(predict_data).tolist()

                            # 文本框追加输出命令行
                            self.ui.textEdit.append(str(predict_data))
                            self.ui.textEdit.append(print_res + "\n")
                            # 饼图
                            show = pie(predict_data)
                            showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
                            self.ui.label_2.setPixmap(QPixmap.fromImage(showImage))

                            save_data = [str(class_indict[str(predict_cla)]),
                                         "{:.3}".format(pro),
                                         predict_data[0],
                                         predict_data[1],
                                         predict_data[2]]
                            save_list.append(save_data)

                elif file and file.endswith(".jpg"):
                    # load image
                    img = Image.open(file)

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
                    pro = predict_data[predict_cla].numpy()
                    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)], pro)
                    predict_data = np.array(predict_data).tolist()

                    # 文本框追加输出命令行
                    self.ui.textEdit.append(str(predict_data))
                    self.ui.textEdit.append(print_res + "\n")
                    # 饼图
                    show = pie(predict_data)
                    showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
                    self.ui.label_2.setPixmap(QPixmap.fromImage(showImage))

                    save_data = [str(class_indict[str(predict_cla)]),
                                 "{:.3}".format(pro),
                                 predict_data[0],
                                 predict_data[1],
                                 predict_data[2]]
                    save_list.append(save_data)

            history_save(save_list, save_file)  # 保存历史记录
            self.ui.textEdit.append("save successful! " + save_file + "\n")

        _thread.start_new_thread(in_pre, ())

    def open_dir(self):
        # dir_path即为选择的文件夹的绝对路径，第二形参为对话框标题，第三个为对话框打开后默认的路径
        file_dir = QFileDialog.getExistingDirectory(self.ui.btn_open, "选择目录", "./")

        # 判断是否正确打开文件
        if not file_dir:
            QMessageBox.warning(self.ui.btn_lead, "警告", "文件错误打开或打开文件失败！", QMessageBox.Yes)
            return

        self.ui.textEdit.append(f"获取文件夹 {file_dir} 成功")

        self.files.clear()

        for cur_file in os.listdir(file_dir):
            if cur_file.endswith(".mp4") or cur_file.endswith(".jpg"):
                self.files.append(os.path.join(file_dir, cur_file))

        print(os.listdir(file_dir))
        print(self.files)


if __name__ == '__main__':
    app = QApplication()
    app.setStyle('Windows')
    main = Fire()
    main.ui.show()
    app.exec_()
