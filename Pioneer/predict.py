import os
import json
import pandas as pd
import time
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from Pioneer.model import efficientnet_b0 as create_model
from Pioneer.DV import DV
from Pioneer.history_record import history

def main():
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
    #load video
    import cv2

    vc = cv2.VideoCapture('VID_20201228_171007.mp4')  # 读入视频文件，命名cv


    if vc.isOpened():  # 判断是否正常打开
        rval, frame = vc.read()
    else:
        rval = False
    frames =vc.get(cv2.CAP_PROP_FRAME_COUNT)
    print("视频总帧数:%s"%frames)
    number=int(frames)

    for i in range(int(frames)):  # 循环读取视频帧
        rval, img = vc.read()


        img = Image.fromarray(img)
        # [N, C, H, W]
        img = data_transform(img)

        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        # read class_indict
        json_path = './class_indices.json'
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

        json_file = open(json_path, "r")
        class_indict = json.load(json_file)

        # create model
        model = create_model(num_classes=3).to(device)
        # load model weights
        model_weight_path = "weights/model-0.pth"
        model.load_state_dict(torch.load(model_weight_path, map_location=device))
        model.eval()
        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            print(predict)
            predict_cla = torch.argmax(predict).numpy()

        # print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
        #                                              predict[predict_cla].numpy())
        #
        # plt.title(print_res)
        # print(print_res)
        # plt.show()
        # data visualization
        predict_data = np.array(predict).tolist()
        dv = DV()
        dv.pie(predict_data)
        ###历史纪录
        now_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        history(now_time,number,predict_data[0],predict_data[1],predict_data[2])#将数据保存到历史记录中
        #####
        cv2.waitKey(1)
    vc.release()
    # load image

    img_path = "flame-1..jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)

    plt.waitforbuttonpress()
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = create_model(num_classes=3).to(device)
    # load model weights
    model_weight_path = "weights/model-0.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        print(predict)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())


    plt.title(print_res)
    print(print_res)
    plt.show()
    # data visualization
    predict_data = np.array(predict).tolist()
    dv = DV()
    dv.pie(predict_data)



if __name__ == '__main__':
    main()
