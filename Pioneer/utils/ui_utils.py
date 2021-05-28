# -- coding: utf-8 --
import io

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd


def pie(predict_data):
    plt.rcParams['font.sans-serif'] = ['Microsoft Yahei']
    # 指定饼图的每个切片名称
    labels = '弱火', '正常', '过火'
    colors = ['r', 'y', 'b']
    # 指定每个切片的数值，从而决定了百分比

    if predict_data[0] > predict_data[1] and predict_data[0] > predict_data[2]:
        explode = (0.1, 0, 0)
    elif predict_data[1] > predict_data[0] and predict_data[1] > predict_data[2]:
        explode = (0, 0.1, 0)
    else:
        explode = (0, 0, 0.1)

    fig1, ax1 = plt.subplots()

    ax1.pie(predict_data, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90, colors=colors)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    canvas = fig1.canvas

    # 去掉图片四周的空白

    # 设置画布大小（单位为英寸），每1英寸有100个像素
    fig1.set_size_inches(4, 3)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())  # plt.gca()表示获取当前子图"Get Current Axes"。
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.margins(0, 0)

    # 获取Plt的数据并使用cv2进行保存
    buffer = io.BytesIO()  # 获取输入输出流对象
    canvas.print_png(buffer)  # 将画布上的内容打印到输入输出流对象
    data = buffer.getvalue()  # 获取流的值
    # print("plt的二进制流为:\n", data)
    buffer.write(data)  # 将数据写入buffer
    img = Image.open(buffer)  # 使用Image打开图片数据
    img = np.asarray(img)

    # cv2.imwrite("02.jpg", img)
    buffer.close()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.clf()  # 防止内存泄漏。清除当前figure的所有axes，但是不关闭这个window，所以能继续复用于其他的plot。
    # plt.close()  # 关闭 window，如果没有指定，则指当前 window

    return img


def history_save(predict_data, save_file):
    df = pd.DataFrame(predict_data, columns=['弱火', '正常', '过火', '结果'], dtype=float)
    df.to_csv(save_file)


def history_to_chart(file):
    plt.rcParams['font.sans-serif'] = ['Microsoft Yahei']
    data_frame = pd.read_csv(file, index_col=0, encoding='utf-8', low_memory=False)

    chart_data = data_frame.drop('结果', axis=1)
    fig1, ax1 = plt.subplots()

    ax1.plot(chart_data)

    canvas = fig1.canvas
    # 获取Plt的数据并使用cv2进行保存
    buffer = io.BytesIO()  # 获取输入输出流对象
    canvas.print_png(buffer)  # 将画布上的内容打印到输入输出流对象
    data = buffer.getvalue()  # 获取流的值
    # print("plt的二进制流为:\n", data)
    buffer.write(data)  # 将数据写入buffer
    img = Image.open(buffer)  # 使用Image打开图片数据
    img = np.asarray(img)

    # cv2.imwrite("02.jpg", img)
    buffer.close()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.clf()

    # cv2.imshow("img", img)
    # cv2.waitKey(0)

    return img, data_frame


if __name__ == '__main__':
    history_to_chart("../history/2021_05_28_12_54_08.csv")
