import matplotlib.pyplot as plt
class DV():

    def pie(self,List):
               # 设定字体为微软雅黑
        plt.rcParams['font.sans-serif']=['Microsoft Yahei']
        # 指定饼图的每个切片名称
        labels = '弱火', '正常', '过火'
        colors=['deepskyblue','darkorange','red']
        # 指定每个切片的数值，从而决定了百分比

        explode = (0, 0.1, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

        fig1, ax1 = plt.subplots()
        ax1.pie(List, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90,colors=colors)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.show()

        plt.close()




