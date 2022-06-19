"""
show the oracle image.
绘制甲骨文数据集的图像
"""
import OM_reader
import parameters
import torch
import OM_reader
from matplotlib import pyplot as plt

def get_oracle_mnist_labels(labels,language="CN"):
    """返回Oracle-MNIST数据集的中文或英文文本标签"""
    text_labels_en = ['big', 'sun', 'moon', 'cattle', 'next',
                   'field', 'not', 'arrow', 'time', 'wood']
    text_labels_CN = ['大', '日', '月', '牛', '翌',
                   '田', '勿', '矢', '巳', '木']
    text_labels = text_labels_en if language=="en" else text_labels_CN
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """绘制图像列表"""
    # 设置 中文title显示
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy(),cmap="gray")
        else:
            # PIL图片
            ax.imshow(img,cmap="gray")
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])

    plt.show()
    return axes

if __name__ == "__main__":
    train_loader,test_loader = OM_reader.load_oracle_mnist_data(12)

    imgs,labels = next(iter(train_loader))
    print("imgs.shape:" + str(imgs.shape))
    print("labels.shape:" + str(labels.shape))

    show_images(imgs.reshape(12, 28, 28), 4, 3, titles=get_oracle_mnist_labels(labels))