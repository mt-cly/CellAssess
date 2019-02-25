import cv2 as cv
import cfg
import matplotlib.pyplot as plt
import numpy as np


# datas: [[X, Y, mean], ,.....]
def show_XY(datas):
    plt.plot(datas[0][0], datas[0][1], label='background')
    plt.plot(datas[1][0], datas[1][1], label='nucleus')
    plt.plot(datas[2][0], datas[2][1], label='blood cell sap')
    plt.legend()
    plt.xlabel("VALUE")
    plt.ylabel("PROPOTION (%)")
    plt.ylim(0, 0.12)
    plt.title("VALUE-PROPORTION in Green Channel")
    plt.savefig(cfg.PLOT_PATH)
    plt.show()

def get_XY_mean(image, channel=-1):
    source_img = image
    if channel == -1:
        source_img = cv.cvtColor(source_img, cv.COLOR_RGB2GRAY)
    else:
        source_img = cv.split(source_img)[channel]
    values = [i for i in range(0, 256)]
    counts = [0.] * 256
    mean_value = 0.
    for i in source_img.flatten():
        counts[i] += 1
        mean_value += i
    mean_value /= len(source_img.flatten())
    counts[0] = 0
    counts = np.array(counts)
    counts /= sum(counts)
    return values, counts, mean_value



if __name__ == "__main__":
    image = cv.imread(cfg.IMG_1_PATH)
    image_nucleus = cv.imread(cfg.IMG_NUCLEUS_PATH)
    image_cell = cv.imread(cfg.IMG_CELL_PATH)
    _image_bg =image.copy()
    _image_bg[image_cell==0] = 0
    _image_nucleus = image.copy()
    _image_nucleus[image_nucleus>0] = 0
    _image_cell= image.copy()
    _image_cell[image_cell>0] = 0
    _image_cell[image_nucleus==0] = 0
    cv.imshow("background", _image_bg)
    cv.imshow("cell_sap", _image_cell)
    cv.imshow("nucleus", _image_nucleus)
    info_image = get_XY_mean(_image_bg, 1)
    info_nucleus =  get_XY_mean(_image_nucleus, 1)
    info_cell = get_XY_mean(_image_cell, 1)
    datas = [info_image, info_nucleus, info_cell]
    # ============= KL ==================
    KL = (info_image[1] / (info_nucleus[1]+1e-8))
    KL[KL==0] = 1
    KL = info_image[1] * np.log(KL)
    print(sum(KL))
    show_XY(datas)

    cv.waitKey()