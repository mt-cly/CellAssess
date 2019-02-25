import cfg
import cv2 as cv
import morphsnakes as snakes
from matplotlib import pyplot as plt
import numpy as np


def get_thresh_by_proportion(image, proportion=0.005):
    v_max = 256
    num = image.size * proportion
    counts = [0] * v_max
    for i in image.flatten():
        counts[i] += 1
    for i in range(v_max):
        num = num - counts[i]
        if num < 0:
            return i


def get_image_channel(image, channel):
    if channel == -1:
        _image = image.copy()
        return cv.cvtColor(image, _image, cv.COLOR_BGR2GRAY)
    return cv.split(image)[channel]


def visual_callback_2d(background, fig=None):
    # Prepare the visual environment.
    if fig is None:
        fig = plt.figure()
    fig.clf()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(background, cmap=plt.cm.gray)

    ax2 = fig.add_subplot(1, 2, 2)
    ax_u = ax2.imshow(np.zeros_like(background), vmin=0, vmax=1)
    plt.pause(0.001)

    def callback(levelset):

        if ax1.collections:
            del ax1.collections[0]
        ax1.contour(levelset, 0.3, colors='r')
        ax_u.set_data(levelset)
        fig.canvas.draw()
        plt.pause(0.001)

    return callback


def dilate_erode_image(image, iteration=1):
    for i in range(iteration):
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        cv.dilate(image, kernel, image)
        cv.erode(image, kernel, image)
    return image


def divide_with_hull(image):
    mask = image.copy()
    mask[mask > 0] = 255
    last_hulls = 0
    last_contours = 0
    index = 0
    while True:
        _, contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        hulls = [cv.convexHull(contour) for contour in contours]
        contours_area = sum([cv.contourArea(contour) for contour in contours])
        hulls_area = sum([cv.contourArea(hull) for hull in hulls])
        if abs(hulls_area - last_hulls) < 3 and abs(contours_area - last_contours) < 3:
            break
        last_hulls = hulls_area
        last_contours = contours_area
        for (contour, hull) in zip(contours, hulls):
            # ============ for each object ===================================
            boundary = mask.copy()
            boundary[:] = 0
            cv.drawContours(boundary, [hull], 0, (255, 255, 255), -1)
            cv.drawContours(boundary, [contour], 0, (0, 0, 0), -1)
            # ===================== get this object's contours ================
            _, temp_contours, _ = cv.findContours(boundary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            if len(temp_contours) < 2:
                continue
            # ======= pick the max 2 areas (big enough) for calculating the closed points pair ================
            temp_contours = sorted(temp_contours, key=lambda x: cv.contourArea(x), reverse=True)
            if cv.contourArea(temp_contours[0]) < cfg.T or cv.contourArea(temp_contours[1]) < cfg.T:
                continue
            # cv.imshow("contours_{}".format(index), boundary)
            cv.imwrite("handled_data\contours_{}.jpg".format(index), boundary)
            contour_1, contour_2 = temp_contours[0].reshape(-1, 2), temp_contours[1].reshape(-1, 2)
            min_distance = 1e4
            pnt_best = []
            for pnt_1 in contour_1:
                for pnt_2 in contour_2:
                    distance = (pnt_1[0] - pnt_2[0]) ** 2 + (pnt_1[1] - pnt_2[1]) ** 2
                    if distance < min_distance:
                        min_distance = distance
                        pnt_best.append([pnt_1, pnt_2])
            # ===== update image ============
            print(tuple(pnt_best[-1][0]), tuple(pnt_best[-1][1]))
            boundary[:] = 0
            cv.drawContours(boundary, temp_contours[0:2], -1, (255, 0, 0), 0)
            boundary = cv.line(boundary, tuple(pnt_best[-1][0]), tuple(pnt_best[-1][1]), (255, 0, 0), thickness=2)
            # cv.imshow("after_contours_{}".format(index), boundary)
            cv.imwrite("handled_data/after_contours_{}.jpg".format(index), boundary)
            mask = cv.line(mask, tuple(pnt_best[-1][0]), tuple(pnt_best[-1][1]), (0, 0, 0), thickness=2)
            image[mask == 0] = 0
            # cv.imshow("divided_{}".format(index), image)
            cv.imwrite("handled_data/divided_{}.jpg".format(index), image)
            index += 1
    return image


def sift_cell(cell, nucleus):
    image = cell.copy()  # result
    mask_cell = cell.copy()
    mask_cell[cell > 0] = 1
    mask_nucleus = nucleus.copy()
    mask_nucleus[nucleus > 0] = 1
    _, cell_contours, _ = cv.findContours(mask_cell, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for cell_contour in cell_contours:
        if cv.contourArea(cell_contour) > cfg.Ta_max or cv.contourArea(cell_contour) < cfg.Ta_min:
            image = cv.drawContours(image, [cell_contour], -1, (0, 0, 0), -1)  # remove
            continue
        temp_image = cell.copy()
        temp_image[:] = 0
        temp_image = cv.drawContours(temp_image, [cell_contour], -1, (255, 0, 0), -1)
        temp_image[mask_nucleus == 0] = 0
        P = temp_image.sum() / 255. / cv.contourArea(cell_contour)
        if P > cfg.Tp_max or P < cfg.Tp_min:
            image = cv.drawContours(image, [cell_contour], -1, (0, 0, 0), -1)  # remove
    return image


def check_image(path):
    image = cv.imread(path)
    image_channel = get_image_channel(image, cfg.NUCLEUS_BG_CHANNEL)

    # =============== get nucleus =============
    image_init_nucleus = image_channel.copy()
    thresh = get_thresh_by_proportion(image_init_nucleus)
    image_init_nucleus[image_init_nucleus > thresh] = 0
    cv.imshow("init_nucleus", image_init_nucleus)
    cv.imwrite("handled_data/init_nucleus.jpg", image_init_nucleus)

    # ============== set as level-set =============
    ls_init_nucleus = image_init_nucleus.copy()
    ls_init_nucleus[ls_init_nucleus > 0] = 1

    # ============== find (cell) boundary with snake ============
    ls_init_cell = snakes.morphological_chan_vese(image_channel, iterations=cfg.SUB_ITERATION,
                                                  init_level_set=ls_init_nucleus,
                                                  smoothing=cfg.SMOOTH, lambda1=cfg.LAMBDA_1, lambda2=cfg.LAMBDA_2)
    # ls_init_cell = snakes.morphological_geodesic_active_contour(image_channel, iterations=cfg.SUB_ITERATION,smoothing=0,
    #                                               init_level_set=ls_init_nucleus, iter_callback=visual_callback_2d(image_channel))
    image_init_cell = image_channel.copy()
    image_init_cell[ls_init_cell == 0] = 0
    # cv.imshow("init_cell", image_init_cell)
    cv.imwrite("handled_data/init_cell.jpg", image_init_cell)

    # ===================== iterations =========================
    ls_cell = ls_init_cell
    ls_nucleus = ls_init_nucleus
    image_nucleus = image_channel.copy()
    image_cell = image_channel.copy()
    for i in range(cfg.ITERATION):
        # ========== get more precise nucleus ===================
        mask = ls_cell.copy()
        ls_nucleus = snakes.morphological_chan_vese_mask(image_channel, mask=mask, iterations=cfg.SUB_ITERATION,smoothing=cfg.SMOOTH,
                                                         init_level_set=ls_nucleus)
        # ls_nucleus = snakes.morphological_geodesic_active_contour(image_channel, iterations=cfg.SUB_ITERATION,
        #                                                     init_level_set=ls_nucleus)
        image_nucleus[ls_nucleus == 0] = 0
        # cv.imshow("image_nucleus_{}".format(i), image_nucleus)
        cv.imwrite("handled_data/image_nucleus_{}.jpg".format(i), image_nucleus)

        # =============== get more precise cell ==================
        mask = 1 - ls_nucleus
        ls_cell = snakes.morphological_chan_vese_mask(image_channel, mask=mask, iterations=cfg.SUB_ITERATION,smoothing=cfg.SMOOTH,
                                                      init_level_set=ls_cell)
        # ls_cell = snakes.morphological_geodesic_active_contour(image_channel,  iterations=cfg.SUB_ITERATION,
        #                                               init_level_set=ls_cell)
        ls_cell[ls_nucleus > 0] = 1
        image_cell[ls_cell == 0] = 0
        # cv.imshow("image_cell_{}".format(i), image_cell)
        cv.imwrite("handled_data/image_cell_{}.jpg".format(i), image_cell)

    # =================== ready for find contours ==========================
    image_cell = dilate_erode_image(image_cell, 0)
    cv.imshow("erode_dilate", image_cell)

    # =================== find contours and divide ===========================
    image_cell = divide_with_hull(image_cell)
    cv.imshow("divided", image_cell)
    # =================== calculate result ===============================
    image_cell = sift_cell(image_cell, image_nucleus)
    cv.imshow("result", image_cell)
    cv.imwrite("handled_data/result.jpg", image_cell)

    level = image_cell.copy()
    level[level > 0] = 1
    _, contours, _ = cv.findContours(level, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    image = cv.drawContours(image, contours, -1, (255, 0, 0), 1)
    print("cell area:{}".format(level.sum()))
    level[ls_nucleus==0]=0
    _, contours, _ = cv.findContours(level, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    image = cv.drawContours(image, contours, -1, (255, 255, 255), 1)
    cv.imshow("final_result", image)
    cv.imwrite("handled_data/final_result.jpg", image)
    print("nucleus area:{}".format(level.sum()))


if __name__ == "__main__":
    check_image(cfg.IMG_3_PATH)
    cv.waitKey()
