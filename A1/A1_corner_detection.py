import cv2
import numpy as np
from A1_image_filtering import cross_correlation_2d, get_gaussian_filter_2d
from module import *

def compute_corner_response(img):
    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    dx = cross_correlation_2d(img, sobel_x)
    dy = cross_correlation_2d(img, sobel_y)

    ixix = dx ** 2
    ixiy = dx * dy
    iyiy = dy ** 2

    window = np.ones((5, 5))
    S_x2 = cross_correlation_2d(ixix, window)
    S_xy = cross_correlation_2d(ixiy, window)
    S_y2 = cross_correlation_2d(iyiy, window)

    R = (S_x2 * S_y2 - S_xy ** 2) - 0.04 * ((S_x2 + S_y2) ** 2)

    R[R < 0] = 0
    R /= np.max(R)

    return R

def non_maximum_suppression_win(R, winSize):
    size = winSize // 2
    suppressed_R = np.zeros((R.shape[0], R.shape[1]))

    for i in range(size, R.shape[0]-size):
        for j in range(size, R.shape[1]-size):
            window = R[i-size:i+size+1, j-size:j+size+1]
            w_max = window.max()
            if w_max > 0.1 and w_max == R[i][j]:
                suppressed_R[i][j] = R[i][j]

    return suppressed_R

def draw_circle(img, point_img):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    point = np.where(point_img > 0)

    for x, y in zip(point[1], point[0]):
        img = cv2.circle(img, (x, y), 5, (0, 255, 0), 2)

    return img

def main():
    lenna = load_image("lenna")
    shapes = load_image("shapes")
    size, sigma = 7, 1.5

    # lenna
    print("************lenna************")
    # 3.1
    kernel = get_gaussian_filter_2d(size, sigma)
    filtered_img = cross_correlation_2d(lenna, kernel)

    # 3.2
    start = time_start()
    R = compute_corner_response(filtered_img)
    lead_time = time_end(start)
    print(f"It costs {lead_time} seconds for 'compute_corner_response'.")
    showsave_image(R * 255, "part_3_corner_raw_lenna")

    # 3.3 a) b)
    color_img = cv2.cvtColor(lenna, cv2.COLOR_GRAY2BGR)
    color_img[R > 0.1] = [0, 255, 0]
    showsave_image(color_img, "part_3_corner_bin_lenna")
    # 3.3 c) d)
    winSize = 11
    start = time_start()
    suppressed_R = non_maximum_suppression_win(R, winSize)
    lead_time = time_end(start)
    print(f"It costs {lead_time} seconds for 'non_maximum_suppression_win'.")
    circle_img = draw_circle(lenna, suppressed_R)
    showsave_image(circle_img, "part_3_corner_sup_lenna")

    #shapes
    print("***********shapes************")
    # 3.1
    kernel = get_gaussian_filter_2d(size, sigma)
    filtered_img = cross_correlation_2d(shapes, kernel)

    #3.2
    start = time_start()
    R = compute_corner_response(filtered_img)
    lead_time = time_end(start)
    print(f"It costs {lead_time} seconds for 'compute_corner_response'.")
    showsave_image(R*255, "part_3_corner_raw_shapes")

    #3.3 a) b)
    color_img = cv2.cvtColor(shapes, cv2.COLOR_GRAY2BGR)
    color_img[R > 0.1] = [0, 255, 0]
    showsave_image(color_img, "part_3_corner_bin_shapes")
    #3.3 c) d)
    winSize = 11
    start = time_start()
    suppressed_R = non_maximum_suppression_win(R, winSize)
    lead_time = time_end(start)
    print(f"It costs {lead_time} seconds for 'non_maximum_suppression_win'.")
    circle_img = draw_circle(shapes, suppressed_R)
    showsave_image(circle_img, "part_3_corner_sup_shapes")


if __name__ == '__main__':
    main()