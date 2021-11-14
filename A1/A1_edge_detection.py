import cv2
import numpy as np
from A1_image_filtering import cross_correlation_2d, get_gaussian_filter_2d
from module import *

def compute_image_gradient(img):
    sobel_x = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])
    sobel_y = np.array([[-1,-2,-1], [0,0,0], [1,2,1]])
    
    dx = cross_correlation_2d(img, sobel_x)
    dy = cross_correlation_2d(img, sobel_y)

    mag = np.sqrt(dx**2 + dy**2)
    dir = np.arctan2(dy, dx)

    return mag, dir

def non_maximum_suppression_dir(mag, dir):
    dir = np.rad2deg(dir) + 180
    suppressed_mag = mag.copy()

    for i in range(1, mag.shape[0]-1):
        for j in range(1, mag.shape[1]-1):
            deg = dir[i][j]
            if (deg < 22.5) or (157.5 <= deg < 202.5) or (337.5 < deg):
                if mag[i][j] <= mag[i][j+1] or mag[i][j] < mag[i][j-1]:
                    suppressed_mag[i][j] = 0
            elif (22.5 <= deg < 67.5) or (202.5 <= deg < 247.5):
                if mag[i][j] < mag[i-1][j-1] or mag[i][j] <= mag[i+1][j+1]:
                    suppressed_mag[i][j] = 0
            elif (67.5 <= deg < 112.5) or (247.5 <= deg < 292.5):
                if mag[i][j] < mag[i-1][j] or mag[i][j] <= mag[i+1][j]:
                    suppressed_mag[i][j] = 0
            elif (112.5 <= deg < 157.5) or (292.5 <= deg < 337.5):
                if mag[i][j] <= mag[i-1][j+1] or mag[i][j] < mag[i+1][j-1]:
                    suppressed_mag[i][j] = 0

    return suppressed_mag

def mag_view(mag):
    for i in range(mag.shape[0]):
        for j in range(mag.shape[1]):
            if mag[i][j] < 0:
                mag[i][j] = 0
            elif mag[i][j] > 255:
                mag[i][j] = 255
    return mag

def main():
    lenna = load_image("lenna")
    shapes = load_image("shapes")
    size, sigma = 7, 1.5

    # lenna
    print("************lenna************")
    #2.1
    kernel = get_gaussian_filter_2d(size, sigma)
    filtered_img = cross_correlation_2d(lenna, kernel)

    #2.2
    start = time_start()
    mag, dir = compute_image_gradient(filtered_img)
    lead_time = time_end(start)
    print(f"It costs {lead_time} seconds for 'compute_image_gradient'.")
    mag = mag_view(mag)
    showsave_image(mag, "part_2_edge_raw_lenna")

    #2.3
    start = time_start()
    suppressed_mag = non_maximum_suppression_dir(mag, dir)
    lead_time = time_end(start)
    print(f"It costs {lead_time} seconds for 'non_maximum_suppression_dir'.")
    showsave_image(suppressed_mag, "part_2_edge_sup_lenna")

    # shapes
    print("***********shapes************")
    # 2.1
    kernel = get_gaussian_filter_2d(size, sigma)
    filtered_img = cross_correlation_2d(shapes, kernel)

    # 2.2
    start = time_start()
    mag, dir = compute_image_gradient(filtered_img)
    lead_time = time_end(start)
    print(f"It costs {lead_time} seconds for 'compute_image_gradient'.")
    mag = mag_view(mag)
    showsave_image(mag, "part_2_edge_raw_shapes")

    # 2.3
    start = time_start()
    suppressed_mag = non_maximum_suppression_dir(mag, dir)
    lead_time = time_end(start)
    print(f"It costs {lead_time} seconds for 'non_maximum_suppression_dir'.")
    showsave_image(suppressed_mag, "part_2_edge_sup_shapes")


if __name__ == '__main__':
    main()