import cv2
import numpy as np
import math
from module import *

def padding_2d(img, kernel_size):
    pad_size = (kernel_size - 1) // 2
    pad_output = np.zeros((img.shape[0] + 2*pad_size, img.shape[1] + 2*pad_size))

    h, w = pad_output.shape[0], pad_output.shape[1]

    #가운데
    pad_output[pad_size:h-pad_size, pad_size:w-pad_size] = img[:, :].copy()

    #위쪽
    pad_output[0:pad_size, 0:pad_size] = img[0][0]
    pad_output[0:pad_size, pad_size:w-pad_size] = img[0, :].copy()
    pad_output[0:pad_size, w-pad_size:w] = img[0][-1]

    #아래쪽
    pad_output[h-pad_size:, 0:pad_size] = img[-1][0]
    pad_output[h-pad_size:, pad_size:w-pad_size] = img[-1, :].copy()
    pad_output[h-pad_size:, w-pad_size:w] = img[-1][-1]

    #왼쪽 오른쪽
    for i in range(pad_size):
        pad_output[pad_size:h-pad_size, i] = img[:, 0].copy()
        pad_output[pad_size:h-pad_size, -1-i] = img[:, -1].copy()

    return pad_output

def cross_correlation_1d(img, kernel):
    h, w = img.shape[0], img.shape[1]
    k = kernel.shape[0]
    output = np.zeros((h, w))

    if kernel.ndim == 1: #horizontal
        # padding_1d
        pad_size = (k - 1) // 2
        pad_img = np.zeros((h + 2 * pad_size, w))
        for p in range(pad_size):
            pad_img[p:p + 1, :] = img[0, :].copy()
            pad_img[h + p:h + p + 1, :] = img[-1, :].copy()
        pad_img[pad_size:pad_size+h, :] = img[:, :].copy()

        kernel = kernel.reshape(1, kernel.shape[0])
        for i in range(h):
            for j in range(w):
                compute_range = pad_img[i:i + k, j]
                compute = np.multiply(compute_range, kernel)
                output[i][j] = compute.sum()

    else: #vertical
        #padding_1d
        pad_size = (k - 1) // 2
        pad_img = np.zeros((h, w + 2 * pad_size))
        for p in range(pad_size):
            pad_img[:, p:p + 1] = img[:, 0].reshape(h,1).copy()
            pad_img[:, w + p:w + p + 1] = img[:, -1].reshape(h,1).copy()
        pad_img[:, pad_size:pad_size+w] = img[:, :].copy()

        kernel = kernel.reshape(1, kernel.shape[0])
        for i in range(h):
            for j in range(w):
                compute_range = pad_img[i, j:j + k]
                compute = np.multiply(compute_range, kernel)
                output[i][j] = compute.sum()

    return output

def cross_correlation_2d(img, kernel):
    output = np.zeros((img.shape[0], img.shape[1]))
    h, w = kernel.shape[0], kernel.shape[1]

    pad_img = padding_2d(img, kernel.shape[0])

    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            compute_range = pad_img[i:i+h, j:j+w]
            compute = np.multiply(compute_range, kernel)
            output[i][j] = compute.sum()

    return output

def get_gaussian_filter_1d(size, sigma):
    output = np.ones(size)
    s = size // 2
    for i in range(size):
        x = i - s
        output[i] = (1 / ((2*math.pi)**(1/2) * sigma)) * np.exp(-((x**2)/(2*(sigma**2))))
    output = output / output.sum()
    return output

def get_gaussian_filter_2d(size, sigma):
    output = np.ones((size, size))
    s = size // 2
    for i in range(size):
        for j in range(size):
            x = i - s
            y = j - s
            output[i][j] = (1 / (2 * math.pi * (sigma**2))) * np.exp(-((x**2 + y**2)/(2*(sigma**2))))
    output = output / output.sum()
    return output

def image_9(img):
    h, w = img.shape[0], img.shape[1]
    output = np.zeros((h * 3, w * 3))

    for i, size in enumerate([5, 11, 17]):
        for j, sigma in enumerate([1, 6, 11]):
            org = (j*w+12, i*h+32)
            font = cv2.FONT_HERSHEY_SIMPLEX
            kernel_2d = get_gaussian_filter_2d(size, sigma)
            filtered_img_2d = cross_correlation_2d(img, kernel_2d)

            output[i*h:(i+1)*h, j*w:(j+1)*w] = filtered_img_2d[:, :].copy()
            cv2.putText(output, f"{size}x{size} s={sigma}", org, font, 1, (0,0,0), 2)

    return output

def different_1d_2d(img, size, sigma):
    kernel_1d = get_gaussian_filter_1d(size, sigma)
    start = time_start()
    filtered_img_1d = cross_correlation_1d(img, kernel_1d)
    filtered_img_1d = cross_correlation_1d(filtered_img_1d, kernel_1d.reshape(size, 1))
    time_1d = time_end(start)

    kernel_2d = get_gaussian_filter_2d(size, sigma)
    start = time_start()
    filtered_img_2d = cross_correlation_2d(img, kernel_2d)
    time_2d = time_end(start)

    different = cv2.subtract(filtered_img_2d, filtered_img_1d)
    different = np.abs(different).sum()
    print(f"Difference between 1d and 2d : {different}")
    print(f"1d time : {time_1d}, 2d time : {time_2d}")


def main():
    lenna = load_image("lenna")
    shapes = load_image("shapes")

    #1-2. c)
    print("++++++Gaussian filter result 1d (size=5, sigma=1)++++++")
    print(get_gaussian_filter_1d(5, 1))
    print("++++++Gaussian filter result 2d (size=5, sigma=1)++++++")
    print(get_gaussian_filter_2d(5, 1))

    #1-2. d)
    img9_lenna = image_9(lenna)
    showsave_image(img9_lenna, "part_1_gaussian_filtered_lenna")

    img9_shapes = image_9(shapes)
    showsave_image(img9_shapes, "part_1_gaussian_filtered_shapes")

    #1-2. e)
    size, sigma = 11, 6
    print("++++++++++++lenna++++++++++++")
    different_1d_2d(lenna, size, sigma)
    print("+++++++++++shapes++++++++++++")
    different_1d_2d(shapes, size, sigma)


if __name__ == '__main__':
    main()