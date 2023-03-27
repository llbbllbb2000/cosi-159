import argparse
import cv2
import numpy as np
import math

def parse_args():
    parser = argparse.ArgumentParser(description='SLIC')
    # parser.add_argument('', type=, default=10, help="number of superpixels")
    parser.add_argument('filename', help="path of the image")
    parser.add_argument('-k', type=int, default=200, help="number of superpixels")
    parser.add_argument('-i', type=int, default=10, help="times of iteration")
    parser.add_argument('-m', type=int, default=10, help="the constant m")
    args = parser.parse_args()

    return args

def main() :
    args = parse_args()
    img = cv2.imread(args.filename)
    lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)

    height, width, _ = lab_img.shape
    step_size = int(np.sqrt((height*width) / args.k))

    label = np.full((height, width), -1)
    dis = np.full((height, width), np.inf)

    soberX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    soberY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    def calc_gradient(h, w) :
        if h - 1 < 0 or h + 1 >= height or \
            w - 1 < 0 or w + 1 >= width:
            return np.inf
        
        window = lab_img[h - 1: h + 2, w - 1 : w + 2]
        Gx = np.sum(window * soberX)
        Gy = np.sum(window * soberY)

        return Gx * Gx + Gy * Gy
    
    # find the lowest gradient position in a 3 * 3 neighborhood
    def lowest_gradient(h, w) :
        tempX = 0
        tempY = 0
        tempG = np.inf

        for x in range(h - 1, h + 2, 1) :
            if x < 0 or x >= height :
                continue

            for y in range(w - 1, w + 2, 1):
                if y < 0 or y >= width :
                    continue

                G = calc_gradient(x, y)

                if G < tempG :
                    tempX = x
                    tempY = y

        return tempX, tempY
    
    centers = []
    for i in range(step_size // 2, height, step_size) :
        for j in range(step_size // 2, width, step_size) :
            x, y = lowest_gradient(i, j)            
            center = [lab_img[x, y, 0], lab_img[x, y, 1], lab_img[x, y, 2], x, y]
            centers.append(center)

    for i in range(args.i):
        for center_index in range(len(centers)):
            center = centers[center_index]
            x_min = max(math.ceil(center[3]-step_size), 0)
            x_max = min(math.floor(center[3]+step_size), height - 1)
            y_min = max(math.ceil(center[4]-step_size), 0)
            y_max = min(math.floor(center[4]+step_size), width - 1)

            for x in range(x_min, x_max + 1, 1) :
                for y in range(y_min, y_max + 1, 1) :
                    dc = np.sum(np.square(lab_img[x, y] - center[:3]))
                    ds = np.sum(np.square(np.array([x, y]) - center[3:]))
                    D = np.sqrt(dc * dc + np.square(ds / step_size * args.m))
                    if D < dis[x, y] :
                        dis[x, y] = D
                        label[x, y] = center_index

        for center_index in range(len(centers)) :
            center_points = lab_img[label == center_index]
            if len(center_points) > 0:
                centers[center_index][:3] = np.mean(center_points, axis=0)


    
    
    SLIC_img = np.copy(lab_img)

    # get the image based on the SLIC
    for x in range(height) :
        for y in range(width) :
            SLIC_img[x, y] = centers[label[x, y]][:3]

    cv2.imwrite("output_" + args.filename, cv2.cvtColor(SLIC_img, cv2.COLOR_Lab2RGB))
 
if __name__ == "__main__" :
    main()