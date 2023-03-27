import cv2
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='SLIC')
    parser.add_argument('filename', help="path of the image")
    parser.add_argument('-i', type=int, default=10, help="times of iteration")
    args = parser.parse_args()

    return args

def main():
    # Load image
    args = parse_args()
    img = cv2.imread(args.filename)

    # Apply SLIC algorithm
    slic = cv2.ximgproc.createSuperpixelSLIC(img)
    slic.iterate(args.i)

    mask_slic = slic.getLabelContourMask()
    # label_slic = slic.getLabels()
    # number_slic = slic.getNumberOfSuperpixels()
    mask_inv_slic = cv2.bitwise_not(mask_slic)
    img_slic = cv2.bitwise_and(img, img, mask = mask_inv_slic)
    cv2.imwrite("output_API_" + args.filename, img_slic)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__" :
    main()