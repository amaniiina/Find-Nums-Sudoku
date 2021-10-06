import cv2
import numpy as np


def main():
    size = (420, 420)
    # size of sudoku grid square
    square_size = 35
    sudoku = cv2.resize(cv2.imread('sudoku.jpg', 0), size).astype('uint8')

    # number chosen: 1, cropped from sudoku image
    cropped1 = cv2.resize(cv2.imread('1.png', 0), size).astype('uint8')
    result = cv2.cvtColor(np.uint8(sudoku).copy(), cv2.COLOR_GRAY2RGB)

    kernel = np.ones((5, 5), np.uint8)
    r, thresh1 = cv2.threshold(cropped1, 107, 255, cv2.THRESH_BINARY_INV)
    thresh1 = cv2.resize(thresh1, (square_size, square_size))
    out = np.float32(thresh1)
    dst = cv2.cornerHarris(out, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    # check number of corners for digit 1 in image
    corners1 = np.sum(dst > 0.01 * dst.max())

    # crop image to get only main sudoku grid
    top_crop = 65
    bottom_crop = 35
    left_crop = 40
    right_crop = 35
    sudoku_cropped = sudoku[top_crop:sudoku.shape[0] - bottom_crop:, right_crop:sudoku.shape[1] - left_crop:]
    # blur image, apply morphological opening and blur again to clean image and get digits
    sudoku_cropped = cv2.GaussianBlur(sudoku_cropped, (7, 7), 0)
    sudoku_cropped = cv2.GaussianBlur(sudoku_cropped, (7, 7), 0)
    sudoku_cropped = cv2.morphologyEx(sudoku_cropped, cv2.MORPH_OPEN, kernel)
    sudoku_cropped = cv2.GaussianBlur(sudoku_cropped, (7, 7), 0)
    # apply inverse binary threshold on image
    thresh = cv2.adaptiveThreshold(sudoku_cropped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # offset_x and offset_y to help loop over grid squares because image is tilted
    offset_x = 15
    offset_y = -5
    # change offset_x each loop because image is tilted
    change = 2
    # resize image to square
    thresh = cv2.resize(thresh, (325, 325))
    for y in range(0, thresh.shape[0], square_size):
        offset_x = offset_x - change
        if 0 < y < 175:  # first 5 rows down
            y = y + offset_y
        if y >= 245:  # last 2 rows
            if y == 245:
                offset_x = 6
            offset_y = 6
            y = y + offset_y
        for x in range(0, thresh.shape[1], square_size):
            if offset_x > 0:
                x = x+offset_x
            if x > thresh.shape[0] or y > thresh.shape[1]:
                continue
            x2 = thresh.shape[0] if x+square_size > thresh.shape[0] else x+square_size
            y2 = thresh.shape[1] if y+square_size > thresh.shape[1] else y+square_size
            # crop current square
            square = thresh[y:y2, x:x2]
            # apply canny edge detection
            square = cv2.Canny(square, 40, 60)
            # if current square is not empty apply harris corner detection
            if y <= 175 and square.sum()/255 > 140 or y > 175 and square.sum()/255 > 135:
                digit = np.float32(square)
                dst = cv2.cornerHarris(digit, 2, 3, 0.04)
                dst = cv2.dilate(dst, None)
                num_corners = np.sum(dst > 0.01 * dst.max())
                # if num of corners of digit are approximately the same as the 1
                if corners1-25 < num_corners < corners1+25:
                    # draw circle in appropriate coordinates
                    cv2.circle(result, (x+right_crop+32, y+top_crop+15), 5, (0, 0, 255), -1)

    cv2.imshow('result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

