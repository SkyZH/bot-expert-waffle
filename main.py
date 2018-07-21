import numpy as np
import cv2
from operator import itemgetter

_bound = {
    "hsv_red": (np.array([0, 101, 67]), np.array([20, 255, 255])),
    "hsv_yellow": (np.array([20, 142, 67]), np.array([90, 255, 255])),
    "hsv_blue": (np.array([80, 69, 67]), np.array([130, 255, 255])),
}

import math
def get_angle(p1, p2, p3):
    try:
        vec1 = (p1[0] - p2[0], p1[1] - p2[1])
        vec2 = (p3[0] - p2[0], p3[1] - p2[1])
        rad = math.acos((vec1[0] * vec2[0] + vec1[1] * vec2[1])
            / math.sqrt(vec1[0] * vec1[0] + vec1[1] * vec1[1])
            / math.sqrt(vec2[0] * vec2[0] + vec2[1] * vec2[1]))
        return rad / math.pi * 180
    except ZeroDivisionError:
        return 0


class OpenCvCapture(object):
    def __init__(self):
        for i in range(10):
            print("Testing for presense of camera #{0}...".format(i))
            cv2_cap = cv2.VideoCapture(i)
            if cv2_cap.isOpened():
                break

        if not cv2_cap.isOpened():
            print("Camera not found!")
            exit(1)

        self.cv2_cap = cv2_cap

    def show_video(self):
        while True:
            ret, img = self.cv2_cap.read()

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # Extract Red, Yellow and Blue from Image
            red_area = cv2.inRange(hsv, _bound["hsv_red"][0], _bound["hsv_red"][1])
            yellow_area = cv2.inRange(hsv, _bound["hsv_yellow"][0], _bound["hsv_yellow"][1])
            blue_area = cv2.inRange(hsv, _bound["hsv_blue"][0], _bound["hsv_blue"][1])

            # Combind all three colors to find the triangle
            processed_area = cv2.bitwise_or(cv2.bitwise_or(red_area, yellow_area), blue_area)
            processed_area = cv2.blur(processed_area, (5, 5))

            _, contours, _ = cv2.findContours(
                processed_area, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            coordinates = []
            avg_area = [cv2.contourArea(cnt) for cnt in contours]

            for cnt in contours:
                # Simplify polygens and find triangles
                approx = cv2.approxPolyDP(
                    cnt, 0.03 * cv2.arcLength(cnt, True), True)

                if len(approx) == 3:
                    points = np.array(approx[:,0])
                    # Find the max angle
                    angles = enumerate([get_angle(points[1], points[0], points[2]), get_angle(points[0], points[1], points[2]), get_angle(points[0], points[2], points[1])])
                    max_angle = max(angles, key=itemgetter(1))
                    if 75 <= max_angle[1] and max_angle[1] <= 105:
                        coordinates.append([cnt])
                        cv2.drawContours(img, [cnt], 0, (0, 0, 255), 3)
                        # Get the top point and the middle of bottom segment
                        top = points[max_angle[0]]
                        bottom = (points[0] + points[1] + points[2] - top) / 2
                        colors = []
                        lst_color = ''
                        for (x, y) in zip(np.linspace(top[0], bottom[0], 10), np.linspace(top[1], bottom[1], 10)):
                            x = int(x)
                            y = int(y)
                            color = 'r' if red_area[y, x] >= 128 else 'b' if blue_area[y, x] >= 128 else 'y' if yellow_area[y, x] >= 128 else ''
                            if color != '' and color != lst_color:
                                colors.append(color)
                                lst_color = color
                        direction = bottom - top

                        # Print color combinition and steering degree
                        print(colors, np.angle(direction[0] + direction[1] * 1j, deg=True))

            cv2.imshow('PROCESSED', cv2.resize(processed_area, (384, 216)))
            cv2.imshow('Y', cv2.resize(yellow_area, (384, 216)))
            cv2.imshow('B', cv2.resize(blue_area, (384, 216)))
            cv2.imshow('R', cv2.resize(red_area, (384, 216)))
            cv2.imshow('CAPTURE', img)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        cv2.destroyAllWindows()

if __name__ == '__main__':
    OpenCvCapture().show_video()
