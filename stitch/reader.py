import os

import cv2


class Reader:
    def __init__(self, filepath1, filepath2):
        self.path1 = os.listdir(filepath1)
        self.path2 = os.listdir(filepath2)
        self.final_path1 = []
        self.final_path2 = []
        for i in range(len(self.path1)):
            self.final_path1.append(os.path.join(filepath1, self.path1[i]))
        for j in range(len(self.path2)):
            self.final_path2.append(os.path.join(filepath2, self.path2[j]))

    def get_imgs(self):
        img1s = []
        img2s = []
        for i in self.final_path1:
            img1s.append(cv2.imread(i, -1))
        for j in self.final_path2:
            img2s.append(cv2.imread(j, -1))
        return img1s, img2s
