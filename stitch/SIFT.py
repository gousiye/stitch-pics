import cv2
import numpy as np


class SIFTMatcher:
    """
    便于进行SIFT匹配特征点
    """

    def __init__(self):
        self.matcher = cv2.BFMatcher()
        self.sift = cv2.SIFT_create()
        self.raws = []
        self.goods = []

    def knn_match(self, desc1, desc2, k=2):
        """
        KNN匹配, default: k = 2 
        """
        self.raws = self.matcher.knnMatch(desc1, desc2, k)
        return self.raws

    def good_matching(self, ratio=0.75):
        """
        过滤符号的匹配, default ratio = 0.75
        """
        self.goods = []
        for p1s, p2s in self.raws:
            if p1s.distance < p2s.distance * ratio:
                self.goods.append((p1s.trainIdx, p2s.queryIdx))
        return self.goods

    def get_points(self, kp1, kp2):
        assert len(self.goods) > 4  # 至少4对特征点才能解出方程
        # 获取配对点的坐标
        src_match = np.float32([kp1[i].pt for _, i in self.goods])
        dst_match = np.float32([kp2[i].pt for i, _ in self.goods])
        return src_match, dst_match

    def detect(self, img):
        """
        SIFTMatch
        """
        return self.sift.detectAndCompute(img, None)

    def match(self, img1, img2, ratio=0.75):
        # gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        # gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        kp1, desc1 = self.detect(img1)
        kp2, desc2 = self.detect(img2)
        self.knn_match(desc1, desc2)
        self.good_matching(ratio)
        src_match, dst_match = self.get_points(kp1, kp2)
        return src_match, dst_match
