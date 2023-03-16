import cv2
from SIFT import SIFTMatcher
from homography import Homography
from ransac import RANSAC
import numpy as np
from reader import Reader
import math


class Stitcher:

    def __init__(self):
        self.stable_img = None  # 填充的图片
        self.warp_img = None  # 要warp的图片
        self.H = None  # H矩阵
        self.warped = None  # warp_img warp后的图片
        self.over_width = 0  # 用于加权融合的重叠长度
        # 加权融合认为的重叠部分的边界
        self.j1 = None
        self.j2 = None

    # 进行图像拼接
    def stitch(self, img1, img2):
        self.warp_img = img1  # 哪张图片要被warp
        self.stable_img = img2
        # sift匹配，找到对应的特征点
        sift = SIFTMatcher()

        srcmatch, destmatch = sift.match(img1, img2)

        ransac = RANSAC(inlier_prob=0.6, size=10)
        ransac.iterate(srcmatch, destmatch)
        H = ransac.H

        # H, status = cv2.findHomography(
        #     srcmatch, destmatch, cv2.RANSAC, ransacReprojThreshold=5)

        # img1在右, img2在左
        result1 = cv2.warpPerspective(
            img1, H, (img1.shape[1]+img2.shape[1], img1.shape[0]))
        result1[0:img2.shape[0], 0:img2.shape[1]] = img2

        # img1在左，img2在右
        H_inv = np.linalg.inv(H)
        result2 = cv2.warpPerspective(
            img2, H_inv, (img1.shape[1]+img2.shape[1], img2.shape[0]))
        result2[0:img1.shape[0], 0:img1.shape[1]] = img1

        # 通过拼接后黑色面积大小判断左右关系是否正确，错误的位置黑色更多
        gray1 = cv2.cvtColor(result1, cv2.COLOR_BGR2GRAY)
        sum1 = np.sum((gray1 != 0))
        gray2 = cv2.cvtColor(result2, cv2.COLOR_BGR2GRAY)
        sum2 = np.sum((gray2 != 0))

        # 保证warped处于拼接后的右边
        if sum1 > sum2:
            result = result1
        else:
            result = result2
            H = H_inv
            self.warp_img, self.stable_img = self.stable_img, self.warp_img

        self.H = H
        self.j1 = self.j2 = self.stable_img.shape[1]
        return result

    # 返回重合的图像，以及重合的PSNR
    def cal_PSNR(self):

        # self.j1 = self.warp_img.shape[1] + self.stable_img.shape[1] + 2
        # self.j2 = -1

        self.warped = cv2.warpPerspective(
            self.warp_img, self.H, (self.warp_img.shape[1]+self.stable_img.shape[1], self.warp_img.shape[0]))
        # 变换视角图片的重叠部分
        overlap_warped = self.warped[:, :self.stable_img.shape[1]]
        # 没有变换视角图片的重叠部分
        overlap_stable = self.stable_img.copy()
        MSE = 0
        cnt = 0  # 统计哪些像素点是重合的
        for j in range(overlap_warped.shape[1]):
            c = 0  # 统计改列重叠部分的比例
            for i in range(overlap_warped.shape[0]):
                if any(overlap_warped[i, j, :]) == False:
                    overlap_stable[i, j, :] = [0, 0, 0]  # 不重合的地方为黑色
                else:
                    c += 1
                    pixel1 = overlap_warped[i, j, :]
                    pixel2 = overlap_stable[i, j, :]
                    MSE = MSE + np.sum((pixel1/1.0 - pixel2/1.0)**2)
                    cnt += 3
                # 当重叠>0.7，判断为重合部分的边界
                if c >= 0.7 * self.warped.shape[0]:
                    self.j1 = min(self.j1, j)
                    self.j2 = max(self.j2, j)

        MSE = MSE / cnt
        if MSE < 1e-10:
            PSNR = 100
        else:
            PSNR = 20 * math.log10(255 / math.sqrt(MSE))
        self.over_width = self.j2 - self.j1
        return overlap_warped, overlap_stable, PSNR

    def get_weight(self, d, k):
        """
        d: 重合部分的长度
        k: 权重参数
        """
        x = np.arange(-d/2, d/2)
        y = 1/(1+np.exp(-k*x))
        return y

    # 加权融合
    def blend(self, result):
        output = result
        l = self.over_width
        if l <= 0:
            return output
        # 两个拼接前的图片
        img1 = output[:, 0:self.stable_img.shape[1], :]
        img2 = self.warped[:, self.j1:output.shape[1]]
        w = self.get_weight(self.over_width, 0.05)  # 0.05经验感觉不错

        for i in range(0, l):
            t = img1.shape[1] - l + i
            for j in range(output.shape[0]):
                # 如果黑色也加权融合会导致图片信息的缺失
                # if any(img2[j, i, :]) == True:
                if img2[j, i, 0] > 0 and img2[j, i, 1] > 0 and img2[j, i, 2] > 0:
                    # if True:
                    output[j, img1.shape[1] - l + i,
                           0] = (1-w[i])*img1[j, img1.shape[1]-l + i, 0]+w[i]*img2[j, i, 0]
                    output[j, img1.shape[1] - l + i,
                           1] = (1-w[i])*img1[j, img1.shape[1]-l + i, 1]+w[i]*img2[j, i, 1]
                    output[j, img1.shape[1] - l + i,
                           2] = (1-w[i])*img1[j, img1.shape[1]-l + i, 2]+w[i]*img2[j, i, 2]
        # w_expand = np.tile(w, (output.shape[0], 1))
        # output[:, img1.shape[1] -
        #        l:img1.shape[1], 0] = (1-w_expand)*img1[:, img1.shape[1]-l:, 0]+w_expand*img2[:, :l, 0]
        # output[:, img1.shape[1] -
        #        l:img1.shape[1], 1] = (1-w_expand)*img1[:, img1.shape[1]-l:, 1]+w_expand*img2[:, :l, 1]
        # output[:, img1.shape[1] -
        #        l:img1.shape[1], 2] = (1-w_expand)*img1[:, img1.shape[1]-l:, 2]+w_expand*img2[:, :l, 2]
        return output

    def cut_black(self, result):
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        for j in range(self.stable_img.shape[1], result.shape[1]):
            if sum(gray[:, j]) > 0:
                continue
            break
        return result[:, 0:j]
