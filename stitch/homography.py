import numpy as np
import math
import copy
import cv2


class Homography:

    def __init__(self):
        pass

    def normalize(self, points):
        """
        对点进行标准化
        """
        # 得到均值
        original_pts = copy.deepcopy(points)
        padding = np.ones(points.shape[0])
        u = np.mean(points, axis=0)
        pts = points - u  # 中心化

        # 标准化
        pts_square = np.square(pts)
        pts_sum = np.sum(pts_square, axis=1)
        pts_mean = np.mean(np.sqrt(pts_sum))
        scale = 1/(pts_mean + 1e-8)
        C = np.diag((scale, scale, 1))
        C[0][2] = -scale * u[0]
        C[1][2] = -scale * u[1]
        original_pts = np.column_stack((original_pts, padding))
        new_pts = C.dot(original_pts.T)
        return new_pts.T[:, :2], C

        # 通过DLT建立增广矩阵
    def augmented_matrix(self, src_pts, dest_pts):
        sample_size = src_pts.shape[0]  # 一次RANSAC采样的点数
        A = np.zeros((2*sample_size, 9))
        # DLT方程组
        for i in range(sample_size):
            A[2*i, 0] = src_pts[i, 0]
            A[2*i, 1] = src_pts[i, 1]
            A[2*i, 2] = 1
            A[2*i, 6] = -src_pts[i, 0] * dest_pts[i, 0]
            A[2*i, 7] = -src_pts[i, 1] * dest_pts[i, 0]
            A[2*i, 8] = -dest_pts[i, 0]

            A[2*i+1, 3] = src_pts[i, 0]
            A[2*i+1, 4] = src_pts[i, 1]
            A[2*i+1, 5] = 1
            A[2*i+1, 6] = -src_pts[i, 0] * dest_pts[i, 1]
            A[2*i+1, 7] = -src_pts[i, 1] * dest_pts[i, 1]
            A[2*i+1, 8] = -dest_pts[i, 1]
        return A

    def get_H(self, src_pts, dest_pts):
        npt_s, C1 = self.normalize(src_pts)
        npt_d, C2 = self.normalize(dest_pts)
        A = self.augmented_matrix(npt_s, npt_d)
        W, U, V = cv2.SVDecomp(A)  # SVD求解H矩阵
        H = V[-1, :].reshape((3, 3))
        H = np.linalg.inv(C2).dot(H).dot(C1)  # 反规一化
        H = H/H[2, 2]  # H[2,2] = 1
        return H
