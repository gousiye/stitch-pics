import math
import numpy as np
import random
from homography import Homography

# Random Sample Consensus


class RANSAC:
    def __init__(self, inlier_prob=0.5, model_prob=0.995, threshold=20, size=10, optimal_k=True, max_k=1000):
        """
        inlier_prob: 一个点为内点的概率
        model_prob: RANSAC模型得到正确模型的概率
        thresold: 判断为内点的阈值
        size: 一次枚举的点的数量
        optimal_k: 是否要根据概率计算最优的迭代次数
        max_k: 最大的迭代次数
        """
        self.prob = inlier_prob
        self.M_prob = model_prob
        self.threshold = threshold
        self.size = size
        self.cal_k = optimal_k
        self.H = None

        if self.cal_k:
            self.N = max(self.get_N(), max_k)
            # print(self.prob)
        else:
            self.N = max_k

    def get_N(self):
        n = math.log(1-self.M_prob)/math.log(1 -
                                             math.pow(self.prob, self.size) - 1e-8)
        # n = math.log(1-self.M_prob)/math.log(1 -
        #                                    math.pow((1-self.prob), self.size) - 1e-8)
        n = int(np.round(n))
        return n
        # return int(round(math.log(1 - self.prob) / math.log(1 - math.pow((1 - self.inlier_p), self.m) + 1e-8)))

    def cal_distance(self, src_point, dest_point, H):
        """
        计算H映射后的点与原本dest中点的距离
        src_point: 原图中该点的位置
        dest_point: 拼接图中该点的位置
        H: 当前抽样下的H矩阵
        """
        x1, y1 = src_point
        x2, y2 = dest_point
        src_vector = np.transpose(np.array([x1, y1, 1.]))
        f_vector = np.dot(H, src_vector)  # 点经过H映射后的坐标
        f_vector /= (f_vector[2] + 1e-8)  # 齐次坐标归一化

        dest_vector = np.transpose(np.array([x2, y2, 1.]))
        error = f_vector - dest_vector
        return np.linalg.norm(error)  # 计算距离

    def iterate(self, src, dest):
        """
        迭代抽样寻找最优模型
        src: 原图中所有特征点
        dest: 目标图中所有特征点
        """
        # 记录最大内点的点集
        smax_inliers = []
        dmax_inliers = []

        points_range = list(np.arange(len(src)))
        h = Homography()

        for i in range(self.N):
            # 随机抽取点
            index = random.sample(points_range, self.size)

            # 选择到的点集
            chosen_src = src[index, :]
            chosen_dst = dest[index, :]

            H = h.get_H(chosen_src, chosen_dst)
            # print(H)
            # 记录内点数
            src_inliers = []
            dst_inliers = []

            for s, d in zip(src, dest):
                distance = self.cal_distance(s, d, H)
                if distance <= self.threshold:
                    src_inliers.append(s)
                    dst_inliers.append(d)
            if len(src_inliers) > len(smax_inliers):
                smax_inliers = src_inliers
                dmax_inliers = dst_inliers
                self.H = H
