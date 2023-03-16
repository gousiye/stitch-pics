import cv2
from reader import Reader
import os
from stitcher import Stitcher
import numpy as np

#Output_test = "./Output_test"
#PSNR_test = "./PSNR_test"

Output_pic = "./Output"
Output_PSNR = "./PSNR"
PSNR_list = []  # 先将所有的PSNR记录到一个列表中，最后再保存
reader = Reader("./Pics/input1", "./Pics/input2")
img1s, img2s = reader.get_imgs()


stitcher = Stitcher()

for i in range(len(img1s)):
    img1 = img1s[i]
    img2 = img2s[i]

    # 进行图像拼接
    result = stitcher.stitch(img1, img2)

    _, _, psnr = stitcher.cal_PSNR()
    PSNR_list.append(psnr)

    print(reader.path1[i] + " is complished     PSNR: " + str(PSNR_list[-1]))

    result = stitcher.blend(result)
    result = stitcher.cut_black(result)

    # 记录拼接后的图片
    if not os.path.exists(Output_pic):
        os.makedirs(Output_pic)

    # stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    # (retval, pano) = stitcher.stitch([img1, img2])

    cv2.imwrite(os.path.join(Output_pic, reader.path1[i]), result)


PSNR_list.append(np.mean(PSNR_list))

# 将PSNR写入文件:
if not os.path.exists(Output_PSNR):
    os.makedirs(Output_PSNR)
with open(os.path.join(Output_PSNR, "PSNR.txt"), "w") as f:
    for i in range(len(PSNR_list)-1):
        f.write("PSNR of group " +
                reader.path1[i] + " is : " + str(PSNR_list[i]) + '\n')
    f.write("Average PSNR is : " + str(PSNR_list[-1]))


# test

# img1 = cv2.imread('./Pics/input1/02.jpg', -1)
# img2 = cv2.imread('./Pics/input2/02.jpg', -1)
# #img2 = stitcher.blend(img1, img2)

# result = stitcher.stitch(img1, img2)
# overlap_warped, overlap_stable, PSNR = stitcher.cal_PSNR()

# # 加权融合，消除拼接缝
# result = stitcher.blend(result)
# result = stitcher.cut_black(result)

# # stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
# # (retval, pano) = stitcher.stitch([img1, img2])

# cv2.imshow("1", overlap_warped)
# cv2.imshow("2", overlap_stable)
# cv2.imshow("result", result)
# print(PSNR)


# cv2.waitKey(0)
# cv2.destroyAllWindows()
