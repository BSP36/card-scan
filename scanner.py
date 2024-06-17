import cv2
import numpy as np
import matplotlib.pyplot as plt


class Scanner:
    def __init__(self):
        pass

    def detect_edge(
            self,
            img,
            min_occupancy: float=0.05,
            max_occupancy: float=0.95,
        ):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, gray = cv2.threshold(gray, 230, 255, cv2.THRESH_TOZERO_INV)
        gray = cv2.bitwise_not(gray) # 白黒の反転
        ret, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_level = 0
        area_tot = img.shape[0] * img.shape[1]

        out_cnt = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_occupancy < area / area_tot < max_occupancy:
                epsilon = 0.01 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                cv2.drawContours(img, [approx], -1, (255, 0, 0), 1, cv2.LINE_AA, hierarchy, max_level)
                out_cnt.append(approx)

        return img, out_cnt

    def to_front(self, img, contours):
        out_imgs = []
        tot_area = img.shape[0] * img.shape[1]
        for cnt in contours:
            pts1 = np.array(cnt, dtype=np.float32).reshape([4, 2])
            pts1 = self.__rearrange_upper_left(pts1, image_size=tot_area)
            print(pts1)
            # 

            # x_min, x_max = np.min(pts1[:, 0]), np.max(pts1[:, 0])
            # y_min, y_max = np.min(pts1[:, 1]), np.max(pts1[:, 1])

            w2 = int(np.sqrt((pts1[2, 0] - pts1[1, 0]) ** 2 + (pts1[2, 1] - pts1[1, 1]) ** 2))
            h2 = int(np.sqrt((pts1[0, 0] - pts1[1, 0]) ** 2 + (pts1[0, 1] - pts1[1, 1]) ** 2))
            pts2 = np.float32([[0, 0],[0, h2],[w2, h2],[w2, 0]])

            mat = cv2.getPerspectiveTransform(pts1, pts2)
            img2 = cv2.warpPerspective(img, mat, (w2, h2), borderValue=(255,255,255))


            # `pts2` を整数座標に変換
            pts2_int = pts1.astype(int)

            # `pts2_int` を2次元リストに変換（cv2.polylines に必要な形状）
            pts2_list = pts2_int.reshape((-1, 1, 2))

            # # 長方形を描画
            cv2.polylines(img, [pts2_list], isClosed=False, color=(0, 255, 0), thickness=5)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.show()


            out_imgs.append(img2)
        
        return out_imgs
    

    def __rearrange_upper_left(self, pts, image_size):
        min_val = image_size
        min_index = 0
        reverse = True
        for i in range(4):
            v = pts[i, 0] * pts[i, 1]
            if v < min_val:
                min_val = v
                min_index = i
        if pts[min_index, 0] < pts[(min_index+1)%4, 0]:
            if (pts[min_index, 0] >= pts[(min_index+3)%4, 0]) or (pts[min_index, 1] > pts[(min_index+1)%4, 1]):
                reverse = True
            # dummy = pts[(min_index+1) % 4, :].copy()
            # pts[(min_index+1)%4, :] = pts[(min_index-1 + 4)%4, :]
            # pts[(min_index-1+4)%4, :] = dummy
            # direction = -1
        
        for _ in range(min_index):
            dummy = pts[0, :].copy()
            for i in range(3):
                pts[i, :] = pts[i+1, :]
            pts[3, :] = dummy
        
        if reverse:
            dummy = pts[1, :].copy()
            pts[1, :] = pts[3, :]
            pts[3, :] = dummy
        
        return pts



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # img = cv2.imread("./images/card1.jpg")
    img = cv2.imread("./images/card2.webp")
    sc = Scanner()
    img_detected, contours = sc.detect_edge(img)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    imgs = sc.to_front(img, contours)
    for im in imgs:
        plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        plt.show()

