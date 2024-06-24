import cv2
import numpy as np
import matplotlib.pyplot as plt


class CardScanner:
    def __init__(self):
        pass

    def detect_card_contours(
            self,
            img,
            min_occupancy: float = 0.05,
            max_occupancy: float = 0.95,
        ):
        """
        Detects edges in the image and finds contours that likely represent cards.

        Args:
            img (numpy.ndarray): The input image.
            min_occupancy (float): Minimum area occupancy ratio to consider a contour.
            max_occupancy (float): Maximum area occupancy ratio to consider a contour.

        Returns:
            tuple: Processed image with drawn contours and a list of detected contours.
        """
        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply threshold to convert image to binary format
        ret, gray = cv2.threshold(gray, 230, 255, cv2.THRESH_TOZERO_INV)

        # Invert colors to highlight the card's edges
        gray = cv2.bitwise_not(gray)

        # Use Otsu's method to binarize the image
        ret, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Find contours in the binary image
        contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_level = 0
        area_tot = img.shape[0] * img.shape[1]  # Total area of the image

        out_cnt = []
        for cnt in contours:
            area = cv2.contourArea(cnt)  # Calculate the area of the contour
            # Check if the contour's area is within the specified range
            if min_occupancy < area / area_tot < max_occupancy:
                epsilon = 0.01 * cv2.arcLength(cnt, True)  # Approximation precision
                approx = cv2.approxPolyDP(cnt, epsilon, True)  # Approximate contour to polygon
                # Draw the approximated contour on the image
                cv2.drawContours(img, [approx], -1, (255, 0, 0), 1, cv2.LINE_AA, hierarchy, max_level)
                out_cnt.append(approx)

        return img, out_cnt

    def transform_to_front_view(self, img, contours):
        """
        Applies perspective transformation to the detected contours to get a front view.

        Args:
            img (numpy.ndarray): The input image.
            contours (list): List of contours to be transformed.

        Returns:
            list: List of images with front view of detected cards.
        """
        out_imgs = []
        for cnt in contours:
            # Convert contour to a numpy array of float32
            pts1 = np.array(cnt, dtype=np.float32).reshape([4, 2])
            pts1 = self.__order_points_counterclockwise(pts1)  # Ensure points are in counterclockwise order

            # Calculate width and height for the new perspective
            w2 = int(np.sqrt((pts1[2, 0] - pts1[1, 0]) ** 2 + (pts1[2, 1] - pts1[1, 1]) ** 2))
            h2 = int(np.sqrt((pts1[0, 0] - pts1[1, 0]) ** 2 + (pts1[0, 1] - pts1[1, 1]) ** 2))
            pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]])

            # Compute perspective transform matrix
            mat = cv2.getPerspectiveTransform(pts1, pts2)
            # Apply the perspective transform
            img2 = cv2.warpPerspective(img, mat, (w2, h2), borderValue=(255, 255, 255))

            # # Draw the perspective rectangle on the original image for visualization
            # pts2_int = pts1.astype(int)
            # pts2_list = pts2_int.reshape((-1, 1, 2))
            # cv2.polylines(img, [pts2_list], isClosed=False, color=(0, 255, 0), thickness=5)

            # # Display the image with matplotlib
            # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # plt.show()

            out_imgs.append(img2)
        
        return out_imgs
    
    def __order_points_counterclockwise(self, pts, counterclockwise=True):
        """
        Orders a set of four points in a counterclockwise sequence.
        Given a set of four points, this function rearranges them so that
        they are ordered in a counterclockwise manner starting from the upper-left point.

        Args:
            pts (numpy.ndarray):
                A 4x2 array of points representing (x, y) coordinates.
            counterclockwise (boolean):
                If True, this function returns points ordered in a counterclockwise direction.
                If False, it returns points ordered in a clockwise direction.

        Returns:
            numpy.ndarray:
                A 4x2 array of points ordered in a counterclockwise manner:
                upper-left (UL) -> lower-left (LL) -> lower-right (LR)-> upper-right (UR).

        """
        rect = np.zeros((4, 2), dtype="float32")
        # find UL and LR
        d = pts.sum(axis=1) # Manhattan distance
        rect[0] = pts[np.argmin(d)]  # UL
        rect[2] = pts[np.argmax(d)]  # LR
        # find LL and UR
        diff = np.diff(pts, axis=1)
        if counterclockwise:
            rect[3] = pts[np.argmin(diff)]  # UR
            rect[1] = pts[np.argmax(diff)]  # LL
        else:
            rect[1] = pts[np.argmin(diff)]  # LL
            rect[3] = pts[np.argmax(diff)]  # UR

        return rect

if __name__ == "__main__":
    img = cv2.imread("./image.jpg")
    sc = CardScanner()
    img_detected, contours = sc.detect_card_contours(img)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    imgs = sc.transform_to_front_view(img, contours)
    for im in imgs:
        plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        plt.show()

