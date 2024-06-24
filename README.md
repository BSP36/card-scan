
### README for Card Detection and Perspective Transformation

## Overview

This Python project is designed to detect business card-like objects in an image and apply perspective transformation to correct their orientation. The program utilizes OpenCV to process the image, detect contours of cards, and transform the perspective to obtain a front-facing view of the detected cards.

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- NumPy (`numpy`)
- Matplotlib (`matplotlib`)

To install the necessary libraries, you can use:

```bash
pip install opencv-python numpy matplotlib
```

## Code Structure

- **`scanner.py`**: The main script containing the class `CardScanner` which includes methods to detect card contours and apply perspective transformation.

### Key Functions

- **`detect_card_contours(self, img, min_occupancy, max_occupancy)`**
    - **Description**: Detects edges in the input image and finds contours that are likely to represent cards.
    - **Parameters**:
        - `img`: The input image as a NumPy array.
        - `min_occupancy`: Minimum area occupancy ratio for a contour to be considered a card.
        - `max_occupancy`: Maximum area occupancy ratio for a contour to be considered a card.
    - **Returns**: The processed image with drawn contours and a list of detected contours.

- **`transform_to_front_view(self, img, contours)`**
    - **Description**: Applies perspective transformation to detected contours to get a front view.
    - **Parameters**:
        - `img`: The input image as a NumPy array.
        - `contours`: List of detected contours.
    - **Returns**: List of images with a front view of the detected cards.

### Example

Here is a basic example of how to use the `CardScanner` class in your script:

```python
from card_detector import CardScanner
import cv2

# Load the image
image = cv2.imread('path_to_your_image.jpg')

# Create a CardScanner object
scanner = CardScanner()

# Detect card contours
processed_image, contours = scanner.detect_card_contours(image)

# Transform to front view
front_views = detector.transform_to_front_view(image, contours)

# Save or display the results
for i, img in enumerate(front_views):
    cv2.imwrite(f'card_{i}.jpg', img)
    cv2.imshow(f'Card {i}', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---
