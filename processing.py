import cv2
import numpy as np

# ==========================
# 1. Stylization Function
# ==========================
def stylization(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    smooth = cv2.bilateralFilter(img_rgb, 9, 75, 75)

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        9, 9
    )

    cartoon = cv2.bitwise_and(smooth, smooth, mask=edges)
    return cartoon


# ==========================
# 2. Halftoning Function
# ==========================
def halftone(img, block=8):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    new_h = h - (h % block)
    new_w = w - (w % block)
    gray = gray[:new_h, :new_w]

    halftone_img = np.zeros_like(gray)

    for i in range(0, new_h, block):
        for j in range(0, new_w, block):
            region = gray[i:i+block, j:j+block]
            intensity = np.mean(region)
            halftone_img[i:i+block, j:j+block] = 255 if intensity > 127 else 0

    return halftone_img


# ==========================
# 3. Image Blending Function
# ==========================
def image_blending(img1, img2, alpha=0.5):
    img1 = cv2.resize(img1, (500, 500))
    img2 = cv2.resize(img2, (500, 500))
    blended = cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)
    return blended
