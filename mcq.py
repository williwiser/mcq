import numpy as np
import cv2
from pdf2image import convert_from_path

def preprocess(img):
    # 1. Denoise (light blur)
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # 2. Improve contrast (VERY important for PDFs)
    # img = cv2.equalizeHist(img)

    # # 3. Adaptive threshold (handles uneven lighting)
    # thresh = cv2.adaptiveThreshold(
    #     img,
    #     255,
    #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #     cv2.THRESH_BINARY_INV,
    #     11,
    #     2
    # )

    #  4. Morphological cleanup
    # kernel = np.ones((3, 3), np.uint8)
    # cleaned = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    return img

def read_pdf(path):
    images = []
    pages = convert_from_path(path, dpi=200)
  
    for page in pages:
        img = np.array(page)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        images.append(np.array(gray))
    cv2.imwrite("temp.png", images[0])
    return images

def order_points(pts):
    pts = np.array(pts, dtype="float32")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")

def multi_scale_match(image, template, scales, threshold=0.7):
    h0, w0 = template.shape
    boxes = []
    scores = []

    for scale in scales:
        # resize template
        resized = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        h, w = resized.shape

        # skip if template becomes bigger than image
        if h > image.shape[0] or w > image.shape[1]:
            continue

        res = cv2.matchTemplate(image, resized, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)

        for pt in zip(*loc[::-1]):
            x, y = pt
            boxes.append([x, y, w, h])
            scores.append(res[y, x])

    return boxes, scores

def warp_page(img, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    return warped

# def map_grid(img):
#     cv2.

def get_anchor_pnts(img):
    img2 = img.copy()
    anchor = cv2.imread("anchor.png", cv2.IMREAD_GRAYSCALE)
    anchor = cv2.resize(anchor, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    w, h = anchor.shape[::-1]

    scales = np.linspace(0.6, 1.5, 20)
    

    res = cv2.matchTemplate(img, anchor, cv2.TM_CCOEFF_NORMED)
    threshold = 0.7
    loc = np.where(res >= threshold)

    boxes, scores = multi_scale_match(img, anchor, scales, threshold=0.8)

    indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        score_threshold=threshold,
        nms_threshold=0.3
    )

  
    final_points = []

    if len(indices) > 0:
        for i in indices:
            i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
            x, y, w, h = boxes[i]

            cx = x + w // 2
            cy = y + h // 2

            final_points.append((cx, cy))
            cv2.rectangle(img2, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # if len(final_points) > 4:
    #     # keep top 4 by score
    #     paired = list(zip(final_points, scores))
    #     paired = sorted(paired, key=lambda x: x[1], reverse=True)
    #     final_points = [p for p, _ in paired[:4]]

    
    return np.array(final_points, dtype="float32")


def main():
    example = read_pdf("MCQ_600dpi_2016.pdf")[0]
    example = cv2.resize(example, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    
    # template_img = cv2.imread("template.png", cv2.IMREAD_GRAYSCALE)
    # template_img = cv2.resize(template_img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
   
    anchor_pnts = get_anchor_pnts(example)
    warped = warp_page(example, anchor_pnts)

    cv2.imwrite("res.png", warped)
    cv2.imshow("MCQ", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()