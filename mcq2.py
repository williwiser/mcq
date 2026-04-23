import cv2
import numpy as np
from pdf2image import convert_from_path

def read_pdf(path):
    images = []
    pages = convert_from_path(path, dpi=200)
  
    for page in pages:
        img = np.array(page)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        images.append(np.array(gray))
    cv2.imwrite("temp.png", images[0])
    return images

def align_to_template(template, img):
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(template, None)
    kp2, des2 = orb.detectAndCompute(img, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)[:200]

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    h, w = template.shape
    aligned = cv2.warpPerspective(img, H, (w, h))
    return aligned

def crop_region(img, x, y, w, h):
    return img[y:y+h, x:x+w]

#738 144
#184.5 36
#179, 33
def map_grid(img):
    f = np.zeros((30, 5), dtype=np.uint8)
    tl = (180, 33)
    
    def group(tl, x_gap, y_gap, row_offset):
        for j in range(5):
            for i in range(5):
                filled = is_circle_filled(img, cx=tl[0]+(x_gap*i), cy=tl[1]+(y_gap*j), radius=5)
                f[row_offset + j][i] = filled
                cv2.circle(img, (tl[0]+(x_gap*i), tl[1]+(y_gap*j)), 5, (0, 0, 255), 1)

    for g in range(6):
        group((tl[0], tl[1]+(g*82)), 12, 13, row_offset=g*5)
    
    print(f)


def is_circle_filled(img, cx, cy, radius, threshold=127, fill_ratio=0.5):
    # create a mask for the circle
    mask = np.zeros(img.shape, dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, 255, -1)  # -1 fills the circle
    
    # get pixels within the circle
    pixels = img[mask == 255]
    
    # ratio of dark pixels
    dark_pixels = np.sum(pixels < threshold)
    ratio = dark_pixels / len(pixels)
    
    return ratio > fill_ratio  # True if mostly filled/dark

    

def main():
    example = read_pdf("MCQ_600dpi_2016.pdf")[2]
    example = cv2.resize(example, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)

    template_img = cv2.imread("template.png", cv2.IMREAD_GRAYSCALE)
    template_img = cv2.resize(template_img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)

    x, y, w, h = 20, 65, 410, 540   # adjust to your region of interest
    template_crop = crop_region(template_img, x, y, w, h)
   

    al = align_to_template(template_img, example)
    region = crop_region(al, x, y, w, h)

    map_grid(region)
    cv2.imwrite("f_out.png", region)

if __name__ == "__main__":
    main()