import cv2
import numpy as np
from typing import Final
from pdf2image import convert_from_path
import csv

MEMO: Final = np.array([
    [1,0,0,0,0],
    [0,0,1,0,0],
    [0,1,0,0,0],
    [0,0,0,1,0],
    [0,0,0,0,1],

    [0,1,0,0,0],
    [1,0,0,0,0],
    [0,0,0,0,1],
    [0,0,1,0,0],
    [0,0,0,1,0],

    [0,0,0,1,0],
    [0,1,0,0,0],
    [0,0,0,0,1],
    [1,0,0,0,0],
    [0,0,1,0,0],

    [0,0,1,0,0],
    [0,0,0,1,0],
    [1,0,0,0,0],
    [0,1,0,0,0],
    [0,0,0,0,1],

    [0,0,0,0,1],
    [0,0,1,0,0],
    [0,1,0,0,0],
    [0,0,0,1,0],
    [1,0,0,0,0],

    [1,0,0,0,0],
    [0,0,0,1,0],
    [0,0,1,0,0],
    [0,1,0,0,0],
    [0,0,0,0,1],

    [0,1,0,0,0],
    [0,0,0,0,1],
    [1,0,0,0,0],
    [0,0,1,0,0],
    [0,0,0,1,0],

    [0,0,0,1,0],
    [1,0,0,0,0],
    [0,1,0,0,0],
    [0,0,0,0,1],
    [0,0,1,0,0],

    [0,0,1,0,0],
    [0,0,0,0,1],
    [0,0,0,1,0],
    [1,0,0,0,0],
    [0,1,0,0,0],

    [0,1,0,0,0],
    [0,0,1,0,0],
    [0,0,0,1,0],
    [0,0,0,0,1],
    [1,0,0,0,0],

    [0,0,0,0,1],
    [0,1,0,0,0],
    [1,0,0,0,0],
    [0,0,1,0,0],
    [0,0,0,1,0],

    [1,0,0,0,0],
    [0,0,1,0,0],
    [0,1,0,0,0],
    [0,0,0,0,1],
    [0,0,0,0,1],
 ], dtype=int)

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
def map_student_grid(img, tl):
    f = np.zeros((30, 5), dtype=np.uint8)
    # tl = (180, 33)
    
    def group(tl, x_gap, y_gap, row_offset):
        for j in range(5):
            for i in range(5):
                filled = is_circle_filled(img, cx=tl[0]+(x_gap*i), cy=tl[1]+(y_gap*j), radius=5)
                f[row_offset + j][i] = filled
                cv2.circle(img, (tl[0]+(x_gap*i), tl[1]+(y_gap*j)), 5, (0, 0, 255), 1)

    for g in range(6):
        group((tl[0], tl[1]+(g*82)), 12, 13, row_offset=g*5)
    
    return np.array(f)

def map_details_grid(img, tl):
    f = np.zeros((26, 7), dtype=np.uint8)
    # tl = (180, 33)
    
    def group(tl, x_gap, y_gap):
        for j in range(26):
            for i in range(7):
                filled = is_circle_filled(img, cx=tl[0]+(x_gap*i), cy=tl[1]+(y_gap*j), radius=5)
                f[j][i] = filled
                cv2.circle(img, (tl[0]+(x_gap*i), tl[1]+(y_gap*j)), 5, (0, 255, 0), 1)

    group((tl[0], tl[1]+(82)), 14, 14)
    
    return np.array(f)
                    

def get_student_ans(img):
    c1 = map_student_grid(img, (180, 33))
    c2 = map_student_grid(img, (298, 33))
    grid= np.concatenate((c1, c2), axis=0)

    letters = "ABCDE"
    answers = []
    for q in grid:
        blank = True
        for i in range(5):
            if q[i] == 1:
                answers.append(letters[i])
                blank = False
                break
        if blank:
            answers.append("_")
        

    return grid, answers

def get_student_num(img):
    student_num = "g"
    student_num_grid = map_details_grid(img, (30, 5))
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for j in range(2):
        for i in range(len(student_num_grid[0:10, j])):
            if (student_num_grid[0:10, j][i] == 1):
                student_num += str(i)
    
    for i in range(len(student_num_grid[:, 2])):
        if (student_num_grid[:, 2][i] == 1):
            student_num += alphabet[i]

    for j in range(3, 7):
        for i in range(len(student_num_grid[0:10, j])):
            if (student_num_grid[0:10, j][i] == 1):
                student_num += str(i)
    return student_num

def is_circle_filled(img, cx, cy, radius, threshold=180, fill_ratio=0.5):
    # create a mask for the circle
    mask = np.zeros(img.shape, dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, 255, -1)  # -1 fills the circle
    
    # get pixels within the circle
    pixels = img[mask == 255]
    
    # ratio of dark pixels
    dark_pixels = np.sum(pixels < threshold)
    ratio = dark_pixels / len(pixels)
    
    return ratio > fill_ratio  # True if mostly filled/dark

def grade(answers, memo):
    corr = 0

    for i in range(len(memo)):
        if (answers[i] == memo[i]).all():
            corr += 1

    return corr, len(memo)

def main():
    example = read_pdf("MCQ_600dpi_2016.pdf")[3]
    example = cv2.resize(example, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)

    template_img = cv2.imread("template.png", cv2.IMREAD_GRAYSCALE)
    template_img = cv2.resize(template_img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)

    x, y, w, h = 20, 65, 410, 540   # adjust to your region of interest
    template_crop = crop_region(template_img, x, y, w, h)
   

    al = align_to_template(template_img, example)
    region = crop_region(al, x, y, w, h)

    # map_grid(region, (180, 33))
    # map_grid(region, (298, 33))
    student_ans = get_student_ans(region)
    # student_num = map_details_grid(region, (30, 5))
    student_num = get_student_num(region)
    print(get_student_num(region))
    print(student_ans[1])
    print(grade(student_ans[0], MEMO))
    cv2.imwrite("f_out.png", region)

    data = [["question", "answer"]]

    for i, ans in enumerate(student_ans[1], start=1):
        data.append([i, ans])

    with open(f"mcq_{student_num}.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data) # Writes all rows at once

if __name__ == "__main__":
    main()