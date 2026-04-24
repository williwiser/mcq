# Written by William Wani (G21W7943)
import cv2
import numpy as np
from typing import Final
from pdf2image import convert_from_path
import csv
import os
import pandas as pd

MEMO: Final = np.array([
    "A","B","C","D","E","A","C","B","D","E",
    "B","D","A","E","C","C","A","E","B","D",
    "D","C","B","A","E","E","B","D","C","A",
    "A","E","C","B","D","B","A","D","E","C",
    "C","D","E","A","B","E","C","A","D","B",
    "B","A","D","C","E","D","B","E","A","C"
])

def read_pdf(path):
    """Converts a pdf to list of images"""
    images = []
    pages = convert_from_path(path, dpi=200)
  
    for page in pages:
        img = np.array(page)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        images.append(np.array(gray))
    cv2.imwrite("temp.png", images[0])
    return images

def align_to_template(template, img):
    """Warps an image according to template"""
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
    """Crops a region of an image"""
    return img[y:y+h, x:x+w]

def map_student_grid(img, tl):
    """Draws grid circles around answer bubbles"""
    f = np.zeros((30, 5), dtype=np.uint8)
    # tl = (180, 33)
    
    def group(tl, x_gap, y_gap, row_offset):
        for j in range(5):
            for i in range(5):
                filled = is_circle_filled(img, cx=tl[0]+(x_gap*i), cy=tl[1]+(y_gap*j), radius=5)
                f[row_offset + j][i] = filled
                #cv2.circle(img, (tl[0]+(x_gap*i), tl[1]+(y_gap*j)), 5, (0, 0, 255), 1)

    for g in range(6):
        group((tl[0], tl[1]+(g*82)), 12, 13, row_offset=g*5)
    
    return np.array(f)

def map_details_grid(img, tl):
    """draws grid circles around student no bubbles"""
    f = np.zeros((26, 7), dtype=np.uint8)
    # tl = (180, 33)
    
    def group(tl, x_gap, y_gap):
        for j in range(26):
            for i in range(7):
                filled = is_circle_filled(img, cx=tl[0]+(x_gap*i), cy=tl[1]+(y_gap*j), radius=5, fill_ratio=0.25)
                f[j][i] = filled
                #cv2.circle(img, (tl[0]+(x_gap*i), tl[1]+(y_gap*j)), 5, (0, 0, 255), 1)

    group((tl[0], tl[1]+(82)), 14, 14)
    
    return np.array(f)

def map_task_grid(img, tl):
    """draws grid circles around task no circles"""
    f = np.zeros((10, 2), dtype=np.uint8)
    # tl = (180, 33)
    
    def group(tl, x_gap, y_gap):
        for j in range(10):
            for i in range(2):
                filled = is_circle_filled(img, cx=tl[0]+(x_gap*i), cy=tl[1]+(y_gap*j), radius=5)
                f[j][i] = filled
                cv2.circle(img, (tl[0]+(x_gap*i), tl[1]+(y_gap*j)), 5, (0, 0, 255), 1)

    group((tl[0], tl[1]), 14, 14)
    
    return np.array(f)

def get_task_num(img):
    """gets task number from image"""
    task_num_grid = map_task_grid(img, (98, 263))
    c1 = task_num_grid[:, 0]
    c2 = task_num_grid[:, 1]

    task_num = ""
    for i in range(len(c1)):
        if c1[i] == 1:
            task_num += str(i)
            break

    for i in range(len(c2)):
        if c2[i] == 1:
            task_num += str(i)
            break

    if task_num == "":
        return "Invalid"

    return int(task_num)
                    

def get_student_ans(img):
    """gets student answers from image"""
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
        

    return answers

def student_num_valid(sn):
    """check if student no is valid"""
    if len(sn) != 8:
        return False
    if not sn[0] == "g":
        return False
    for n in sn[1:3] + sn[4:]:
        if not n.isdigit():
            return False
    if not sn[3].isalpha():
        return False
    return True

def get_student_num(img):
    """get student no from image"""
    student_num = "g"
    student_num_grid = map_details_grid(img, (30, 5))
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for j in range(2):
        for i in range(len(student_num_grid[0:10, j])):
            if (student_num_grid[0:10, j][i] == 1):
                student_num += str(i)
                break
    
    for i in range(len(student_num_grid[:, 2])):
        if (student_num_grid[:, 2][i] == 1):
            student_num += alphabet[i]
            break

    for j in range(3, 7):
        for i in range(len(student_num_grid[0:10, j])):
            if (student_num_grid[0:10, j][i] == 1):
                student_num += str(i)
                break
    return student_num

def is_circle_filled(img, cx, cy, radius, threshold=180, fill_ratio=0.5):
    """check if circle is filled"""
    
    mask = np.zeros(img.shape, dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, 255, -1)
    
    pixels = img[mask == 255]
    
    dark_pixels = np.sum(pixels < threshold)
    ratio = dark_pixels / len(pixels)
    
    return ratio > fill_ratio 

def grade(answers, memo):
    """grade student answers against memo and return mark and total"""
    mark = 0
    memo_df = pd.read_csv(memo)
    total= float(memo_df["weighting"].sum())

    for i in range(1, len(memo_df)):
        if answers[i] == memo_df["answer"][i]:
            weighting = float(memo_df.loc[memo_df["question"] == i, "weighting"].values[0])
            mark += weighting

    return mark, total

def write_file(st_num, task_num, st_ans):
    """create file of student answers"""
    data = [["question", "answer"]]

    for i, ans in enumerate(st_ans, start=1):
        data.append([i, ans])

    os.makedirs("marked", exist_ok=True)
    os.makedirs("marked_invalid_student_nums", exist_ok=True)

    if student_num_valid(st_num):
        with open(f"marked/mcq_{st_num}_task{task_num}.csv", 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)
    else:
        with open(f"marked_invalid_student_nums/mcq_{st_num}_task{task_num}.csv", 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data) 


def process_mcq(mcq_path, ans_path):
    """processes mcq and outputs csv files and grades"""
    mcq_pdf = read_pdf(path=mcq_path)
    template = cv2.imread("template.png", cv2.IMREAD_GRAYSCALE)
    template = cv2.resize(template, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    x, y, w, h = 20, 65, 410, 540
    cropped_pages = []
    grades_df = pd.DataFrame()
    grades_invalid_df = pd.DataFrame()

    for page in mcq_pdf:
        page = cv2.resize(page, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        al_page = align_to_template(template, page)
        cropped = crop_region(al_page, x, y, w, h)
        cropped_pages.append(cropped)

    for page in cropped_pages:
        student_num = get_student_num(page)
        task_num = get_task_num(page)
        student_ans = get_student_ans(page)
        student_mark = grade(student_ans, ans_path)

        percentage = (student_mark[0]/student_mark[1])*100
        if student_num_valid(student_num):
            new_grade = pd.DataFrame([{"student_no": student_num, "grade": student_mark[0], "grade(%): ": f"{percentage:.2f}"}])
            grades_df = pd.concat([grades_df, new_grade])
        else:
            new_grade = pd.DataFrame([{"student_no": student_num, "grade": student_mark[0], "grade(%): ": f"{percentage:.2f}"}])
            grades_invalid_df = pd.concat([grades_invalid_df, new_grade])

        write_file(st_num=student_num, task_num=task_num, st_ans=student_ans)

        print("---------------------------")
        print(f"Student number: {student_num}")
        print(f"Task number: {task_num}")
        print(f"Student mark: {student_mark}")
        print("---------------------------")
        print("")
    grades_df.to_csv("grades.csv", index=False)
    grades_invalid_df.to_csv("grades_invalid_student_nums.csv", index=False)

def main():
    print("-------------------------------------")
    print("             MCQ READER              ")
    print("-------------------------------------\n")

    mcq_path = input("Enter MCQ scripts path (pdf): ")
    ans_path = input("Enter answer sheet path (csv): ")
    process_mcq(mcq_path, ans_path)

if __name__ == "__main__":
    main()