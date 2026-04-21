import cv2


# load an image
img = cv2.imread("saber_battle.jpg", cv2.IMREAD_COLOR)
img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

height = img.shape[0]
width = img.shape[1]

img[400:800,  400:800] = cv2.rotate(img[400:800, 400:800], cv2.ROTATE_90_COUNTERCLOCKWISE)

# save image
cv2.imwrite("new_img.jpg", img)

# show the image
cv2.imshow("Image", img)

cv2.waitKey(0)
cv2.destroyAllWindows()