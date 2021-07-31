import cv2
import numpy as np
import pytesseract
import imutils

img = cv2.imread("license-plate.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
filter = cv2.bilateralFilter(gray, 5, 250, 250)
edge = cv2.Canny(filter, 50, 200)


contours = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #default contours value
cnts = imutils.grab_contours(contours)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10] #sorted area in image
screen = None

for c in cnts:
    epsilon = 0.018 * cv2.arcLength(c, True) #formul
    approx = cv2.approxPolyDP(c, epsilon, True)
    if len(approx) == 4:
        screen = approx
        #quadrilateral detection

mask = np.zeros(gray.shape, np.uint8) #black screen
maskImage = cv2.drawContours(mask, [screen], 0, (255, 255, 255), -1)
maskImage = cv2.bitwise_and(img, img, mask=mask)


#crop plate
(x, y) = np.where(mask == 255)
(topX, topY) = (np.min(x), np.min(y))
(bottomX, bottomY) = (np.max(x), np.max(y))
crop = gray[topX:bottomX+1, topY:bottomY+1]

license = pytesseract.image_to_string(crop, lang="eng", config = '--psm 6') #read text on image
print(license)

cv2.imshow("img", img)
cv2.imshow("crop", crop)

cv2.waitKey(0)
cv2.destroyAllWindows()



