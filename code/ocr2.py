import cv2
import numpy as np
from ocr import findSubChyrons

path = "C:\\users\\roy\\Downloads"

# At Prep time:
# Find bounding Boxes for Chyrons
# Bounding Box (parameter)
x, y, w, h = 64-2, 52-2, 430+4, 28+4
# Image (parameter)
image = cv2.imread(path + '\\' + 'frame11.tif')

chyron = image[y:y+h,x:x+w]
# Find sub bounding Boxes within Chyrons. Some may need to be inverted
subChyrons = findSubChyrons(chyron)

# At run time:
# Convert to B&W
chyron = cv2.cvtColor(chyron, cv2.COLOR_BGR2GRAY)
# Increase Contract (make the whites whiter)
mask = cv2.compare(chyron,200,cv2.CMP_GT)
chyron = cv2.bitwise_or(chyron, mask)

for idx, coords in enumerate(subChyrons):
    subChyron=chyron[coords["y"]:coords["y"]+coords["h"],coords["x"]:coords["x"]+coords["w"]]
    if coords["invert"]:
        subChyron = 255-subChyron # If the image is mostly dark: invert it
    cv2.imwrite(path + '\\' + "subChyronAlt"+str(idx)+".tif", subChyron)
    # perform OCR



