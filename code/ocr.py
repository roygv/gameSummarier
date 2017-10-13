from PIL import Image
import cv2
import numpy as np
#import pytesseract
print(cv2.__version__)


def decode_fourcc(v):
    v = int(v)
    return "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])

def findSubChyrons(chyron):
    x, y, w, h = 0, 0, chyron.shape[1], chyron.shape[0]
    sequenceOf = 0
    subChyronCoord = []

    # Convert to B&W
    chyron = cv2.cvtColor(chyron, cv2.COLOR_BGR2GRAY)

    # Increase Contract (make the whites whiter)
    mask = cv2.compare(chyron,200,cv2.CMP_GT)
    chyron = cv2.bitwise_or(chyron, mask)

    for i in range(w-1):
        if np.sum(chyron[:,i] == chyron[:,i+1]) <= 1:
            # If adjacent vertical columns are totally different
            # this is a border between two subChyrons
            sequenceOf += 1
        else:
            if sequenceOf >= 3:
                subChyron = chyron[y:y+h,x:i-sequenceOf+1]
                invert = (np.mean(subChyron) <= 200)
                subChyronCoord.append({'x': x, 'y': y, 'w': i - sequenceOf + 1 - x, 'h': h, 'invert': invert})
                x = i
            sequenceOf = 0
    subChyron = chyron[y:y + h, x:i - sequenceOf + 1]
    invert = (np.mean(subChyron) <= 200)
    subChyronCoord.append({'x': x, 'y': y, 'w': i - sequenceOf + 1 - x, 'h': h, 'invert': invert})
    print(subChyronCoord)
    return  subChyronCoord



path = "C:\\users\\roy\\Downloads"
inputFile = "West Ham United vs Tottenham 2017-09-23 2nd half ENG.mp4"
fps=30

vidcap = cv2.VideoCapture(path + '\\' + inputFile)
count = 0
samples = 300
success = True
fps=vidcap.get(cv2.CAP_PROP_FPS)
codec=vidcap.get(cv2.CAP_PROP_FOURCC)
print('FPS: ',fps,' CODEC: ',decode_fourcc(codec))
# Count the number of times each pixel stayed the same between two frames 1 sec apart
while success:
    success, image = vidcap.read()
    pos=vidcap.get(cv2.CAP_PROP_POS_MSEC)  # position in msec
    if pos % 1000 != 0:
        continue # Skip to the next second
    if success != True:
        print('Failed to read frame: ', success)
    count += 1
    imageBW = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert to B&W
    if count == 1:
        accumulator = np.zeros(imageBW.shape)
        image1 = imageBW
        continue
    else:
        image2 = imageBW
    # Compare the two images and add 1 (True is white which is 255 so need to divide by 255)
    # in place of each pixel that hadn't changed
    accumulator += cv2.inRange(image1, (image2 - 2), (image2 + 2))/255
    image1 = image2
    if count > samples:
        break

# Threshold of a point in the mask is 1/2 the sampled images
imageOut = cv2.compare(accumulator,samples/2,cv2.CMP_GT)
cv2.imwrite(path + '\\' + "diff.tif", imageOut)
cv2.imwrite(path + '\\' + "accumulator.tif", accumulator)


# Find the largest contour around the discovered points
img, contours, hierarchy = cv2.findContours(imageOut, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
largest_area = sorted(contours, key=cv2.contourArea)[-1]

# Create a rectangular bounding box
x,y,w,h = cv2.boundingRect(largest_area)
print('Found box: ','x=',x,' y=',y,' w=',w,' h=',h)

# Tighten it around the text by maximizing the percent of white pixels
percentWhite=np.sum(imageOut[y:y+h,x:x+w]==255)/(w*h)
while True:
    if percentWhite < np.sum(imageOut[y+1:y+h,x:x+w]==255)/(w*(h-1)):
        y,h = y+1, h-1
    elif percentWhite < np.sum(imageOut[y:y+h-1,x:x+w]==255)/(w*(h-1)):
        h = h-1
    elif percentWhite < np.sum(imageOut[y:y+h, x+1:x+w] == 255) / ((w-1) * h):
        x, w = x+1, w-1
    elif percentWhite < np.sum(imageOut[y:y+h, x:x+w-1] == 255) / ((w-1) * h):
        w = w - 1
    else:
        break
    percentWhite = np.sum(imageOut[y:y + h, x:x + w] == 255) / (w * h)
print('Adjusted box: ','x=',x,' y=',y,' w=',w,' h=',h)


# Create a mask and apply to an image we want to extract text from
img1 = cv2.imread(path +  '\\' + 'frame11.tif')
mask = np.zeros(img1.shape, np.uint8)
mask[y:y+h,x:x+w] = 255
dst = cv2.bitwise_and(img1, mask)
mask = 255 - mask
roi = cv2.add(dst, mask)
cv2.imwrite(path + '\\' + "roi.tif", roi)
cv2.imwrite(path + '\\' + "dst.tif", dst)
cv2.imwrite(path + '\\' + "mask.tif", mask)

# Prepare image for OCR
img = cv2.imread(path + '\\' + 'roi.tif')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
kernel = np.ones((1,1),np.uint8)
img = cv2.dilate(img, kernel, iterations = 1)
cv2.imwrite(path + '\\' + "x-" + 'roi.tif', img)
img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11,2)
cv2.imwrite(path + '\\' + "y-" + 'roi.tif', img)
    # result = pytesseract.inage_to_string(Image.open(path + '\\' + "y-" + image))