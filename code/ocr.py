# from PIL import Image
import cv2
import numpy as np
from skimage.measure import compare_ssim, compare_mse

# import pytesseract
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
        if np.sum(chyron[:, i] == chyron[:, i + 1]) <= 1:
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
    return subChyronCoord


def findChyrons(inputFile, debug=False):
    path = "C:\\users\\roy\\Downloads"
    vidcap = cv2.VideoCapture(path + '\\' + inputFile)
    count = 0
    samples = 300
    sample_window = 10
    success = True
    chyronCoord = []

    fps = vidcap.get(cv2.CAP_PROP_FPS)
    codec = vidcap.get(cv2.CAP_PROP_FOURCC)
    width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print('FPS: ', fps, ' CODEC: ', decode_fourcc(codec), 'DIMENSIONS: (', width, 'x', height, ')')

    imageBW = None
    while count <= samples:
        if imageBW is not None:
            prev_imageBW = imageBW
            prev_image = image
        success, image = vidcap.read()
        pos = vidcap.get(cv2.CAP_PROP_POS_MSEC)  # position in msec
        if pos % 250 != 0:
            continue  # Skip to the next 1/4 second
        if not success:
            print('Failed to read frame: ', success)
            break
        count += 1

        # Go over previously discovered chyrons, check if they appear in this image (SSIM > 99%)
        # and the content had changed compared to the last saved cropped image (MSE > .
        for idx, coords in enumerate(chyronCoord):
            newClip = image[coords["y"]:coords["y"] + coords["h"], coords["x"]:coords["x"] + coords["w"]]
            clip = coords["crop"]
            score = compare_ssim(clip, newClip, multichannel=True)
            chyronMask[coords["y"] - 2:coords["y"] + coords["h"] + 2, coords["x"] - 2:coords["x"] + coords["w"] + 2] = 1
            print("SSIM: {}".format(score), ', MSE=', compare_mse(newClip, clip))

        imageBW = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to B&W
        if count == 1:
            diff = np.zeros(imageBW.shape)
            chyronMask = np.zeros(imageBW.shape, dtype=np.uint8)
            diffs = []
            continue

        # Compare the two images and add 1 (True is white which is 255 so need to divide by 255)
        # in place of each pixel that hadn't changed
        # diff = cv2.inRange(imageBW, (prev_imageBW - 1), (prev_imageBW + 1))/255
        diff = cv2.inRange(image, (prev_image - 1), (prev_image + 1)) / 255
        diffs.append(diff)
        if len(diffs) < sample_window:  # we have enough samples
            continue
        elif len(diffs) > sample_window:
            del diffs[0]
        # Sum the number of times each pixel stayed the same in the each location
        accumulator = np.sum(diffs, 0, keepdims=True)
        accumulator = np.squeeze(accumulator)
        # Threshold of a point in the mask is 1/2 the sample window (# of frames compared)
        imageOut = cv2.compare(accumulator, sample_window / 2, cv2.CMP_GT)
        imageOut = cv2.bitwise_and(imageOut, (255 * (1 - chyronMask)))  # ignore previously found areas

        # Threahold of a minimum number of fixed pixels
        if (np.sum(imageOut / 255) / (width * height)) <= 0.001:
            continue

        cv2.imwrite(path + '\\' + "diff.tif", imageOut)
        # cv2.imwrite(path + '\\' + "accumulator.tif", accumulator)
        cv2.imwrite(path + '\\' + "imageBW.tif", imageBW)
        cv2.imwrite(path + '\\' + "prev_imageBW.tif", prev_imageBW)
        # Find the largest contour around the discovered points
        img, contours, hierarchy = cv2.findContours(imageOut, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        largest_area = sorted(contours, key=cv2.contourArea)[-1]

        # Create a rectangular bounding box and draw it on the image
        x, y, w, h = cv2.boundingRect(largest_area)
        cv2.rectangle(image, (x - 1, y - 1), (x + w + 1, y + h + 1), (255, 255, 0), 1)
        cv2.imwrite(path + '\\' + "image.tif", image)
        if debug == True:
            print('Found box: ', 'x=', x, ' y=', y, ' w=', w, ' h=', h, 'count=', np.sum(diff))

        # Tighten it around the text by maximizing the percent of white pixels
        percentWhite = np.sum(imageOut[y:y + h, x:x + w] == 255) / (w * h)
        while percentWhite <= 0.98:
            if h == 1 or w == 1:  # Avoid dividing by 0
                break
            if percentWhite < np.sum(imageOut[y + 1:y + h, x:x + w] == 255) / (w * (h - 1)):
                y, h = y + 1, h - 1
            if percentWhite < np.sum(imageOut[y:y + h - 1, x:x + w] == 255) / (w * (h - 1)):
                h = h - 1
            if percentWhite < np.sum(imageOut[y:y + h, x + 1:x + w] == 255) / ((w - 1) * h):
                x, w = x + 1, w - 1
            if percentWhite < np.sum(imageOut[y:y + h, x:x + w - 1] == 255) / ((w - 1) * h):
                w = w - 1
            if abs(percentWhite - np.sum(imageOut[y:y + h, x:x + w] == 255) / (w * h)) <= 0.001:
                break
            percentWhite = np.sum(imageOut[y:y + h, x:x + w] == 255) / (w * h)
        if debug:
            print('Adjusted box: ', 'x=', x, ' y=', y, ' w=', w, ' h=', h, 'len=', len(diffs), ' percentWhite=',
                  percentWhite)

        if h <= 20 or w <= 20:  # Reasonability test
            continue

        if percentWhite > 0.98:
            crop = image[y:y + h, x:x + w]
            cv2.imwrite(path + '\\' + "crop.tif", crop)
            chyronCoord.append({'x': x, 'y': y, 'w': w, 'h': h, 'sec': pos / 1000, 'crop': crop})
    return chyronCoord


inputFile = "West Ham United vs Tottenham 2017-09-23 2nd half ENG.mp4"
chyronCoord = findChyrons(inputFile)
for idx, coords in enumerate(chyronCoord):
    print('Adjusted box: ', 'x=', coords["x"], ' y=', coords["y"], ' w=', coords["w"], ' h=', coords["h"], 'sec=',
          coords["sec"])


#
# # Create a mask and apply to an image we want to extract text from
# img1 = cv2.imread(path + '\\' + 'frame11.tif')
# mask = np.zeros(img1.shape, np.uint8)
# mask[y:y+h, x:x+w] = 255
# dst = cv2.bitwise_and(img1, mask)
# mask = 255 - mask
# roi = cv2.add(dst, mask)
# cv2.imwrite(path + '\\' + "roi.tif", roi)
# cv2.imwrite(path + '\\' + "dst.tif", dst)
# cv2.imwrite(path + '\\' + "mask.tif", mask)
#
# # Prepare image for OCR
# img = cv2.imread(path + '\\' + 'roi.tif')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# kernel = np.ones((1, 1), np.uint8)
# img = cv2.dilate(img, kernel, iterations = 1)
# cv2.imwrite(path + '\\' + "x-" + 'roi.tif', img)
# img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
# cv2.imwrite(path + '\\' + "y-" + 'roi.tif', img)
# # result = pytesseract.inage_to_string(Image.open(path + '\\' + "y-" + image))
