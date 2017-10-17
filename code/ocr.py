# from PIL import Image
import cv2
import os
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
    #path = "C:/users/roy/Downloads"
    home = os.path.expanduser("~")
    path = os.path.join(home,"Downloads")
    path = os.path.join(path,"videos")
    file = os.path.join(path, inputFile)
    vidcap = cv2.VideoCapture(file)
    count = 0
    samples = 500
    sample_window = 10
    success = True
    chyronCoord = []

    fps = vidcap.get(cv2.CAP_PROP_FPS)
    codec = vidcap.get(cv2.CAP_PROP_FOURCC)
    width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    minChyronHeight = height * 0.025
    minChyronWidth = width * 0.025
    print('FPS: ', fps, ' CODEC: ', decode_fourcc(codec), 'DIMENSIONS: (', width, 'x', height, ')')

    imageBW = None
    while count <= samples:
        if imageBW is not None:
            prev_imageBW = imageBW
            prev_image = image
        success, image = vidcap.read()
        pos = vidcap.get(cv2.CAP_PROP_POS_MSEC)  # position in msec
        vidcap.set(cv2.CAP_PROP_POS_MSEC, pos + 250)
        # if pos % 250 != 0:
        #     continue  # Skip to the next 1/4 second
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
            mse = compare_mse(newClip, clip)
            if score > 0.75 and mse < 1000:
                print("Chyron #", idx, ", sec. ", int(pos / 1000), ", SSIM: {}".format(score), ', MSE=',
                      compare_mse(newClip, clip))

        imageBW = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to B&W
        if count == 1:
            diff = np.zeros(imageBW.shape)
            chyronMask = np.zeros(imageBW.shape, dtype=np.uint8)
            diffs = []
            continue

        # Compare the two images and add 1 (True is white which is 255 so need to divide by 255)
        # in place of each pixel that hadn't changed
        # diff = cv2.inRange(imageBW, (prev_imageBW - 1), (prev_imageBW + 1))/255
        # diff = cv2.inRange(image, (prev_image - 1), (prev_image + 1)) / 255
        diff = np.sum(cv2.compare(image, prev_image, cv2.CMP_EQ) / 255, 2)
        # diff = image and image[1:height-1] and image[1:height-1]
        diffs.append(diff)
        if len(diffs) < sample_window:  # we have enough samples
            continue
        elif len(diffs) > sample_window:
            del diffs[0]
        # Sum the number of times each pixel stayed the same in the each location

        accumulator = np.sum(diffs, 0, keepdims=True)  # + np.sum(diffs[-3], 0, keepdims=True)
        accumulator = diffs[-3] + 2 * diffs[-2] + 3 * diffs[-1] + np.squeeze(accumulator)  # Recency bias
        # Threshold of a point in the mask is 1/2 the sample window (# of frames compared)
        imageOut = cv2.compare(accumulator, sample_window + 3, cv2.CMP_GT)
        imageOut = cv2.bitwise_and(imageOut, (255 * (1 - chyronMask)))  # ignore previously found areas

        cv2.imwrite(path + '/' + "diff.tif", imageOut)

        # Threahold of a minimum number of fixed pixels
        if (np.sum(imageOut / 255) / (width * height)) <= 0.001:
            if debug:
                cv2.imwrite(path + '/' + "image.tif", image)
            continue

        # cv2.imwrite(path + '/' + "accumulator.tif", accumulator)
        # cv2.imwrite(path + '/' + "imageBW.tif", imageBW)
        # cv2.imwrite(path + '/' + "prev_imageBW.tif", prev_imageBW)
        # Find the largest contour around the discovered points
        img, contours, hierarchy = cv2.findContours(imageOut, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        largest_area = sorted(contours, key=cv2.contourArea)[-1]

        # Create a rectangular bounding box and draw it on the image
        x, y, w, h = cv2.boundingRect(largest_area)
        if debug:
            cv2.rectangle(image, (x - 1, y - 1), (x + w + 1, y + h + 1), (255, 255, 0), 1)
            cv2.imwrite(path + '/' + "image.tif", image)


        # Tighten it around the text by maximizing the percent of white pixels
        percentWhite = np.sum(imageOut[y:y + h - 1, x:x + w - 1] == 255) / (w * h)
        while percentWhite > 0.7 and percentWhite <= 0.92 and h > minChyronHeight and w > minChyronWidth:
            if np.sum(imageOut[y:y + h, x] == 255) / h < 0.8 and w > minChyronWidth:
                x = x + 1
                w = w - 1
            if np.sum(imageOut[y:y + h, x + w - 1] == 255) / h < 0.8 and w > minChyronWidth:
                w = w - 1
            if np.sum(imageOut[y, x:x + w] == 255) / w < 0.8 and h > minChyronHeight:
                y = y + 1
                h = h - 1
            if np.sum(imageOut[y + h - 1, x:x + w] == 255) / w < 0.8 and h > minChyronHeight:
                h = h - 1
            cv2.rectangle(image, (x - 1, y - 1), (x + w + 1, y + h + 1), (255, 255, 0), 1)
            if debug:
                cv2.imwrite(path + '/' + "image.tif", image)
            # if percentWhite < np.sum(imageOut[y + 1:y + h, x:x + w] == 255) / (w * (h - 1)):
            #     y, h = y + 1, h - 1
            # if percentWhite < np.sum(imageOut[y:y + h - 1, x:x + w] == 255) / (w * (h - 1)):
            #     h = h - 1
            # if percentWhite < np.sum(imageOut[y:y + h, x + 1:x + w] == 255) / ((w - 1) * h):
            #     x, w = x + 1, w - 1
            # if percentWhite < np.sum(imageOut[y:y + h, x:x + w - 1] == 255) / ((w - 1) * h):
            #     w = w - 1
            if abs(percentWhite - np.sum(imageOut[y:y + h, x:x + w] == 255) / (w * h)) <= 0.001:
                break
            percentWhite = np.sum(imageOut[y:y + h, x:x + w] == 255) / (w * h)

        if percentWhite > 0.92:
            crop = image[y:y + h, x:x + w]
            cv2.imwrite(path + '/' + "crop.tif", crop)
            coords = {'x': x, 'y': y, 'w': w, 'h': h, 'sec': pos / 1000, 'crop': crop}
            chyronCoord.append(coords)
            chyronMask[coords["y"] - 2:coords["y"] + coords["h"] + 2, coords["x"] - 2:coords["x"] + coords["w"] + 2] = 1
            if debug:
                print('Found Chyron: ', 'x=', x, ' y=', y, ' w=', w, ' h=', h, 'len=', len(diffs), ' percentWhite=',
                      percentWhite)

    return chyronCoord


inputFile = "West Ham United vs Tottenham 2017-09-23 2nd half ENG.mp4"
chyronCoord = findChyrons(inputFile, False)
for idx, coords in enumerate(chyronCoord):
    print('Adjusted box: ', 'x=', coords["x"], ' y=', coords["y"], ' w=', coords["w"], ' h=', coords["h"], 'sec=',
          coords["sec"])
    home = os.path.expanduser("~")
    path = os.path.join(home, "Downloads", "videos")
    cv2.imwrite(path + '/' + "Chyron" + str(idx) + ".tif", coords["crop"])


#
# # Create a mask and apply to an image we want to extract text from
# img1 = cv2.imread(path + '/' + 'frame11.tif')
# mask = np.zeros(img1.shape, np.uint8)
# mask[y:y+h, x:x+w] = 255
# dst = cv2.bitwise_and(img1, mask)
# mask = 255 - mask
# roi = cv2.add(dst, mask)
# cv2.imwrite(path + '/' + "roi.tif", roi)
# cv2.imwrite(path + '/' + "dst.tif", dst)
# cv2.imwrite(path + '/' + "mask.tif", mask)
#
# # Prepare image for OCR
# img = cv2.imread(path + '/' + 'roi.tif')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# kernel = np.ones((1, 1), np.uint8)
# img = cv2.dilate(img, kernel, iterations = 1)
# cv2.imwrite(path + '/' + "x-" + 'roi.tif', img)
# img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
# cv2.imwrite(path + '/' + "y-" + 'roi.tif', img)
# # result = pytesseract.inage_to_string(Image.open(path + '/' + "y-" + image))
