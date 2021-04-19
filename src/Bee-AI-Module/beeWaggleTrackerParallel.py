###################################################################
# This file determines which bee movements are actually waggles
# This file will take the longest to run out of the processing ones

import pandas as pd
import numpy as np
import cv2
from scipy import interpolate
import os
from multiprocessing import Pool, freeze_support

mask = None

def zeroMask(filePath):
    cap = cv2.VideoCapture('{}'.format(filePath))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    return np.zeros((height, width), dtype=np.uint8)

# find the largest contour within the given bounding box
def findROIContour(thresh, bbox, mask):
    # map works the same as in Java, put listlike data into a function
    # int as a function in this case converts all elements of bbox to ints
    bbox = map(int, bbox)
    # get the coords of the bounding box from the tuple returned by map
    x, y, w, h = bbox
    # ROI based on bounding box coordinates
    threshROI = thresh[y:y+h, x:x+w]
    # return an array of the dimensions given by thresh filled with int 0s

    mask[y:y+h, x:x+w] = threshROI

    # print(mask)

    # longer comment about findContours on the beeDetector.py file
    # cv2.RETR_LIST
    #   retrieves all of the contours without establishing any hierarchical relationships
    # cv2.CHAIN_APPROX_NONE
    #   stores absolutely all the contour points. That is, any 2 subsequent points
    #   (x1,y1) and (x2,y2) of the contour will be either horizontal, vertical or
    #   diagonal neighbors, that is, max(abs(x1-x2),abs(y2-y1))==1.
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    mask[y:y+h, x:x+w] = 0

    # next part of the code references:
    # https://stackoverflow.com/questions/54615166/find-the-biggest-contour-opencv-python-getting-errors
    contourAreas = [(cv2.contourArea(c), c) for c in contours[0]]
    # if no contours found, return None
    if contourAreas is None or len(contourAreas) == 0:
        finalContours = [None, None]
    else:
        # check what this lambda does
        finalContours = max(contourAreas, key=lambda x: x[0])
    return finalContours[1]


# find centroid of the contour - more notes on beeDetector.py
def getContourMomentInt(contour):
    m = cv2.moments(contour)
    x = m['m10'] / m['m00']
    y = m['m01'] / m['m00']
    return int(x), int(y)

def findFullContour(thresh, center, mask):
    x, y = center
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for c in contours[0]:
        # pointPolygonTest determines if a point is inside, outside, or on the edge (vertex) of a contour
        # positive for inside, negative for outside, and zero for on edge
        # pointPolygonTest(contour, point, measureDist)
        # if measureDist, then the return value is a signed distance of the point from its nearest contour edge
        # if measureDist is false, the return value is +1, -1, 0
        # https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html#gadf1ad6a0b82947fa1fe3c3d497f260e0
        dist = cv2.pointPolygonTest(c, (x, y), False)
        if dist > 0:
            finalContour = c
            break
        else:
            # if it cannot find a contour, look around for one
            finalContour = findROIContour(thresh, (x-10, y-10, 20, 20), mask)
    return finalContour

# fits a bounding box tightly around the contour
def getFittedBox(contour):
    # minAreaRect finds a rotated rectangle of the minimum area enclosing the input 2D point set
    # https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga3d476a3417130ae5154aea421ca7ead9
    rect = cv2.minAreaRect(contour)
    # boxPoints returns the four vertices of the rotated rectangle found by minAreaRect
    # int0 converts bounding box floating point values to ints
    # https://stackoverflow.com/questions/48350693/what-is-numpy-method-int0
    box = np.int0(cv2.boxPoints(rect))
    return rect, box

# reset the orientation of a bounding box to 0degrees
def rotatedBoxConverter(box):
    straightBox = np.array(box).T
    x, y = min(straightBox[0]), min(straightBox[1])
    w, h = max(straightBox[0]) - x, max(straightBox[1]) - y
    return x, y, w, h

# calculate avg area of boxes
def avgArea(box, total, count):
    x, y, w, h = box
    total += (w*h)
    avg = total/count
    return total, avg

# resize a given box
def expandBox(img, bbox, mask):
    contour = None
    x, y, w, h = bbox
    x -= 10
    w += 20
    y -= 10
    h += 20
    bbox = (x, y, w, h)
    contour = findROIContour(img, bbox, mask)
    return bbox, contour

# Find which way object is facing by the direction in which bounding box moves, 
# to be coupled with angle of bounding rect
def moveDirection(prevBBox, bbox):
    x, y, w, h = bbox
    x0, y0, w0, h0 = prevBBox
    delX = x - x0
    delY = y - y0
    movement = (delX, delY)
    return movement

# np.zeros returns an array of given shape filled with 0s, used to overlay data on the frame
def createMask(img):
    # create an  array with dimensions i[0] x i[1] x 3 and fill it with integer 255 (black)
    mask = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    mask.fill(255)
    return mask

def anchorInterpolation(bbox, fx, fy, counter):
    fx0, fy0 = int(fx(counter)), int(fy(counter))
    x0, y0, x1, y1 = fx0-30, fy0-30, fx0+30, fy0+30
    x, y, w, h = bbox

    if x in range(x0, x1) and y in range(y0, y1):
        success = True
    else:
        success = False
        bbox = resetBox(fx0, fy0)
    return success, bbox

###########################################################################
# These 1-line functions make several if statements and loops more readable

def roundBox(bbox):
    return int(round(bbox[0], 0)), int(round(bbox[1], 0)), bbox[2], bbox[3]


def checkLow(low):
    return low < 60


def resetBox(x, y):
    return x-15, y-15, 30, 30


def interp(frame, x, y, interpKind):
    return interpolate.interp1d(frame, x, kind=interpKind), interpolate.interp1d(frame, y, kind=interpKind)

# check if the current counding box is out of bounds
def checkOOB(bbox, height, width):
    return (bbox[0] < 0 or bbox[0]+bbox[2] > width or bbox[1] < 0 or bbox[1]+bbox[3] > height)


def makeKernel():
    return np.ones((2, 2), np.uint8)


def checkContour(contour):
    return (contour is None or cv2.contourArea(contour) <= 80)

# adjust the size of the bounding box 
def adjustBox(box, x, y, w, h):
    return box[0]+x, box[1]+y, box[2]+w, box[3]+h


def getEuclid(center, prevCenter):
    return np.sqrt(np.square(center[0] - prevCenter[0]) + np.square(center[1] - prevCenter[1]))

##########################################################################################
# This function sets up the variables for opening the proper pickles files and saving the
# results to the pickleLocation
# processNum is the number of processes used. Making this more than the number of cores
# on your computer will cause problems. On the laptop this was tested with, 3 was slightly
# fasterthan 2, but the computer was unusable while the processing was running if 3 cores
# were used
# The actual processing part of this file is in detectWaggles
def detectClusterWaggles(filePath, pickleLocation, deleteCleaned=True, processNum=2):
    print('Beginning detectClusterWaggles')
    # Get all of the file name and path information:
    drive, path_and_file = os.path.splitdrive(filePath)
    path, fileName = os.path.split(path_and_file)
    prefix = fileName.split('.')[0]
    pickleJar = '{}/{}-Cleaned'.format(pickleLocation, prefix)
    mainDataFrame = pd.read_pickle(
        '{}/WaggleDetections-{}-Cleaned.pkl'.format(pickleLocation, prefix))
    clusterNums = list(mainDataFrame['cluster'].unique())

    # pickleOpenLocations = []
    # filePaths = []

    # for i in clusterNums:
    #     pickleOpenLocation = '{}/WaggleDetections-{}-Cluster{}-Cleaned.pkl'.format(
    #         pickleJar, prefix, i)
    #     pickleOpenLocations.append(pickleOpenLocation)
    #     filePaths.append(filePath)

    pickleOpenLocations = [f'{pickleJar}/WaggleDetections-{prefix}-Cluster{i}-Cleaned.pkl' for i in clusterNums]
    filePaths = [filePath] * len(clusterNums)

    dataFrames = []

    global mask

    mask = zeroMask(filePath)
    masks = [mask] * len(clusterNums)

    ##########################################################################
    # Creates a pool of processes and maps them to the inputs of detectWaggles
    with Pool() as pool:
        dataFrames.append(pool.starmap(
            detectWaggles, zip(masks, filePaths, pickleOpenLocations)))

    fullDataFrame = pd.DataFrame()

    # Move the results into one data frame
    for i in dataFrames:
        fullDataFrame = fullDataFrame.append(i)

    fullDataFrame = fullDataFrame.sort_values(
        by=['cluster']).reset_index(drop=True)

    print('saving full data frame')
    fullDataFrame.to_pickle(
        '{}/WaggleRuns-{}.pkl'.format(pickleLocation, prefix))

    # Can delete the old pickles once they have been processed
    if deleteCleaned:
        for i in pickleOpenLocations:
            print('deleting pickle: {}'.format(i))
            os.remove(i)
        try:
            os.rmdir(pickleJar)
            print('Deleting clean pickle jar')
        except OSError:
            pass

##################################################################################
# This function determines if the movement within a given cluter is a bee waggling
def detectWaggles(mask, filePath, pickleOpenLocation, makeJar=True):
    drive, path_and_file = os.path.splitdrive(filePath)
    path, fileName = os.path.split(path_and_file)
    prefix = fileName.split('.')[0]

    dataFrame = pd.read_pickle(pickleOpenLocation)
    dataFrame.drop('index', axis=1, inplace=True)

    finalDataFrame = pd.DataFrame(
        columns=['x', 'y', 'frame', 'contour', 'bbox', 'size', 'angle', 'euclid', 'cluster'])

    cap = cv2.VideoCapture('{}'.format(filePath))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    threshMin, threshMax = 120, 220

    for i in dataFrame['cluster'].unique():
        print('On Cluster: {}'.format(i))

        clust = dataFrame[dataFrame['cluster'] == i].reset_index()
        start = clust.iloc[0, :]['frame']
        end = clust.iloc[-1, :]['frame']
        cluster = clust.iloc[0, :]['cluster']

        ##################################################################################
        # find what frames the bee should be present in
        # arange(start, stop, step) creates a range of evenly spaced values [start, stop)
        # https://numpy.org/doc/stable/reference/generated/numpy.arange.html
        frameRange = np.arange(start, end, 1)

        # noise might have made the bee not visible in some frames
        # find frames where the bee wasn't detected
        missingFrames = list(set(frameRange) - set(clust.frame.values))

        # open the video
        counter = start
        cap = cv2.VideoCapture('{}'.format(filePath))

        # first parameter of set comes from
        # https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d
        # https://stackoverflow.com/q/11420748
        # so set(1) is cv.CAP_PROP_POS_FRAMES - 0-based index of the frame to be decoded/captured next.
        cap.set(1, start)
        ret, frame = cap.read()

        gray = cv2.GaussianBlur(cv2.cvtColor(
            frame, cv2.COLOR_BGR2GRAY), (15, 15), 1)

        thresh = cv2.threshold(
            gray, threshMin, threshMax, cv2.THRESH_BINARY)[1]

        #######################################################################################################
        # kernel is a 2x2 array of integer 1s to use as a kernel
        # could use getStructuringElement() instead?
        # https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#gac342a1bb6eabf6f55c803b09268e36dc
        kernel = makeKernel()

        ########################################################################################################
        # morphology:
        #   https://docs.opencv.org/master/d9/d61/tutorial_py_morphological_ops.html
        #   https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#ga67493776e3ad1a3df63883829375201f
        #   first erode then dilate to remove noise
        #   then erode again for some reason
        opening = cv2.erode(cv2.morphologyEx(
            thresh, cv2.MORPH_OPEN, kernel, iterations=2), kernel, iterations=1)

        ########################################################################################
        # 'slinear' is spline interpolation of 1st order
        # interpolate the x and y values in the cluster
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html
        # https://en.wikipedia.org/wiki/Spline_interpolation
        fx, fy = interp(clust.frame, clust.x, clust.y, 'slinear')

        # get the current x and y values and create a box around them
        x, y = clust.iloc[0, :]['x'], clust.iloc[0]['y']
        bbox = resetBox(x, y)

        try:
            #######################################################################################################
            # see if there is a contour present in the current bounding box using the 'opened' version of the frame
            # if can't find a contour, then see if the thresholded frame can find one
            # if a contour is found, get its center to get an accurate set of coordinates for the contour
            # if there is an issue in this process such as repeatedly not being able to find a contour,
            # catch the exception and continue
            contour = findROIContour(opening, bbox, mask)
            if contour is None:
                contour = findROIContour(thresh, bbox, mask)
                opening = thresh
            center = getContourMomentInt(contour)
            contour = findFullContour(opening, center, mask)
        except:
            return pd.DataFrame()

        # if the found contour is too large, then try to fix it
        if cv2.contourArea(contour) > clust['size'].max():
            contour = findROIContour(opening, bbox, mask)

        # fit a box around the contour and rotate the box to 0degrees
        rect, box = getFittedBox(contour)
        # bbox is in form: x, y, w, h
        bbox = rotatedBoxConverter(box)

        prevBBox = bbox
        prevCenter = center
        found = True
        # bbox[2]*bbox[3] = w*h
        avg = bbox[2]*bbox[3]
        total = avg
        rois = []

        # scroll through the frames of this cluster
        while counter < end:
            counter += 1
            ret, frame = cap.read()

            gray = cv2.GaussianBlur(cv2.cvtColor(
                frame, cv2.COLOR_BGR2GRAY), (15, 15), 1)

            thresh = cv2.threshold(
                gray, threshMin, threshMax, cv2.THRESH_BINARY)[1]
            kernel = makeKernel()

            opening = cv2.erode(cv2.morphologyEx(
                thresh, cv2.MORPH_OPEN, kernel, iterations=2), kernel, iterations=1)

            # if counter isn't in missingFrames, then we're looking at data from the DF rather than interpolated
            if counter not in missingFrames:
                waggle = clust[clust['frame'] == counter].reset_index()
                x, y = waggle.loc[0, 'x'], waggle.loc[0, 'y']
                bbox = resetBox(x, y)
            # if using interpolated data, base the new bbox on the previous bbox
            # the first frame of a cluster can't be interpolated, so we must have a prevBBox to work with
            else:
                bbox = adjustBox(prevBBox, -5, -5, 10, 10)

            # if the bounding box goes out of video frame, stop tracking
            if checkOOB(bbox, height, width):
                finalDataFrame.loc[len(finalDataFrame)] = 0
                break

            low = threshMin

            ####################################################################
            # Multiple passes over the frame are done using the bounding box
            # to try to get the most accurate path of the waggle
            # These while loops could be moved to a separate function, but
            # it would require a function of a large number of variables, which
            # could be nearly as unwieldy

            while checkContour(contour):
                low -= 5
                thresh = cv2.threshold(
                    gray, low, threshMax, cv2.THRESH_BINARY)[1]
                bbox = roundBox(bbox)
                b0, b1, b2, b3 = bbox
                opening[b1:b1+b3, b0:b0+b2] = thresh[b1:b1+b3, b0:b0+b2]
                contour = findROIContour(opening, bbox, mask)
                if checkLow(low):
                    break

            roiContour = contour
            center = getContourMomentInt(contour)
            bbox = resetBox(center[0], center[1])
            contour = findROIContour(opening, bbox, mask)

            if checkOOB(bbox, height, width):
                finalDataFrame.loc[len(finalDataFrame)] = 0
                break
            low = threshMin

            while checkContour(contour):
                low -= 5
                thresh = cv2.threshold(
                    gray, low, threshMax, cv2.THRESH_BINARY)[1]
                b0, b1, b2, b3 = bbox
                opening[b1:b1+b3, b0:b0+b2] = thresh[b1:b1+b3, b0:b0+b2]
                contour = findROIContour(opening, bbox, mask)
                if checkLow(low):
                    break
            rect, box = getFittedBox(contour)

            found, bbox = anchorInterpolation(bbox, fx, fy, counter)

            if checkOOB(bbox, height, width):
                finalDataFrame[len(finalDataFrame)] = 0
                break

            if not found:
                contour = findROIContour(opening, bbox, mask)
                low = threshMin
                while contour is None:
                    low -= 5
                    thresh = cv2.threshold(
                        gray, low, threshMax, cv2.THRESH_BINARY)[1]
                    b0, b1, b2, b3 = bbox
                    opening[b1:b1+b3, b0:b0+b2] = thresh[b1:b1+b3, b0:b0+b2]
                    contour = findROIContour(opening, bbox, mask)
                    if checkLow(low):
                        break
                rect, box = getFittedBox(contour)
            angle = rect[-1]
            size = cv2.contourArea(contour)

            euclid = getEuclid(center, prevCenter)

            # Fill output df
            finalDataFrame.loc[len(finalDataFrame)] = [
                center[0], center[1], counter, contour, bbox, size, angle, euclid, cluster]

            # Track direction of box movement
            movement = moveDirection(prevBBox, bbox)
            prevCenter = center
            prevBBox = bbox
            # Track avg size of bounding box
            total, avg = avgArea(bbox, total, (counter-start))

        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return finalDataFrame
