#######################################################################################
# This file detects bee movement in the input video
# The method used here is described in:
# https://www.biorxiv.org/content/biorxiv/early/2020/11/22/2020.11.21.354019.full.pdf
# the base code for implementing this processing method is from:
# https://github.com/Jreece18/WaggleDanceTracker
# This github was used as a basis for most of the code in the video processsing
# component of this program. The saveWaggles and clusterWaggles files follow this
# code especially closely

import cv2
import pandas as pd
from sklearn.cluster import DBSCAN
import os
import numpy as np
import numba as nb

# Sensitivity for frame difference
# Lower, more waggles detected
# Higher, fewer waggles detected
CONTOUR_SIZE = 240

# Settings for visual mode
GRAYSCALED_TEXT = 'Grayscaled'
GAUSSIAN_TEXT = 'Gaussian Blur'
THRESHOLD_TEXT = 'Threshold'
DIFF_TEXT = 'Frame Difference'

GRAYSCALED_SIZE = cv2.getTextSize(GRAYSCALED_TEXT, cv2.FONT_HERSHEY_SIMPLEX, 6, 8)[0]
GAUSSIAN_SIZE = cv2.getTextSize(GAUSSIAN_TEXT, cv2.FONT_HERSHEY_SIMPLEX, 6, 8)[0]
THRESHOLD_SIZE = cv2.getTextSize(THRESHOLD_TEXT, cv2.FONT_HERSHEY_SIMPLEX, 6, 8)[0]
DIFF_SIZE = cv2.getTextSize(DIFF_TEXT, cv2.FONT_HERSHEY_SIMPLEX, 6, 8)[0]

################################
# Find center of a given contour
def getContourMoment(c):
    # moments are a way to calculate the centroid of a contour
    m = cv2.moments(c)
    # find coordinates and size of the contour
    x = m['m10']/m['m00']
    y = m['m01']/m['m00']
    return x, y

# run time improvement using numba
@nb.njit
def parent_contour_filter(a):
    mask = (a[:, 2] < 0) & (a[:, 3] < 0)
    return mask

def size_vector(a):
    sizes = np.vectorize(cv2.contourArea)(a)
    return a[sizes > CONTOUR_SIZE]

########################
# Find innermost contour
def findChildContours(frame, contours, hierarchy, frameCount):
    # child contours are innermost shapes - the contour of a filled object should
    # have no child or parent in the contour hierarchy
    # xData, yData, sizes = [], [], []
    # hierarchy[0] is the whole frame - we take its children
    childHierarchy = hierarchy[0]
    # keep only the innermost contours with no child contours


    contours2 = np.array(contours)

    hierarchy_filtered = parent_contour_filter(childHierarchy)
    contour_hierarchy = np.ma.masked_array(contours2, ~hierarchy_filtered)
    size_filtered = size_vector(contours2[hierarchy_filtered])

    xData, yData, sizes = zip(*[(*getContourMoment(c), cv2.contourArea(c)) for c in size_filtered])

    # Draw Contours
    # cv2.drawContours(frame, size_filtered, -1, (0, 255, 0), 3)
    # frame = cv2.resize(frame, (640, 640))
    # cv2.imshow('Frame', frame)

    # save what frame this contour is from
    frameCountData = [frameCount]*len(xData)
    # return this data as a list of tuples
    return list(zip(xData, yData, frameCountData)), list(zip(xData, yData, sizes, frameCountData))

###############################################
# Detect bees in the video given by filePath
# Save output pickles to pickleLocation
# Show visualization if visual
def detectBees(filePath, pickleLocation, visual=False):
    drive, path_and_file = os.path.splitdrive(filePath)
    path, fileName = os.path.split(path_and_file)
    prefix = fileName.split('.')[0]
    displayVideos = visual

    foundVideo = True
    # open video file
    try:
        cap = cv2.VideoCapture(filePath)
    except:
        print('video file does not exist or was mispelled')
        return False

    # store video dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width == 0 or height == 0:
        foundVideo = False

    print('Beginning detectBees with file: {}'.format(filePath))

    if (not foundVideo):
        print('Error opening video')
        return False

    if displayVideos:
        # text overlay settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        textCoord = (40, 40)
        fontScale = 1
        fontColor = (150, 0, 0)
        lineType = 2

        # keyPressDelay is used for quitting out of the windows
        keyPressDelay = 1

    prevFrame = None
    counter = 0
    waggles, wagglesWithSizes = [], []

    while True:
        counter += 1
        # read gets the next frame of video
        # if retVal is false, then the video is over
        # https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html#a473055e77dd7faa4d26d686226b292c1
        ret, frame = cap.read()
        if not ret:
            break

        # next few lines follow the background subtraction process described in:
        # https://www.biorxiv.org/content/biorxiv/early/2020/11/22/2020.11.21.354019.full.pdf
        # main pre-processing stages:(a) conversion to from RGB colour to grayscale,
        #                           (b) low-pass filtering with a Gaussian blur,
        #                           (c) background subtraction and thresholding
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayGauss = cv2.GaussianBlur(grayscale, (21, 21), 0)

        # cv2 has a specific function for background subtraction - look into using that
        # threshold function used for determining when to make a pixel black vs white when converting to black and white for binary functions
        # https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html
        # https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57
        # threshold stores the B&W frame
        threshold = cv2.threshold(grayGauss, 108, 230, cv2.THRESH_BINARY)[1]

        # if first frame, then prevFrame is none
        if prevFrame is None:
            prevFrame = threshold
        currentFrame = threshold

        frameDiff = cv2.absdiff(currentFrame, prevFrame)

        # underscore used as dummy variable
        # https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html#gadf1ad6a0b82947fa1fe3c3d497f260e0
        contours, hierarchy = cv2.findContours(
            frameDiff, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        if hierarchy is not None:
            findWaggles, findWagglesSizes = findChildContours(
                frame, contours, hierarchy, counter)
            waggles += findWaggles
            wagglesWithSizes += findWagglesSizes

        if displayVideos:
            grayscale_temp = grayscale.copy()
            grayGauss_temp = grayGauss.copy()
            threshold_temp = threshold.copy()
            frameDiff_temp = frameDiff.copy()

            grayscale_temp = cv2.cvtColor(grayscale_temp, cv2.COLOR_GRAY2BGR)
            grayGauss_temp = cv2.cvtColor(grayGauss_temp, cv2.COLOR_GRAY2BGR)
            threshold_temp = cv2.cvtColor(threshold_temp, cv2.COLOR_GRAY2BGR)
            frameDiff_temp = cv2.cvtColor(frameDiff_temp, cv2.COLOR_GRAY2BGR)
    
            cv2.putText(grayscale_temp, GRAYSCALED_TEXT, ((grayscale.shape[1] - GRAYSCALED_SIZE[0]) // 2, 250), cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 255, 0), 8)
            cv2.putText(grayGauss_temp, GAUSSIAN_TEXT, ((grayGauss.shape[1] - GAUSSIAN_SIZE[0]) // 2, 250), cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 255, 0), 8)
            cv2.putText(threshold_temp, THRESHOLD_TEXT, ((threshold.shape[1] - THRESHOLD_SIZE[0]) // 2, 250), cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 255, 0), 8)
            cv2.putText(frameDiff_temp, DIFF_TEXT, ((frameDiff.shape[1] - DIFF_SIZE[0]) // 2, 250), cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 255, 0), 8)

            # add frame counter to the video
            # cv2.putText(threshold, str(counter), textCoord,
            #            font, fontScale, fontColor, lineType)

            # cv2.putText(grayscale, 'Grayscaled', (height//2, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 4, cv2.LINE_AA)

            row1 = np.concatenate((grayscale_temp, grayGauss_temp), axis=1)
            row2 = np.concatenate((threshold_temp, frameDiff_temp), axis=1)

            allFrames = np.concatenate((row1, row2), axis=0)

            allFrames = cv2.resize(allFrames, (1280, 1280))

            cv2.imshow('Process', allFrames)

            # show the windows of everything happening
            # cv2.imshow('Thresholded', threshold)
            # cv2.imshow('Frame Diff', frameDiff)
            # cv2.imshow('Grayscale', grayscale)
            # cv2.imshow('GrayBlur', grayGauss)

            # waitKey waits for 1ms (if you are holding down q, it will stop)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

        prevFrame = currentFrame

    lastFrame = counter-1
    # close the file or capturing device
    cap.release()

    # close the windows of the video frames
    cv2.destroyAllWindows()

    # Convert all waggle-like activity to data frame
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
    # and apply the clustering algorithm on that data
    waggleDataFrame = pd.DataFrame(wagglesWithSizes, columns=[
        'x', 'y', 'size', 'frame'])

    # Cluster the waggles
    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
    # DBSCAN is a density-based spatial clustering of applications with noise
    # epsilon (eps) = maximum distance between two samples for one to be considered in the neighborhood of the other
    # min_samples is the number of samples (total weight) in a neighborhood for a point to be considered a core point
    # cluster the data by following the patterns in the waggles array
    cluster1 = DBSCAN(eps=25, min_samples=12).fit(waggles)

    # labels_ is:
    #    labels_ndarray of shape (n_samples)
    #    Cluster labels for each point in the dataset given to fit(). Noisy samples are given the label -1.
    # The slice notation here ":, 'Cluster'" allows for setting the value for an entire column
    # this adds a new column called "Cluster" to the data frame and fills it with the cluster data
    waggleDataFrame.loc[:, 'cluster'] = cluster1.labels_
    withoutNoise = waggleDataFrame[waggleDataFrame['cluster'] != -1]
    lastFrameClusters = withoutNoise[withoutNoise['frame']
                                     == lastFrame]['cluster'].tolist()
    waggleDataFrame = withoutNoise
    # remove any clusters that involve data from the last frame of the video - prevents trying to find waggles with incomplete data
    for i in lastFrameClusters:
        waggleDataFrame = withoutNoise[withoutNoise['cluster'] != i]

    # Try to create the pickle jar
    try:
        os.mkdir(pickleLocation)
    except OSError:
        pass

    # serialize data into a file:
    uncleanedFileName = '{}/WaggleDetections-{}.pkl'.format(
        pickleLocation, prefix)
    print('Saving uncleaned data to {}'.format(uncleanedFileName))
    waggleDataFrame.to_pickle(uncleanedFileName)

    return True
