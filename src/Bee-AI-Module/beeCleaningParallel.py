#########################################################################
# This file takes the output data from beeDetector and 'cleans' it
# to remove duplicate data or data points that are definitely not waggles
# To make this code optimized for parallel processing, each cluster of 
# potential waggles is saved as its own file inside of the pickleJar.
# This allows for waggleTrackerParallel to run multiple clusters at once

import pandas as pd
import numpy as np
import os

def cleanBees(filePath, pickleLocation, makeJar=True):
    print('Beginning cleanBees')
    drive, path_and_file = os.path.splitdrive(filePath)
    path, fileName = os.path.split(path_and_file)
    prefix = fileName.split('.')[0]
    waggleDataFrame = (pd.read_pickle('{}/WaggleDetections-{}.pkl'.format(pickleLocation, prefix))
                       ).sort_values(by=['cluster', 'frame']).reset_index(drop=True)

    # duplicates are rows where frame and cluster values are the same, i.e.
    # where frame1 = frame2 and Cluster1 = Cluster2
    # using waggleDataFrame[...] gives the elements in waggleDataFrame with the indices found by duplicated
    # without using the outer waggleDataFrame[] brackets, the array would just be 2D array of indices and booleans
    dupes = waggleDataFrame[waggleDataFrame.duplicated(
        subset=['frame', 'cluster'], keep=False)]
    nonDupes = waggleDataFrame[~waggleDataFrame.duplicated(
        subset=['frame', 'cluster'], keep=False)]

    # list every data point that is a duplicate as well as its neighboring data points (a-1, a, a+1)
    a = dupes.index.values
    indices = np.unique(np.concatenate((a, a-1, a+1)))

    # index is the row labels of the DataFrame - essentially the list of all data points
    # isin(idx) finds values in idx that are in waggleDataFrame
    # so df gets the values in waggleDataFrame with those indices
    # .index.isin seems to be used to prevent trying to get invalid indices
    # the second call to reset_index adds the level_0 column
    newDataFrame = waggleDataFrame[waggleDataFrame.index.isin(
        indices)].reset_index().reset_index()

    # stores the rows of newDataFrame using level_0 indexing as arrays (points is an array of arrays)
    points = newDataFrame[['level_0', 'x',
                           'y', 'size', 'index', 'frame']].values

    # return_counts returns the number of times each unique item appears in the array
    # find all of the unique elements in the 'frame' column of points[:,-1] which appear >=2 times
    # https://numpy.org/doc/stable/reference/generated/numpy.unique.html
    samePoints = [np.argwhere(i[0] == points[:, -1]) for i in np.array(
        np.unique(points[:, -1], return_counts=True)).T if i[1] >= 2]

    saveRow = []

    for i in samePoints:
        dist = []
        prevPoint = min(i) - 1
        for j in i:
            distPrev = np.sqrt((points[prevPoint, 1] - points[j, 1])
                               ** 2 + (points[prevPoint, 2] - points[j, 2])**2)
            dist.append(distPrev)
        saveRow.append(i[np.argmin(dist)][0])

    finalPoints = points[saveRow]

    finalIndices = np.unique(np.concatenate(
        (nonDupes.index.values, finalPoints[:, 3])))
    waggleDataFrame = waggleDataFrame.reset_index()
    newDataFrame = waggleDataFrame[waggleDataFrame['index'].isin(finalIndices)]

    # find euclidean distance between points in every cluster - the start of each cluster has value 0
    newDataFrame.loc[:, 'euclid'] = np.sqrt(np.square(
        newDataFrame['x'] - newDataFrame['x'].shift(1)) + np.square(newDataFrame['y'] - newDataFrame['y'].shift(1)))
    newDataFrame = newDataFrame.reset_index(drop=True)
    startOfClusters = [0]
    for i in range(1, len(newDataFrame) - 1):
        if newDataFrame.loc[i, 'cluster'] != newDataFrame.loc[i+1, 'cluster']:
            startOfClusters.append(i + 1)
    for i in startOfClusters:
        newDataFrame.loc[i, 'euclid'] = 0

    newDataFrame = newDataFrame.drop(columns=['index']).reset_index()

    # each cluster begins with a NAN, this replaces them with 0s
    newDataFrame.fillna(0, inplace=True)
    newDataFrame = newDataFrame[newDataFrame['euclid']
                                < newDataFrame.euclid.quantile(0.9)]
    
    # try to make pickle folder and save output data to it
    if makeJar:
        pickleJar = '{}/{}-Cleaned'.format(pickleLocation, prefix)
        try:
            print('Making pickleFolder: {}'.format(pickleJar))
            os.mkdir(pickleJar)
        except OSError:
            pass

        for i in list(newDataFrame['cluster'].unique()):
            clust = newDataFrame[newDataFrame['cluster']
                                 == i].reset_index(drop=True)
            pickleName = '{}/WaggleDetections-{}-Cluster{}-Cleaned.pkl'.format(
                pickleJar, prefix, i)
            print('Saving cluster {} to {}'.format(i, pickleName))
            clust.to_pickle(pickleName)

        cleanedFileName = '{}/WaggleDetections-{}-Cleaned.pkl'.format(
            pickleLocation, prefix)
        print('Saving full cleaned data to {}'.format(cleanedFileName))
        newDataFrame.to_pickle(cleanedFileName)
