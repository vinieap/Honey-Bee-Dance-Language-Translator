###############################################################################################
# This file determines if multiple waggles detected are actually part of the same waggle dance
# Bees waggle multiple times in a row when communicating, so this tries to collect those
# multiple waggles together
# This file is almost exactly from this project:
# https://github.com/Jreece18/WaggleDanceTracker
# Additional documentation for some of the functions used in this file are available there

import numpy as np
import pandas as pd
import warnings
from scipy import signal
from scipy import stats
from sklearn.mixture import GaussianMixture
import itertools
import math
import os

warnings.filterwarnings('ignore')

# Find slope of a dataset via linear regression
def getSlope(x, y, frames):
    f = frames.astype('float32')
    x = x.astype('float32')
    y = y.astype('float32')

    slopeX, _, _, _, _ = stats.linregress(f, x)
    slopeY, _, _, _, _ = stats.linregress(f, y)

    return (slopeX, slopeY)

# Remove data points that appear to be outliers, calculated with Gaussian
def removeOutliers(slopes):
    results = GaussianMixture(n_components=1, covariance_type='full',
                              verbose=0).fit_predict(X=np.array(slopes)).tolist()
    mode = stats.mode(results).mode.tolist()[0]
    outliers = [i for i, x in enumerate(results) if x != mode]
    return outliers

# Determine general direction of a movement
def directionQuadrant(distance):
    x, y = distance
    if x >= 0:
        return 1 if y < 0 else 2
    if x < 0:
        return 3 if y >= 0 else 4

# Determine which data points are outliers in a long waggle dance
def findOutliers(xSlopes, ySlopes, xMedian, yMedian):
    x = [i >= 0 for i in xSlopes] if xMedian >= 0 else [i < 0 for i in xSlopes]
    y = [i >= 0 for i in ySlopes] if yMedian < 0 else [i < 0 for i in ySlopes]
    outliers = [a*b for a, b in zip(x, y)]
    return outliers

#############################################################
# Bees waggle multiple times in a row when communicating, 
# so this function to collect those multiple waggles together
# and updates the pickle files accordingly
def clusterWaggles(filePath, pickleLocation):
    print('Beginning clusterWaggles')
    # Setting up variables with file names and prefixes
    drive, path_and_file = os.path.splitdrive(filePath)
    path, fileName = os.path.split(path_and_file)
    prefix = fileName.split('.')[0]
    dataFrame = pd.read_pickle(
        '{}/WaggleRuns-{}.pkl'.format(pickleLocation, prefix))

    # Open the data frame and collect data about where waggles are located, their angles, and how long they last    
    waggles = pd.DataFrame(columns=['xmean', 'ymean', 'x0', 'y0', 'x1', 'y1', 'frame0', 'frame1', 'timeTaken', 'frequency',
                                    'cluster', 'angle', 'distance'])
    for i in dataFrame['cluster'].unique():
        clust = dataFrame[dataFrame['cluster'] == i]
        distance = getSlope(clust.x, clust.y, clust.frame)
        waggles.loc[len(waggles)] = clust.x.mean(), clust.y.mean(), clust.x.min(), clust.y.min(), clust.x.max(), \
            clust.y.max(), clust.frame.min(), clust.frame.max(), clust.frame.max() - clust.frame.min(), \
            len(signal.find_peaks(clust.angle.values)[0]) / ((clust.frame.max() - clust.frame.min() + 1) / 60), \
            clust.cluster.max(), clust.angle.mean(), distance
    waggles = waggles[waggles['timeTaken'] != 0]

    # Look at which data points are within certain distance, time,
    # and angle ranges of the current waggle
    for i in waggles['cluster'].unique():
        target = waggles[waggles['cluster'] == i]
        frame = float(target.frame1)
        angle = float(target.angle)
        # Change this line based on framerate of video 
        # the current camera script is based on 120fps
        search = waggles[waggles['frame0'].between(frame+5, frame+125)]
        search.loc[:, 'xmean'] = search['xmean'] - float(target['xmean'])
        search.loc[:, 'ymean'] = search['ymean'] - float(target['ymean'])
        search.loc[:, 'euclid'] = np.sqrt(
            np.square(search['xmean'])+np.square(search['ymean']))

        minEuclid = search.euclid.min()
        nextWaggle = search[search['euclid'] == minEuclid]

        if len(nextWaggle) > 0:
            waggles.loc[target.index, 'nextCluster'] = int(
                nextWaggle.iloc[0, :]['cluster'])
            waggles.loc[target.index,
                        'nextEuclid'] = nextWaggle.iloc[0, :]['euclid']
            waggles.loc[target.index, 'nextAngleDiff'] = abs(
                angle) - abs(nextWaggle.iloc[0, :]['angle'])

    # More processing for determining if data points are sufficiently close together to be
    # considered within the same cluster
    singleCluster = False
    if(len(waggles.cluster.unique()) > 1):
        dupes = waggles[waggles.duplicated(subset=['nextCluster'], keep=False)]
        nonDupes = waggles[~waggles.duplicated(subset=['nextCluster'], keep=False)]
        nans = waggles[pd.isnull(waggles['nextCluster'])]
        points = dupes[['cluster', 'nextCluster',
                        'nextAngleDiff', 'nextEuclid']].dropna().values
        dupes.head()

        points = points[np.argsort(points[:, 1], axis=0)]
        samePoints = [np.argwhere(i[0] == points[:, 1]) for i in np.array(
            np.unique(points[:, 1], return_counts=True)).T if i[1] >= 2]
        saveRow = []

        for i in samePoints:
            dist = []
            for j in i:
                point = points[j, -1]
                dist.append(point)
            saveRow.append(i[np.argmin(dist)][0])
        finalPoints = points[[saveRow]]

        waggles = dupes[dupes['cluster'].isin(list(finalPoints[:, 0].astype(int)))]
        otherDupes = dupes[~dupes['cluster'].isin(
            list(finalPoints[:, 0].astype(int)))]
        otherDupes.loc[:, ['nextCluster', 'nextEuclid']] = np.nan
        waggles = pd.concat([waggles, nonDupes, nans, otherDupes]
                            ).sort_index().drop_duplicates()
        waggles.head()

        # Use quantile to determine if data points are 'close enough'
        quantile = waggles.nextEuclid.quantile(0.85)

        waggles.loc[:, 'nextCluster'] = np.where(
            (waggles['nextEuclid'] >= quantile), np.nan, waggles['nextCluster'])
        waggles.loc[:, 'nextEuclid'] = np.where(
            (waggles['nextEuclid'] >= quantile), np.nan, waggles['nextEuclid'])
        index = waggles.cluster.tolist()
        nextIndex = waggles.nextCluster.tolist()
    else:
        singleCluster = True
        index = waggles.cluster.tolist()
        nextIndex = []
        # print(waggles)

    final = []

    for i in index:
        if i in itertools.chain(*final):
            continue

        currentNo = i
        currentIndex = index.index(currentNo)
        currentList = []
        currentList.append(currentNo)

        while not (math.isnan(currentNo)):
            if currentNo > (len(index) - 1):
                break
            if singleCluster or math.isnan(nextIndex[currentIndex]):
                break
            else:
                currentNo = int(nextIndex[currentIndex])
                currentIndex = int(index.index(currentNo))
                currentList.append(currentNo)
        final.append(currentList)

    longDances = [x for x in final if len(x) > 2]
    shortDances = [x for x in final if len(x) == 2]

    # Check that there actually are dances present before trying to cluster
    if (len(longDances) == 0 and len(shortDances) == 0):
        print('No dances found')
        return False

    # Classify dances as either long or short
    finalShortDances = []
    for dance in shortDances:
        distance = []
        angle = []
        nextEuclid = []
        for run in dance:
            waggle = waggles[waggles['cluster'] == run].iloc[0, :]
            distance.append(waggle.distance)
            angle.append(waggle.angle)
            nextEuclid.append(waggle.nextEuclid)

        try:
            if nextEuclid[0] >= 100:
                continue
            if distance[0][0] not in np.arange(distance[1][0]-0.5, distance[1][0]+0.5):
                continue
            if distance[0][1] not in np.arange(distance[1][1]-0.5, distance[1][1]+0.5):
                continue
            if angle[0] not in np.arange(angle[1]+10, angle[0]-10):
                continue
        except ValueError:
            continue

        finalShortDances.append(dance)

    for i, dance in enumerate(longDances):
        for run in dance:
            waggles.loc[waggles[waggles['cluster'] == run].index, 'dance'] = i

    waggles[waggles['dance'].notna()]['dance'].unique()
    waggles.head()

    # Fill in more information about the paths of the dances
    for i in waggles[waggles['dance'].notna()]['dance'].unique():
        dataFrame = waggles[waggles['dance'] == i]
        clusters = list(dataFrame.loc[:, 'cluster'])

        slopes = dataFrame.distance.tolist()
        slopeX, slopeY = [i[0] for i in slopes], [i[1] for i in slopes]
        slopeXmedian, slopeYmedian = np.median(slopeX), np.median(slopeY)
        slopesMed = findOutliers(slopeX, slopeY, slopeXmedian, slopeYmedian)

        gm = GaussianMixture(n_components=2, covariance_type='tied',
                             verbose=0).fit_predict(X=np.array(slopes))
        gmCount = stats.mode(gm)[1][0]
        gmMode = stats.mode(gm)[0][0]
        gmBool = [i == gmMode for i in list(gm)]

        combinedAvg = [a+b for a, b in zip(gmBool, slopesMed)]

        idx = []
        if combinedAvg.count(False) >= 1 and gmBool.count(False) <= len(gmBool)/2:
            idx = [i for i, x in enumerate(combinedAvg) if x == False]

        outliers = list(dataFrame.iloc[idx, :]['cluster'])

        if len(outliers) >= 1:
            waggles.loc[waggles[waggles['cluster'].isin(
                outliers)].index, 'dance'] = np.nan

        if combinedAvg.count(False) >= 3:
            waggles.loc[waggles[waggles['cluster'].isin(idx)].index, 'dance'] = len(
                waggles['dance'].unique())

    longDancesNew = []
    for i in waggles['dance'].unique():
        dataFrame = waggles[waggles['dance'] == i]

    waggles.loc[:, 'direction'] = waggles['distance'].apply(directionQuadrant)
    if not singleCluster:
        waggles.drop(['xmean', 'ymean', 'nextCluster', 'nextEuclid',
            'nextAngleDiff'], axis=1, inplace=True)

    # Save all resulting pickles
    waggles.to_pickle(
        '{}/WaggleRunsFinal-{}.pkl'.format(pickleLocation, prefix))

    detections = pd.read_pickle(
        '{}/WaggleRuns-{}.pkl'.format(pickleLocation, prefix))
    danceDict = pd.Series(waggles.dance, index=waggles.cluster).to_dict()
    detections.loc[:, 'dance'] = detections['cluster'].map(danceDict)
    detections.to_pickle(
        '{}/WaggleDetectionsFinal-{}.pkl'.format(pickleLocation, prefix))
    return True
