################################################################
# This file is used to run the program with parallel processing. 

from beeDetector import detectBees
from beeCleaningParallel import cleanBees
from beeWaggleTrackerParallel import detectClusterWaggles
from clusterWaggles import clusterWaggles
from saveWaggles import saveBees
from multiprocessing import freeze_support
import sys
from glob import glob
import os
import time

#####################################################################
# PickleJars are the folders which store output pickle files.
# If the program crashes or their is old data to be removed,
# this function makes sure pickle folders created by a certain video 
# are deleted. deleteFullPickles deletes the rest of the pickle files
def deletePickleJars(filePath, pickleLocation):
    drive, path_and_file = os.path.splitdrive(filePath)
    path, fileName = os.path.split(path_and_file)
    prefix = fileName.split('.')[0]
    cleanedPickleJar = '{}/{}-Cleaned'.format(pickleLocation, prefix)
    waggleRunPickleJar = '{}/{}-WaggleRuns'.format(pickleLocation, prefix)
    if os.path.isdir(cleanedPickleJar):
        cleanedPickles = '{}/*.*'.format(cleanedPickleJar)
        for file in glob(cleanedPickles):
            os.remove(file)
        try:
            os.rmdir(cleanedPickleJar)
        except OSError:
            pass
    if os.path.isdir(waggleRunPickleJar):
        wagglePickles = '{}/*.*'.format(waggleRunPickleJar)
        for file in glob(wagglePickles):
            os.remove(file)
        try:
            os.rmdir(waggleRunPickleJar)
        except OSError:
            pass

##################################################################
# This function deletes the pickles stored outside of individual 
# folders
def deleteFullPickles(filePath, pickleLocation):
    drive, path_and_file = os.path.splitdrive(filePath)
    path, fileName = os.path.split(path_and_file)
    prefix = fileName.split('.')[0]
    toDelete = ['{}/WaggleDetections-{}.pkl'.format(pickleLocation, prefix),
                '{}/WaggleDetections-{}-Cleaned.pkl'.format(
                    pickleLocation, prefix),
                '{}/WaggleDetectionsFinal-{}.pkl'.format(
                    pickleLocation, prefix),
                '{}/WaggleRuns-{}.pkl'.format(pickleLocation, prefix),
                '{}/WaggleRunsFinal-{}.pkl'.format(pickleLocation, prefix)]
    for i in toDelete:
        if os.path.isfile(i):
            os.remove(i)

##################################################################
# This function deletes all pickles associated with an input video
def deleteAllPickles(filePath, pickleLocation):
    deleteFullPickles(filePath, pickleLocation)
    deletePickleJars(filePath, pickleLocation)

###########################################################################
# This function attempts to fully process an input video
# filePath is the video file path such as inputVideo/bee.mp4
# pickleLocation is where the processing pickles will be saved
# videoSaveLocation is where the resulting video clips will be saved
# UIfilePath and outputTextFileLocation are used for sending data to the UI
# visual refers to the visualizer for processing the bee videos 
# - slows down performance if True
def getVideos(filePath, pickleLocation, videoSaveLocation, UIfilePath, outputTextFileLocation):
    detectedBees = detectBees(filePath, pickleLocation, visual=False)
    # if no bees detected, stop processing
    if (not detectedBees):
        print('Video not processed, closing')
        deleteAllPickles(filePath, pickleLocation)
        return
    else:
        cleanBees(filePath, pickleLocation)
        detectClusterWaggles(filePath, pickleLocation)
        foundDances = clusterWaggles(filePath, pickleLocation)
        # if no waggle dances found, stop processing
        if (not foundDances):
            print('No dances were found, closing')
            deleteAllPickles(filePath, pickleLocation)
            return
        # if waggle dances found, save the clips of the dances
        else:
            saveBees(filePath, pickleLocation, videoSaveLocation,
                     UIfilePath, outputTextFileLocation, visual=False)

########################################################################
# Function for getting the video file location from the input parameters
def getFileLocation():
    # Eliminate script name
    args = sys.argv[1:]

    # Check if user entered file path
    if len(args) == 0:
        print("Please enter the file path of the video as a parameter")
        sys.exit()

    # Return the file path of the video
    return args[0]

######################################
# Main function
# freeze_support() used for windows
# runtime of code printed at the end
if __name__ == "__main__":
    start_time = time.time()
    freeze_support()
    filePath = getFileLocation()
    # replace the next line with your own path names if needed
    getVideos(filePath, 'pickles', 'outputVideos',
              '.././DB/filenames.txt', '.././DB/danceData')

    end_time = time.time()

    seconds = int(end_time - start_time) % 60
    minutes = int(end_time - start_time) // 60
    print(f"Execution Time: {minutes:02d}:{seconds:02d} minutes")
