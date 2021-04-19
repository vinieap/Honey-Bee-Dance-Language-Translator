##################################################################################################
# This file saves the waggle dances to their own video files and updates the UI files accordingly

import pandas as pd
import numpy as np
import cv2
import os
import json
import time
import platform

######################################################################
# DanceBox is used for determining how to crop the video
# padding is how much extra space around the dance coordinates is used
def getDanceBox(dance, padding):
    return dance.x0.min() - padding, dance.x1.max() + padding, dance.y0.min() - padding, dance.y1.max() + padding

##################################################################################
# Save waggle dances are their own video clips in the .mp4 format
# Update the UI to be able to display the videos and their associated information
def saveBees(filePath, pickleLocation, videoSaveLocation, UIfilePath, outputTextFileLocation, visual=False):
    print(np.__file__)
    print('Beginning saveBees')
    
    # Setting up file name and path variables
    drive, path_and_file = os.path.splitdrive(filePath)
    path, fileName = os.path.split(path_and_file)
    prefix = fileName.split('.')[0]
    dataFrame = pd.read_pickle(
        '{}/WaggleRunsFinal-{}.pkl'.format(pickleLocation, prefix))
    cap = cv2.VideoCapture('{}'.format(filePath))

    # Setting up video fps and scale information
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    danceNum = -1

    # currently this code only saves the dances that don't have a 'dance' column of NaN
    dataFrame = dataFrame[dataFrame['dance'] == dataFrame['dance']]

    plt = platform.system()

    # Video saving settings determine on OS
    if plt == 'Windows':
        fourcc = -1
    else:
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')

    # Make sure the folder for saving text output exists
    try:
        os.mkdir(outputTextFileLocation)
    except OSError:
        pass

    danceList = []

    for dance in dataFrame.dance.unique():
        # Look at the list of dances and decide what coordinates to use for cropping
        danceNum += 1
        dance = dataFrame[dataFrame['dance'] == dance]
        print(dance)
        danceList.append(dance)
        x0, x1, y0, y1 = getDanceBox(dance, 100)
        frame0 = dance.frame0.min()

        frame0 = max(0, frame0)
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(x1, width)
        y1 = min(y1, height)

        # Make sure the folder for saving video clips exists
        try:
            os.mkdir(videoSaveLocation)
        except OSError:
            pass

        # Save a clip of the current dance
        outVideoName = "{}/{}-{}.mp4".format(
            videoSaveLocation, prefix, danceNum)
        outVideoNameWithoutLocation = "{}-{}.mp4".format(prefix, danceNum)
        out = cv2.VideoWriter(
            outVideoName, fourcc, fps, (x1-x0, y1-y0))

        # Write down the text information about the video
        danceFileName = '{}/{}-{}-danceData.txt'.format(
            outputTextFileLocation, prefix, danceNum)

        with open(danceFileName, 'w') as f:
            f.write('vidName\t{}-{}.mp4\n'.format(prefix, danceNum))
            f.write('origName\t{}\n'.format(fileName))
            f.write('recordDate\t{}\n'.format(
                time.ctime(os.path.getctime(filePath))))
            f.write('procDate\t{}\n'.format(time.ctime(int(time.time()))))
            f.write('FPS\t{}\n'.format(fps))
            f.write('numWaggles\t{}\n'.format(len(dance.index)))
            for i in range(len(dance.index)):
                currentRow = dance.iloc[i]
                f.write('waggle{}\t{}\t{}\t{}\t{}\n'.format(
                    i+1, currentRow['frame0'], currentRow['frame1'], int(currentRow['angle']), currentRow['direction']))
        f.close()

        # Write the JSON information about the video
        danceDict = {}

        danceDict['vidName'] = f'{prefix}-{danceNum}.mp4'
        danceDict['origName'] = f'{fileName}'
        danceDict['recordDate'] = f'{str(time.ctime(os.path.getctime(filePath)))}'
        danceDict['procDate'] = f'{str(time.ctime(int(time.time())))}'
        danceDict['FPS'] = f'{str(fps)}'
        danceDict['numWaggles'] = f'{str(len(dance.index))}'

        for i in range(len(dance.index)):
            currentRow = dance.iloc[i]
            strWaggle = f'waggle-{i}'
            danceDict[strWaggle] = {}
            danceDict[strWaggle]['init_frame'] = str(currentRow['frame0'])
            danceDict[strWaggle]['final_frame'] = str(currentRow['frame1'])
            danceDict[strWaggle]['angle'] = str(currentRow['angle'])
            danceDict[strWaggle]['direction'] = str(currentRow['direction'])

        # Put the JSON data in the json file
        danceFileNameJSON = '{}/{}-{}-danceData.json'.format(
            outputTextFileLocation, prefix, danceNum)

        with open(danceFileNameJSON, 'w') as f:
            json.dump(danceDict, f)

        with open(UIfilePath, 'r') as f:
            data = json.load(f)
            data = set(data)
            data.add(outVideoNameWithoutLocation)
            data = list(data)
        with open(UIfilePath, 'w') as f:
            json.dump(data, f)


        # More setting for the output video, including visual for
        # showing the video as it is saved and drawing a rectangle around the 
        # detected bee waggling
        cap.set(1, frame0-5)
        counter = frame0 - 5
        run = False
        while True:
            ret, frame = cap.read()
            if counter in list(dance.frame0):
                run = True
            elif counter in list(dance.frame1):
                run = False

            if run:
                loc = np.where(((dance['frame0'] <= counter).values *
                                (dance['frame1'] >= counter).values))[0][0]
                loc = dance.iloc[loc, :]
                cv2.rectangle(frame, (loc.x0 - 20, loc.y0 - 20),
                              (loc.x1 + 20, loc.y1 + 20), (131, 215 ,252), 3)

            out.write(frame[y0:y1, x0:x1])
            if visual:
                cv2.imshow('frame', frame)
                cv2.waitKey(1)

            counter += 1
            if counter-10 >= dance.iloc[-1, :]['frame1']:
                out.release()
                break

        if counter > 200:
            break

    cap.release()
    cv2.destroyAllWindows()
