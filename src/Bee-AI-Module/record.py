'''
Record, save, and process footage in parallel
'''
import sys
import time
import re
from recorder import WebcamRecorder
import subprocess

'''
Currently unused but can be used to record for a period of time

Outputs: length of time to record in seconds
'''


def getSeconds():
    # Get parameters
    args = sys.argv[1:]

    # Check if parameters exists
    if len(args) == 0:
        raise Exception("Time must be indicated in parameter: -t hh:mm:ss")

    # Find the time parameter
    index = args.index('-t') + 1

    # Get time for hh:mm:ss format
    hours, minutes, seconds = map(int, re.findall(r'\d+', args[index]))

    # Calculate number of seconds
    length = hours * 60 * 60 + minutes * 60 + seconds

    # Make sure length of time is positive
    if length <= 0:
        raise Exception('Time must be greater than 0 seconds')

    return length


if __name__ == "__main__":
    # seconds = getSeconds()

    print("[Event] Starting Capture")

    # Create video capture object
    cam = WebcamRecorder()

    # Initialize counter for tracking number of clips recorded
    counter = 1

    # Save current time to calculate time passed later
    start = time.time()

    while True:
        # Save current time
        now = time.time()

        # Check if enough time has passed for saving and creating a new clip
        if now - start > 60:
            # Create new video writer
            cam.changeWriter()

            # Reset starting time
            start = time.time()

            print(f'[Info] Clip {counter} Created')

            # Process the newly saved clip
            subprocess.Popen(
                ['python3',
                 'runAllParallel.py',
                 f'inputVideo/{time.strftime("%m_%d_%Y")}-{counter}.avi'],
                stdout=subprocess.PIPE)

            # Increase clip counter
            counter += 1

    print('[Event] Ending Capture')
