'''
Utility file for multithreaded video capture and saving
 - Allows for 120 FPS video capture

Based on the the imutils.video.webcamvideostream for multithreaded capture
https://github.com/jrosebr1/imutils/blob/master/imutils/video/webcamvideostream.py

and this stackoverflow answer for multithreaded video writing
https://stackoverflow.com/questions/55494331/recording-video-with-opencv-python-multithreading
'''
from threading import Thread
import cv2
import time


class WebcamRecorder:
    def __init__(self):
        # Counter for changing which file the frames are saved to
        # Allows for parallel recording and processing
        self.counter = 1

        # Setting camera settings
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Creating video writer for saving frames
        self.writer = cv2.VideoWriter(
            f'videos/{time.strftime("%m_%d_%Y")}-{self.counter}.avi',
            cv2.VideoWriter_fourcc(*'XVID'), 60.0, (1280, 720)
        )

        # Creating thread for capturing frames
        # And setting daemon to True for independence
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

        # Start the video capture
        self.start_recording()

    def update(self):
        # While the webcam is on, retrieve the next available frame
        while True:
            if self.cap.isOpened():
                (self.ret, self.frame) = self.cap.read()

    def changeWriter(self):
        # Increase counter for new file name
        self.counter += 1

        # Create new video writer for saving frames to
        self.writer = cv2.VideoWriter(
            f'inputVideo/{time.strftime("%m_%d_%Y")}-{self.counter}.avi',
            cv2.VideoWriter_fourcc(*'XVID'), 60.0, (1280, 720)
        )

    def save_frame(self):
        # Write frames to file
        self.writer.write(self.frame)

    def start_recording(self):
        # Keeps capturing frames until capture is closed
        def start_thread():
            while(True):
                try:
                    self.save_frame()
                except AttributeError:
                    pass

            # Destroy all capture devices, video writers, and and misc. windows
            self.cap.release()
            self.writer.release()
            cv2.destroyAllWindows()

        # Creating seperate video writer thread for independent processing
        self.writing_thread = Thread(target=start_thread, args=())
        self.writing_thread.daemon = True
        self.writing_thread.start()
