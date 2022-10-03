
from threading import Thread
import cv2


class VideoStream:
    """Camera object"""
    def __init__(self, resolution=(640,480),framerate=30,PiOrUSB=2,src=0):

        # Create a variable to indicate if it's a USB camera or PiCamera.
        # PiOrUSB = 1 will use PiCamera. PiOrUSB = 2 will use USB camera.
        self.PiOrUSB = PiOrUSB

        if self.PiOrUSB == 1: # PiCamera
            # Import packages from picamera library
            from picamera.array import PiRGBArray
            from picamera import PiCamera

            # Initialize the PiCamera and the camera image stream
            self.camera = PiCamera()
            self.camera.resolution = resolution
            self.camera.framerate = framerate
            self.rawCapture = PiRGBArray(self.camera,size=resolution)
            self.stream = self.camera.capture_continuous(
                self.rawCapture, format = "bgr", use_video_port = True)

            self.frame = []

        if self.PiOrUSB == 2: # USB camera
            self.stream = cv2.VideoCapture(src)
            ret = self.stream.set(3,resolution[0])
            ret = self.stream.set(4,resolution[1])
            ret = self.stream.set(5,framerate) #Doesn't seem to do anything so it's commented out

            # első frame3
            (self.grabbed, self.frame) = self.stream.read()

    # változó a kamera megállásához
        self.stopped = False

    def start(self):
    # videóból való beolvasás
        Thread(target=self.update,args=()).start()
        return self

    def update(self):

        if self.PiOrUSB == 1: # PiCamera - we did not use this 
            
            # A ciklus a végtelenségig folytatódik, amíg a szál le nem áll.
            for f in self.stream:
                # Ragadja ki a képkockát a folyamból, és törölje a folyamot a következő képkocka előkészítéséhez.
                self.frame = f.array
                self.rawCapture.truncate(0)

                if self.stopped:
                    #kamera forrás leállítás
                    self.stream.close()
                    self.rawCapture.close()
                    self.camera.close()

        if self.PiOrUSB == 2: # USB camera

            # 
            while True:
                if self.stopped:
                    self.stream.release()
                    return

                # kövi framet kapjuk el
                (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # vissza legjobb framéhetz
        return self.frame

    def stop(self):
        # kamerának mutatja hogy álljon le
        self.stopped = True
