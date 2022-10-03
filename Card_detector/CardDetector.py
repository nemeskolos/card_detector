

# Import  packages
import cv2
import numpy as np
import time
import os
import Cards
import VideoStream


# Define constants and initialize variables

## Camera settings
IM_WIDTH = 1280
IM_HEIGHT = 720 
FRAME_RATE = 10

## Initialize calculated frame rate 
frame_rate_calc = 1
freq = cv2.getTickFrequency()

## Define font to use
font = cv2.FONT_HERSHEY_SIMPLEX

videostream = VideoStream.VideoStream((IM_WIDTH,IM_HEIGHT),FRAME_RATE,2,0).start()
time.sleep(1) # Give the camera time to warm up


path = 'Card_Imgs/'
train_ranks = Cards.load_ranks( path)
train_suits = Cards.load_suits( path)



# The main loop repeatedly grabs frames from the video stream
# and processes them to find and identify playing cards.

cam_quit = 0 # Loop control variable

# Begin capturing frames
while cam_quit == 0:

    # Grab frame from video stream
    image = videostream.read()
    t1 = cv2.getTickCount()

    # Pre-process camera image (gray, blur, and threshold it)
    pre_proc = Cards.preprocess_image(image)
	
    # Find and sort the contours of all cards in the image (query cards)
    cnts_sort, cnt_is_card = Cards.find_cards(pre_proc)

    # If there are no contours, do nothing
    if len(cnts_sort) != 0:

        # Initialize a new "cards" list to assign the card objects.
        # k indexes the newly made array of cards.
        cards = []
        k = 0

        # For each contour detected:
        for i in range(len(cnts_sort)):
            if (cnt_is_card[i] == 1):

                cards.append(Cards.preprocess_card(cnts_sort[i],image))

                # Find the best rank and suit match for the card.
                cards[k].best_rank_match,cards[k].best_suit_match,cards[k].rank_diff,cards[k].suit_diff = Cards.match_card(cards[k],train_ranks,train_suits)

                # Draw center point and match result on the image.
                image = Cards.draw_results(image, cards[k])
                k = k + 1
	    
        # Draw card contours on image 
        if (len(cards) != 0):
            temp_cnts = []
            for i in range(len(cards)):
                temp_cnts.append(cards[i].contour)
            cv2.drawContours(image,temp_cnts, -1, (255,0,0), 2)
        
        
    cv2.putText(image,"FPS: "+str(int(frame_rate_calc)),(10,26),font,0.7,(255,0,255),2,cv2.LINE_AA)

  
    cv2.imshow("Card Detector",image)

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc = 1/time1
    
    key = cv2.waitKey(250) & 0xFF
    if key == ord("q"):
        cam_quit = 1

cv2.destroyAllWindows()
videostream.stop()

