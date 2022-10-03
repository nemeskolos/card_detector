
import cv2
import numpy as np
import time
import os
import Cards
import VideoStream


# Konstans és változók

## Kamera
IM_WIDTH = 1280
IM_HEIGHT = 720 
FRAME_RATE = 10

## Kistzámolt frame rate számolása
frame_rate_calc = 1
freq = cv2.getTickFrequency()

## Használandó betűtípus meghatározása
font = cv2.FONT_HERSHEY_SIMPLEX

videostream = VideoStream.VideoStream((IM_WIDTH,IM_HEIGHT),FRAME_RATE,2,0).start()
time.sleep(1) 


path = 'Card_Imgs/'
train_ranks = Cards.load_ranks( path)
train_suits = Cards.load_suits( path)



# A fő ciklus ismételten megragadja a képkockákat a videófolyamból, és feldolgozza azokat, hogy megtalálja és azonosítsa a játékkártyákat.

cam_quit = 0 # Loop control variable

# Képkockák rögzítésének megkezdése
while cam_quit == 0:

    # Képkocka kinyerése a videófolyamból
    image = videostream.read()
    t1 = cv2.getTickCount()

    # Kamera képek előfeldolgozása (gray, blur,  threshold)
    pre_proc = Cards.preprocess_image(image)
	
    # A képen lévő összes kártya kontúrjának megkeresése és rendezése (query cards)
    cnts_sort, cnt_is_card = Cards.find_cards(pre_proc)

    # Ha nincs kontúr akkor ne csináljon semmit
    if len(cnts_sort) != 0:

        # Egy új "kártyák" lista inicializálása a kártyaobjektumok hozzárendeléséhez.
        # k indexek az új kártya tömböknek
        cards = []
        k = 0

        # Adott kontúr detektálása
        for i in range(len(cnts_sort)):
            if (cnt_is_card[i] == 1):

                cards.append(Cards.preprocess_card(cnts_sort[i],image))

                # Keresse meg a kártyának legjobban megfelelő rangot és színt.
                cards[k].best_rank_match,cards[k].best_suit_match,cards[k].rank_diff,cards[k].suit_diff = Cards.match_card(cards[k],train_ranks,train_suits)

                # Középpont rajzolása 
                image = Cards.draw_results(image, cards[k])
                k = k + 1
	    
        # Kontúrok rajzolása
        if (len(cards) != 0):
            temp_cnts = []
            for i in range(len(cards)):
                temp_cnts.append(cards[i].contour)
            cv2.drawContours(image,temp_cnts, -1, (255,0,0), 2)
        
        
    cv2.putText(image,"FPS: "+str(int(frame_rate_calc)),(10,26),font,0.7,(255,0,255),2,cv2.LINE_AA)

  
    cv2.imshow("Card Detector",image)

    # Frame rate kalkuláció
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc = 1/time1
    
    key = cv2.waitKey(250) & 0xFF
    if key == ord("q"):
        cam_quit = 1

cv2.destroyAllWindows()
videostream.stop()

