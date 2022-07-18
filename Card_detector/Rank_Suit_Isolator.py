import cv2
import numpy as np
import time
import Cards
import os

IM_WIDTH = 1280
IM_HEIGHT = 720

RANK_WIDTH = 70
RANK_HEIGHT = 125

SUIT_WIDTH = 70
SUIT_HEIGHT = 100

i = 1

for Name in ['Ace', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight',
             'Nine', 'Ten', 'Jack', 'Queen', 'King', 'Spades', 'Diamonds',
             'Clubs', 'Hearts']:

    filename = Name + '.jpg'

    print('Következő lap beolvasása :  ' + filename)

    if Name=="Ace":
        image = cv2.imread('Ace.jpg')

    if Name=="Two":
        image = cv2.imread('Two.jpg')

    if Name=="Three":
        image = cv2.imread('Three.jpg')

    if Name=="Four":
        image = cv2.imread('Four.jpg')

    if Name=="Five":
        image = cv2.imread('Five.jpg')

    if Name=="Six":
        image = cv2.imread('Six.jpg')

    if Name=="Seven":
        image = cv2.imread('Seven.jpg')

    if Name=="Eight":
        image = cv2.imread('Eight.jpg')

    if Name=="Nince":
        image = cv2.imread('Nine.jpg')

    if Name=="Ten":
        image = cv2.imread('Ten.jpg')

    if Name=="Jack":
        image = cv2.imread('Jack.jpg')

    if Name=="Queen":
        image = cv2.imread('Queen.jpg')

    if Name=="King":
        image = cv2.imread('King.jpg')

    if Name=="Spades":
        image = cv2.imread('Nine.jpg')

    if Name=="Diamonds":
        image = cv2.imread('Diamond.jpg')

    if Name=="Clubs":
        image = cv2.imread('Four.jpg')

    if Name=="Hearts":
        image = cv2.imread('Heart.jpg')

    # Képek előfeldolgozása
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    retval, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)

    # Kontúr keresése és osztályozásuk méret szerint
    cnts, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # Legnagyobb kontúr - kértya
    flag = 0
    image2 = image.copy()

    if len(cnts) == 0:
        print('Kontúrt nem található!')
        quit()

    card = cnts[0]

    # Sarokpontok közelítése
    peri = cv2.arcLength(card, True)
    approx = cv2.approxPolyDP(card, 0.01 * peri, True)
    pts = np.float32(approx)

    x, y, w, h = cv2.boundingRect(card)

    # 200 x 300 csökkentjük
    warp = Cards.flattener(image, pts, w, h)

    # Sarok kivágása, küszöbérték állítás és nagyítás
    corner = warp[0:84, 0:32]
    #corner_gray = cv2.cvtColor(corner,cv2.COLOR_BGR2GRAY)
    corner_zoom = cv2.resize(corner, (0, 0), fx=4, fy=4)
    corner_blur = cv2.GaussianBlur(corner_zoom, (5, 5), 0)
    retval, corner_thresh = cv2.threshold(corner_blur, 100, 255, cv2.THRESH_BINARY_INV)


    # Elkülöníti a formát vagy a számot/betűt
    if i <= 13:  # Szám/betű
        rank = corner_thresh[20:185, 0:128]  # Grabs portion of image that shows rank
        rank_cnts, hier = cv2.findContours(rank, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        rank_cnts = sorted(rank_cnts, key=cv2.contourArea, reverse=True)
        x, y, w, h = cv2.boundingRect(rank_cnts[0])
        rank_roi = rank[y:y + h, x:x + w]
        rank_sized = cv2.resize(rank_roi, (RANK_WIDTH, RANK_HEIGHT), 0, 0)
        final_img = rank_sized

    if i > 13:  # Kártyaszín
        suit = corner_thresh[186:336, 0:128]  # Grabs portion of image that shows suit
        suit_cnts, hier = cv2.findContours(suit, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        suit_cnts = sorted(suit_cnts, key=cv2.contourArea, reverse=True)
        x, y, w, h = cv2.boundingRect(suit_cnts[0])
        suit_roi = suit[y:y + h, x:x + w]
        suit_sized = cv2.resize(suit_roi, (SUIT_WIDTH, SUIT_HEIGHT), 0, 0)
        final_img = suit_sized

    cv2.imshow("Image", final_img)

    # Példa kép mentése
    cv2.imwrite('Card_Imgs/' + filename, final_img)

    i = i + 1

cv2.destroyAllWindows()
