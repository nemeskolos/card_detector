
import numpy as np
import cv2
import time


# Adaptive threshold
BKG_THRESH = 80
CARD_THRESH = 50

CORNER_WIDTH = 32
CORNER_HEIGHT = 84

# figura dimenziói
RANK_WIDTH = 70
RANK_HEIGHT = 125

# tanítókáryták dimenziói
SUIT_WIDTH = 70
SUIT_HEIGHT = 100

RANK_DIFF_MAX = 2000
SUIT_DIFF_MAX = 700

CARD_MAX_AREA = 120000
CARD_MIN_AREA = 25000

font = cv2.FONT_HERSHEY_SIMPLEX

#A lekérdezési kártya és a tanító kártya adatainak tárolására szolgáló struktúrák

class Query_card:
    """Structure to store information about query cards in the camera image."""

    def __init__(self):
        self.contour = [] # kártya kontúr
        self.width, self.height = 0, 0 #szélesség magasság
        self.corner_pts = [] # sarok pont a kárytának
        self.center = [] # kártya közepe
        self.warp = [] # 200x300, flattened, grayed, blurred 
        self.rank_img = [] # Threshold kép méret
        self.suit_img = [] # Threshold kép méret
        self.best_rank_match = "Unknown" # legjobb rank
        self.best_suit_match = "Unknown" # legjobb suit
        self.rank_diff = 0 # A rangsorolt kép és a legjobban illeszkedő tanító rangsorolt képe közötti különbség
        self.suit_diff = 0 

class Train_ranks:
    """Structure to store information about train rank images."""

    def __init__(self):
        self.img = [] # Threshold, sized rank
        self.name = "Placeholder"

class Train_suits:
    """Structure to store information about train suit images."""

    def __init__(self):
        self.img = [] # merevlemezről - elöző komment
        self.name = "Placeholder"

### Függvények ###
def load_ranks(filepath):
    """Loads rank images from directory specified by filepath. Stores
    them in a list of Train_ranks objects."""

    train_ranks = []
    i = 0
    
    for Rank in ['Ace','Two','Three','Four','Five','Six','Seven',
                 'Eight','Nine','Ten','Jack','Queen','King']:
        train_ranks.append(Train_ranks())
        train_ranks[i].name = Rank
        filename = Rank + '.jpg'
        train_ranks[i].img = cv2.imread(filepath+filename, cv2.IMREAD_GRAYSCALE)
        i = i + 1

    return train_ranks

def load_suits(filepath):
    """Loads suit images from directory specified by filepath. Stores
    them in a list of Train_suits objects."""

    train_suits = []
    i = 0
    
    for Suit in ['Spades','Diamonds','Clubs','Hearts']:
        train_suits.append(Train_suits())
        train_suits[i].name = Suit
        filename = Suit + '.jpg'
        train_suits[i].img = cv2.imread(filepath+filename, cv2.IMREAD_GRAYSCALE)
        i = i + 1

    return train_suits

def preprocess_image(image):
    """Returns a grayed, blurred, and adaptively thresholded camera image."""

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)

    #A legjobb küszöbérték a környezeti fényviszonyoktól függ. Erős megvilágítás esetén magas küszöbértéket kell használni,
    #hogy a kártyákat el lehessen különíteni a háttértől. 
    #A kártyaérzékelőnek a fényviszonyoktól való függetlenné tétele érdekében a következő adaptív küszöbérték-módszert alkalmazzuk.
   
    img_w, img_h = np.shape(image)[:2]
    bkg_level = gray[int(img_h/100)][int(img_w/2)]
    thresh_level = bkg_level + BKG_THRESH

    retval, thresh = cv2.threshold(blur,thresh_level,255,cv2.THRESH_BINARY)
    
    return thresh

def find_cards(thresh_image):
    """Finds all card-sized contours in a thresholded camera image.
    Returns the number of cards, and a list of card contours sorted
    from largest to smallest."""

    # kontúr kereseés és válogatás index szerint
    cnts,hier = cv2.findContours(thresh_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    index_sort = sorted(range(len(cnts)), key=lambda i : cv2.contourArea(cnts[i]),reverse=True)

    # nincs kontúr nem történik semmi
    if len(cnts) == 0:
        return [], []
    
    # Ellenkező esetben üres, rendezett kontúr- és hierarchialisták inicializálása.
    cnts_sort = []
    hier_sort = []
    cnt_is_card = np.zeros(len(cnts),dtype=int)

    #Üres listák feltöltése rendezett kontúrral és rendezett hierarchiával. Most a kontúrlista indexei még mindig 
    #megfelelnek a hierarchialista indexeinek. A hierarchia tömb segítségével ellenőrizhetjük, hogy a kontúroknak vannak-e szülei vagy sem.
    
    for i in index_sort:
        cnts_sort.append(cnts[i])
        hier_sort.append(hier[0][i])

#Határozza meg, hogy a kontúrok közül melyek a kártyák, a következő kritériumok alkalmazásával: 1) Kisebb terület, mint a maximális kártyaméret,
#2) nagyobb terület, mint a minimális kártyaméret, 
#3) nincsenek szüleik,
#4) négy sarkuk van
        
        
    for i in range(len(cnts_sort)):
        size = cv2.contourArea(cnts_sort[i])
        peri = cv2.arcLength(cnts_sort[i],True)
        approx = cv2.approxPolyDP(cnts_sort[i],0.01*peri,True)
        
        if ((size < CARD_MAX_AREA) and (size > CARD_MIN_AREA)
            and (hier_sort[i][3] == -1) and (len(approx) == 4)):
            cnt_is_card[i] = 1

    return cnts_sort, cnt_is_card

def preprocess_card(contour, image):
    """Uses contour to find information about the query card. Isolates rank
    and suit images from the card."""

    # úk Query_card inicialiuzáció
    qCard = Query_card()

    qCard.contour = contour

    # Keresse meg a kártya kerületét, és használja azt a sarokpontok közelítésére.
    peri = cv2.arcLength(contour,True)
    approx = cv2.approxPolyDP(contour,0.01*peri,True)
    pts = np.float32(approx)
    qCard.corner_pts = pts

    # A kártya határoló téglalapjának szélességének és magasságának megkeresése
    x,y,w,h = cv2.boundingRect(contour)
    qCard.width, qCard.height = w, h

    # Keressük meg a kártya középpontját a négy sarok x és y átlagának figyelembevételével.
    average = np.sum(pts, axis=0)/len(pts)
    cent_x = int(average[0][0])
    cent_y = int(average[0][1])
    qCard.center = [cent_x, cent_y]

    #A kártyát a perspektivikus transzformáció segítségével alakítsuk 200x300-as lapított képpé.
    qCard.warp = flattener(image, pts, w, h)

    # Fogjuk meg a torzított kártya képének sarkát, és végezzünk 4x-es nagyítást.
    Qcorner = qCard.warp[0:CORNER_HEIGHT, 0:CORNER_WIDTH]
    Qcorner_zoom = cv2.resize(Qcorner, (0,0), fx=4, fy=4)

    # Vegyen mintát az ismert fehér képpontok intenzitásából a jó küszöbérték meghatározásához.
    white_level = Qcorner_zoom[15,int((CORNER_WIDTH*4)/2)]
    thresh_level = white_level - CARD_THRESH
    if (thresh_level <= 0):
        thresh_level = 1
    retval, query_thresh = cv2.threshold(Qcorner_zoom, thresh_level, 255, cv2. THRESH_BINARY_INV)
    
    # Osszuk fel a felső és alsó felére (a felső a rangot, az alsó a színt mutatja).
    Qrank = query_thresh[20:185, 0:128]
    Qsuit = query_thresh[186:336, 0:128]

    # Keressük meg a rangkontúr és a határoló téglalapot, különítsük el és keressük meg a legnagyobb kontúrt.
    Qrank_cnts, hier = cv2.findContours(Qrank, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    Qrank_cnts = sorted(Qrank_cnts, key=cv2.contourArea,reverse=True)

    # Keressük meg a legnagyobb kontúr határoló téglalapját, és használjuk a lekérdezési rangsor átméretezéséhez.
    #képet a vonatrangú kép méreteinek megfelelően.
    if len(Qrank_cnts) != 0:
        x1,y1,w1,h1 = cv2.boundingRect(Qrank_cnts[0])
        Qrank_roi = Qrank[y1:y1+h1, x1:x1+w1]
        Qrank_sized = cv2.resize(Qrank_roi, (RANK_WIDTH,RANK_HEIGHT), 0, 0)
        qCard.rank_img = Qrank_sized

    # Az illesztési kontúr és a határoló téglalap megkeresése, a legnagyobb kontúr elkülönítése és megtalálása.
    Qsuit_cnts, hier = cv2.findContours(Qsuit, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    Qsuit_cnts = sorted(Qsuit_cnts, key=cv2.contourArea,reverse=True)
    
#Keresse meg a legnagyobb kontúr határoló téglalapját, és használja fel a lekérdezési minta képének átméretezéséhez,
#hogy az megfeleljen a tanulási minta képének méreteinek.
    if len(Qsuit_cnts) != 0:
        x2,y2,w2,h2 = cv2.boundingRect(Qsuit_cnts[0])
        Qsuit_roi = Qsuit[y2:y2+h2, x2:x2+w2]
        Qsuit_sized = cv2.resize(Qsuit_roi, (SUIT_WIDTH, SUIT_HEIGHT), 0, 0)
        qCard.suit_img = Qsuit_sized

    return qCard

def match_card(qCard, train_ranks, train_suits):
    """Finds best rank and suit matches for the query card. Differences
    the query card rank and suit images with the train rank and suit images.
    The best match is the rank or suit image that has the least difference."""

    best_rank_match_diff = 5000000
    best_suit_match_diff = 5000000
    best_rank_match_name = "Unknown"
    best_suit_match_name = "Unknown"
    i = 0

    # Ha a preprocess_card függvényben nem találtunk kontúrokat a lekérdezési kártyán, akkor az img mérete nulla, tehát kihagyjuk. 
    if (len(qCard.rank_img) != 0) and (len(qCard.suit_img) != 0):
        
        #Különbözzük el a lekérdezési kártya rangképét az egyes tanítás rangképektől, és tároljuk a legkisebb különbséggel rendelkező eredményt.
        for Trank in train_ranks:

                diff_img = cv2.absdiff(qCard.rank_img, Trank.img)
                rank_diff = int(np.sum(diff_img)/255)
                
                if rank_diff < best_rank_match_diff:
                    best_rank_diff_img = diff_img
                    best_rank_match_diff = rank_diff
                    best_rank_name = Trank.name

        #Ugyanez a folyamat a suit képekkel.

        for Tsuit in train_suits:
                
                diff_img = cv2.absdiff(qCard.suit_img, Tsuit.img)
                suit_diff = int(np.sum(diff_img)/255)
                
                if suit_diff < best_suit_match_diff:
                    best_suit_diff_img = diff_img
                    best_suit_match_diff = suit_diff
                    best_suit_name = Tsuit.name

    #Kombinálja a legjobb rangsor és a legjobb szín egyezést, hogy megkapja a lekérdezési kártya azonosságát.  
    #Ha a legjobb egyezéseknek túl nagy a különbség értéke, a kártya azonossága továbbra is ismeretlen.

    if (best_rank_match_diff < RANK_DIFF_MAX):
        best_rank_match_name = best_rank_name

    if (best_suit_match_diff < SUIT_DIFF_MAX):
        best_suit_match_name = best_suit_name

    # Visszaadja a kártya azonosítóját, valamint a szín és a rang egyezés minőségét
    return best_rank_match_name, best_suit_match_name, best_rank_match_diff, best_suit_match_diff
    
    
def draw_results(image, qCard):
    """Draw the card name, center point, and contour on the camera image."""

    x = qCard.center[0]
    y = qCard.center[1]
    cv2.circle(image,(x,y),5,(255,0,0),-1)

    rank_name = qCard.best_rank_match
    suit_name = qCard.best_suit_match

    # Kétszer rajzolja meg a kártya nevét, hogy a betűk fekete körvonallal rendelkezzenek.
    cv2.putText(image,(rank_name+' of'),(x-60,y-10),font,1,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(image,(rank_name+' of'),(x-60,y-10),font,1,(50,200,200),2,cv2.LINE_AA)

    cv2.putText(image,suit_name,(x-60,y+25),font,1,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(image,suit_name,(x-60,y+25),font,1,(50,200,200),2,cv2.LINE_AA)
    
    return image

def flattener(image, pts, w, h):
    """Flattens an image of a card into a top-down 200x300 perspective.
    Returns the flattened, re-sized, grayed image.
    See www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/"""
    temp_rect = np.zeros((4,2), dtype = "float32")
    
    s = np.sum(pts, axis = 2)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]

    diff = np.diff(pts, axis = -1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    # A perspektivikus transzformáció előtt létre kell hoznunk egy tömböt, amely a [bal felső, jobb felső, jobb alsó, bal alsó] sorrendben listázza a pontokat.

    if w <= 0.8*h: #  Ha a kártya függőlegesen van tájolva.

        temp_rect[0] = tl
        temp_rect[1] = tr
        temp_rect[2] = br
        temp_rect[3] = bl

    if w >= 1.2*h: # Ha a kártya vízszintesen tájolt.
        temp_rect[0] = bl
        temp_rect[1] = tl
        temp_rect[2] = tr
        temp_rect[3] = br

    # Ha a kártya "rombusz" tájolású, más algoritmust kell használni annak azonosítására, hogy melyik pont a bal felső, jobb felső, bal alsó és jobb alsó pont.
    
    if w > 0.8*h and w < 1.2*h: #Ha a kártya gyémánt irányultságú
        # Ha a kártya balra van címezve, az approxPolyDP a következő sorrendben adja vissza a pontokat: jobb felső, bal felső, bal alsó, bal alsó, jobb alsó.
        if pts[1][0][1] <= pts[3][0][1]:
            #  Ha a bal szélső pont alacsonyabb, mint a jobb szélső pont, a kártya jobbra dől.
            temp_rect[0] = pts[1][0] # bal felső
            temp_rect[1] = pts[0][0] # jobb felső
            temp_rect[2] = pts[3][0] # jobb alsó
            temp_rect[3] = pts[2][0] # bal felső

        # Ha a kártya jobbra van címezve, az approxPolyDP a következő sorrendben adja vissza a pontokat: balra fent, balra lent, jobbra lent, jobbra fent.
        if pts[1][0][1] > pts[3][0][1]:
          
            temp_rect[0] = pts[0][0] # bal felső
            temp_rect[1] = pts[3][0] # jobb felső
            temp_rect[2] = pts[2][0] # jobb alsó
            temp_rect[3] = pts[1][0] # bal felső
            
        
    maxWidth = 200
    maxHeight = 300

    #  Céltömb létrehozása, perspektivikus transzformációs mátrix kiszámítása és a kártya képének elferdítése.
    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0, maxHeight-1]], np.float32)
    M = cv2.getPerspectiveTransform(temp_rect,dst)
    warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    warp = cv2.cvtColor(warp,cv2.COLOR_BGR2GRAY)

        

    return warp
