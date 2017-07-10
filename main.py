
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import PIL.Image as Image
import matplotlib
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import cdist
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D
from gmpy import bincoef

np.set_printoptions(threshold=np.nan)

def wordspotting():
    # TODO: Segmentierung aus gpt laden
    # dataNames: alle Namen der Dateien ohne Endung, kann also fuer GT & pages genutzt werden
    dataNames = [ str(name)+"0"+str(name) for name in range(270,280)]+[str(name)+"0"+str(name) for name in range(300,310)]
    #docs{} ist ein Dictionary, das fuer jede Datei eine Liste mit Listen mit den Grenzen der einzelnen Segmente und den Texten in diesen
    #Segmenten enthaelt
    docs = {}
    for i in range(len(dataNames)):
        obj = open("resources/GT/"+dataNames[i]+".gtp", "r")
        #TODO: alle Segmente aller Objekte speichern
        segs = []   #Liste mit Segementgrenzen und -texten, die in docs{} geschrieben wird
        for line in obj:
            xmin, ymin, xmax, ymax, text = line.split()
            segs.append(list((int(xmin), int(xmax), int(ymin), int(ymax), text)))
        docs[dataNames[i]] = segs
    print docs

    # TODO: SIFT fuer ganzes Bild
    # TODO: Vlfeat alle Deskriptoren fuer alle Bilder berechnen lassen
    step_size = 25
    cell_size = 5
    docframes = {}  #hier werden Frames fuer jedes Dokuemnt hineingeschrieben
    docdescs = {}   #SIFTs fuer jedes Dokument
    #frames, desc = vlfeat.vl_dsift(im_arr, step=step_size, size=cell_size)
    pickle_densesift_fn = 'resources/Sift/2700270-full_dense-%d_sift-%d_descriptors.p' % (step_size, cell_size)
    frames, desc = pickle.load(open(pickle_densesift_fn, 'rb'))
    frames = frames.T
    desc = np.array(desc.T, dtype=np.float)
    for name in dataNames:
        docframes[name] = []
        docdescs[name] = []
    docframes['2700270'] = frames
    docdescs['2700270'] = desc

    """
    # TODO: Visual Vocab mit Lloyd-Algorithmus
    n_centroids = 40
    _,labels = kmeans2(desc,n_centroids,iter =20, minit='points')

    #visualisierung
    document_image_filename = 'resources/pages/'+dataNames[0]+'.png'
    image = Image.open(document_image_filename)
    im_arr = np.asarray(image, dtype='float32')
    draw_descriptor_cells = True
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(im_arr, cmap=cm.get_cmap('Greys_r'))
    ax.hold(True)
    ax.autoscale(enable=False)
    colormap = cm.get_cmap('jet')
    desc_len = cell_size * 4
    for (x, y), label in zip(frames, labels):
        color = colormap(label / float(n_centroids))
        circle = Circle((x, y), radius=1, fc=color, ec=color, alpha=1)
        rect = Rectangle((x - desc_len / 2, y - desc_len / 2), desc_len, desc_len, alpha=0.08, lw=1)
        ax.add_patch(circle)
        if draw_descriptor_cells:
            for p_factor in [0.25, 0.5, 0.75]:
                offset_dyn = desc_len * (0.5 - p_factor)
                offset_stat = desc_len * 0.5
                line_h = Line2D((x - offset_stat, x + offset_stat), (y - offset_dyn, y - offset_dyn), alpha=0.08, lw=1)
                line_v = Line2D((x - offset_dyn , x - offset_dyn), (y - offset_stat, y + offset_stat), alpha=0.08, lw=1)
                ax.add_line(line_h)
                ax.add_line(line_v)
        ax.add_patch(rect)
    
    plt.show()
    """
    # TODO: Deskriptoren fuer Segment filtern (nach Deskriptor Ecke und Koordinaten der Sift-Operatoren)
    
    

    docssifts = {} #hier sollen analog zu docs die zu jedem Segment gehoerenden SIFT-Deskriptoren geschrieben werden
    for doc in docs:    #jedes Dokument durchgehen
        docf = docframes[doc]   #Frames im aktuellen Dokument
        ds = [] #SIFT-Deskriptoren, die zu aktuellem Dokuent gehoeren
        for seg in docs[doc]:    #Segmentgrenzen in aktuellem Dokument durchgehen
            framesifts = [] #SIFT-Deskriptoren, die zu aktuellem Segement gehoeren
            for i in range(len(docf)):  #berechnete SIFT-Deskriporen des aktuellem Dokuments durchgehen
                #Wenn Deskriptor im aktuellen Segment liegt, Deskriptor abspeichern
                if seg[0] <= docf[i][0] and seg[1] >= docf[i][0] and seg[2] <= docf[i][1] and seg[3] >= docf[i][1]:
                    framesifts.append(docdescs[doc][i])
            ds.append(framesifts)   #zu aktuellem Segment gehoerende Deskriptoren zu Liste mit Deskriptoren im Dokument hinzufuegen
        docssifts[doc] = ds #fertige Liste mit Deskriptoren im Dokument ins Dictionary schreiben
    #es fehlen noch die uebrigen dokumente bis jetzt geht nur einss    
    #print docssifts
    
    cluster = {}    #Dictionary mit Zuordnungen der Deskriptoren zu Centroids
    n_centroids = 3 #Anzahl Centroids
    doc = '2700270' #erstmal nur das eine Dokument
    cluster[doc] = []
    for seg in docssifts[doc]:  #fuer jedes Segment im Dokument Zuordnung berechnen
        _, labels = kmeans2(seg, n_centroids, iter=20, minit='points')
        cluster[doc].append(labels) #Zuordnung abspeichern
    print cluster
    
    bof = {}    #Dictionary mit Bag-of-Features-Repraesentationen fuer ganze Segmente
    doc = '2700270'
    bof[doc] = []
    for seg in cluster[doc]:    #Histogramm, d.h. BoF-Repraesentation, fuer jedes Segment berechnen
        hist = np.bincount(seg)
        bof[doc].append(hist) #Histogramm abspeichern
    print bof
            
    # TODO: Spatial Pyramid fuer jedes Segment & Bag-of-Features
    # Spatial Pyramid: SIFT in ganzem, linken, rechten Segment zaehlen (Histogramm)
    # Bag-of-Features: Vektor mit 3*n Werten
    # TODO: Singulaerwertzerlegung der Bag of Features
    # TODO: Distanz des Inputs durch Cosinusdistanz
    # TODO: Fehlerevaluierung
    
if __name__ == '__main__':
    wordspotting()

