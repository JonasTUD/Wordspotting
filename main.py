import cPickle as pickle
import numpy as np
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
            segs.append(list((xmin, xmax, ymin, ymax, text)))
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
    

    # TODO: Visual Vocab mit Lloyd-Algorithmus
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
                print seg[0], seg[1], seg[2], seg[3]
                print docf[i][0], docf[i][1]
            ds.append(framesifts)   #zu aktuellem Segment gehoerende Deskriptoren zu Liste mit Deskriptoren im Dokument hinzufuegen
        docssifts[doc] = ds #fertige Liste mit Deskriptoren im Dokument ins Dictionary schreiben
        
    #Leider bleibt dieses Dictionary leer. Stimmen die Koordinaten der Segmentgrenzen und die der Deskriptoren nicht ueberein?
    print docssifts
            
                        
            
    # TODO: Spatial Pyramid fuer jedes Segment & Bag-of-Features
    # Spatial Pyramid: SIFT in ganzem, linken, rechten Segment zaehlen (Histogramm)
    # Bag-of-Features: Vektor mit 3*n Werten
    # TODO: Singulaerwertzerlegung der Bag of Features
    # TODO: Distanz des Inputs durch Cosinusdistanz
    # TODO: Fehlerevaluierung
    
if __name__ == '__main__':
    wordspotting()
