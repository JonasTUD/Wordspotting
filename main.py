import cPickle as pickle
import numpy as np

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
    step_size = 65
    cell_size = 15
    #frames, desc = vlfeat.vl_dsift(im_arr, step=step_size, size=cell_size)
    pickle_densesift_fn = 'resources/Sift/2700270-full_dense-%d_sift-%d_descriptors.p' % (step_size, cell_size)
    frames, desc = pickle.load(open(pickle_densesift_fn, 'rb'))
    frames = frames.T
    desc = np.array(desc.T, dtype=np.float)

    # TODO: Visual Vocab mit Lloyd-Algorithmus
    # TODO: Deskriptoren fuer Segment filtern (nach Deskriptor Ecke und Koordinaten der Sift-Operatoren)
    # TODO: Spatial Pyramid fuer jedes Segment & Bag-of-Features
    # Spatial Pyramid: SIFT in ganzem, linken, rechten Segment zaehlen (Histogramm)
    # Bag-of-Features: Vektor mit 3*n Werten
    # TODO: Singulaerwertzerlegung der Bag of Features
    # TODO: Distanz des Inputs durch Cosinusdistanz
    # TODO: Fehlerevaluierung
    
if __name__ == '__main__':
    wordspotting()
