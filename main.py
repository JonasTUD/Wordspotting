import cPickle as pickle
import numpy as np

def wordspotting():
    # TODO: Segmentierung aus gpt Laden
    dataNames = [ str(name)+"0"+str(name) for name in range(270,280)]+[str(name)+"0"+str(name) for name in range(300,310)]
    for i in range(len(dataNames)):
        obj = open("resources/GT/"+dataNames[i]+".gtp", "r")
        print obj
        for line in obj:
            print line

    # TODO: SIFT fuer ganzes Bild
    step_size = 65
    cell_size = 15
    #frames, desc = vlfeat.vl_dsift(im_arr, step=step_size, size=cell_size)
    pickle_densesift_fn = 'resources/Sift/2700270-full_dense-%d_sift-%d_descriptors.p' % (step_size, cell_size)
    frames, desc = pickle.load(open(pickle_densesift_fn, 'rb'))
    frames = frames.T
    desc = np.array(desc.T, dtype=np.float)

    # TODO: Visual Vocab mit Lloyd- Algorithmus
    # TODO: Deskriptoren fuer Segment filtern (nach Deskriptor Ecke und Koodirnaten der Sift- Operatoren)
    # TODO: Spatial Pyramid fuer jedes Segment & Bag-of-Features
    # Spatial Pyramid: SIFT in ganzem, linken, rechten Segment zaehlen (Histogramm)
    # Bag-of-Features: Vektor mit 3*n Werten
    # TODO: Singulaerwertzerlegung der Bag of Features
    # TODO: Distanz des Inputs durch Cosinusdistanz
    # TODO: Fehlerevaluierung
wordspotting()