def wordspotting():
    # TODO: Segmentierung aus gpt Laden
    dataNames = [ str(name)+"0"+str(name) for name in range(270,280)]+[str(name)+"0"+str(name) for name in range(300,310)]
    for i in range(len(dataNames)):
        obj = open("ressources/GT/"+dataNames[i]+".gtp", "r")
        print obj
        for line in obj:
            print line
    # TODO: SIFT fuer ganzes Bild
    # TODO: Visual Vocab mit Lloyd- Algorithmus
    # TODO: Deskriptoren fuer Segment filtern (nach Deskriptor Ecke und Koodirnaten der Sift- Operatoren)
    # TODO: Spatial Pyramid fuer jedes Segment & Bag-of-Features
    # Spatial Pyramid: SIFT in ganzem, linken, rechten Segment zaehlen (Histogramm)
    # Bag-of-Features: Vektor mit 3*n Werten
    # TODO: Singulaerwertzerlegung der Bag of Features
    # TODO: Distanz des Inputs durch Cosinusdistanz
    # TODO: Fehlerevaluierung
wordspotting()