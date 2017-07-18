import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import PIL.Image as Image
import matplotlib
import vlfeat
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import cdist, pdist, squareform
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D
from numpy import bincount, argsort
import scipy.sparse

np.set_printoptions(threshold=np.nan)

def wordspottingAllDocs():
    import vlfeat
    #sift, scipy etc laufen bei mir. Habe den vorhandenen ablauf umgeschrieben, so dass er mit den dokumenten klar kommt.
    # dataNames: alle Namen der Dateien ohne Endung, kann also fuer GT & pages genutzt werden
    dataNames = [ str(name)+"0"+str(name) for name in range(270,280)]+[str(name)+"0"+str(name) for name in range(300,310)]

    segmentsInDoc = {}
    imageOfDoc = {}     #bilddaten fuer jedes dokument
    for i in range(len(dataNames))[:2]:
        obj = open("resources/GT/"+dataNames[i]+".gtp", "r")
        segs = []   #Liste mit Segementgrenzen und -texten, die in docs{} geschrieben wird
        for line in obj:
            xmin, ymin, xmax, ymax, text = line.split()
            segs.append(list((int(xmin), int(xmax), int(ymin), int(ymax), text)))
        segmentsInDoc[dataNames[i]] = segs

        image = Image.open("resources/pages/"+dataNames[i]+".png")
        im_arr = np.asarray(image, dtype='float32')
        imageOfDoc[dataNames[i]] = im_arr
    #print segmentsInDoc

    step_size = 45
    cell_size = 10
    framesInDoc = {}    #koordinaten der deskriporen fuer jedes dokument
    descInDoc = {}      #deskriptor vektor fuer jedes dokument
    for i in range(len(dataNames))[:2]:     #wenn es fuer die ersten 2 klappt geht es sicher auch fuer alle
        frames, desc = vlfeat.vl_dsift(imageOfDoc[dataNames[i]], step=step_size, size=cell_size)        #deskriptor koordinaten(frames) und deskriporen(128 dim vektor) berechnen
        frames = frames.T
        desc = np.array(desc.T, dtype=np.float)
        framesInDoc[dataNames[i]] = frames
        descInDoc[dataNames[i]] = desc
        print str(dataNames[i]) + " is done " + str(desc.shape)

    #print framesInDoc
    #print descInDoc
    descForAllDocs = np.vstack(descInDoc.values())
    segsForAllDocs = np.vstack(segmentsInDoc.values())     #sammlung aller segmente und woerter in den dokumenten
    n_centroids = 20
    _,labels = kmeans2(descForAllDocs,n_centroids,iter =40, minit='points') #labels fuer alle deskriptoren in allen dokumenten berechnen
    print "Labels for all desc done " + str(labels.shape)

    #da nur deskriptoren in den segmenten interessant sind werden diese jetzt berechnet
    siftsind = []
    siftslinksind = []
    siftsrechtsind = []
    for name in dataNames[:2]:
        for seg in segmentsInDoc[name]:    #Segmentgrenzen in Dokument durchgehen
            framesifts = [] #Indizes der SIFT-Deskriptoren, die zu aktuellem Segement gehoeren
            framesiftslinks = []
            framesiftsrechts = []
            for i in range(len(framesInDoc[name])):  #Zentren der berechneten SIFT-Deskriporen durchgehen
                #Wenn Deskriptor im aktuellen Segment liegt, Index des Deskriptors abspeichern
                if seg[0] <= framesInDoc[name][i][0] and seg[1] >= framesInDoc[name][i][0] and seg[2] <= framesInDoc[name][i][1] and seg[3] >= framesInDoc[name][i][1]:
                    framesifts.append(i)
                #fuer linken Teil
                if seg[0] <= framesInDoc[name][i][0] and (seg[0]+((seg[1]-seg[0])/2)) >= framesInDoc[name][i][0] and seg[2] <= framesInDoc[name][i][1] and seg[3] >= framesInDoc[name][i][1]:
                    framesiftslinks.append(i)
                #fuer rechten Teil
                if (seg[0]+((seg[1]-seg[0])/2)) < framesInDoc[name][i][0] and seg[1] >= framesInDoc[name][i][0] and seg[2] <= framesInDoc[name][i][1] and seg[3] >= framesInDoc[name][i][1]:
                    framesiftsrechts.append(i)
            siftsind.append(framesifts)   #zu aktuellem Segment gehoerende Deskriptoren zu Liste mit Deskriptoren im Dokument hinzufuegen
            siftslinksind.append(framesiftslinks)
            siftsrechtsind.append(framesiftsrechts)

    print np.array(siftsind).shape  #enthaelt die deskriptoren, die in den segmenten von allen dokumenten liegen

    hist1 = []           #Histogramm fuer gesamte Segmente berechnen
    for seg in siftsind:
        segarr =[]
        for s in seg:
            segarr.append(int(labels[s]))
        hist1.append(np.bincount(np.array(segarr, dtype='int'), minlength = n_centroids))

    hist2 = []           #Histogramm fuer linken Segmentteil berechnen
    for seg in siftslinksind:
        segarr =[]
        for s in seg:
            segarr.append(int(labels[s]))
        hist2.append(np.bincount(np.array(segarr, dtype='int'), minlength = n_centroids))

    hist3 = []           #Histogramm fuer rechten Segmentteil berechnen
    for seg in siftsrechtsind:
        segarr =[]
        for s in seg:
            segarr.append(int(labels[s]))
        hist3.append(np.bincount(np.array(segarr, dtype='int'), minlength = n_centroids))

    print len(hist1)


    bof = []
    for i in range(len(hist1)): #Histogramme zur BoF-Repraesentation zusammenfuehren
        bof.append(np.array(list(hist1[i]) + list(hist2[i]) + list(hist3[i])))

    bof = np.array(bof)
    dist = pdist(bof, 'cosine')
    print dist.shape
    dist = squareform(dist)
    dist = argsort(dist)
    print dist.shape


    wordcount=[]
    for name in dataNames[:2]:
        for i in range(len(segmentsInDoc[name])):
            counter=0
            for j in range(len(segmentsInDoc[name])):
                if segmentsInDoc[name][i][4] == segmentsInDoc[name][j][4]:
                    counter = counter + 1
            wordcount.append(counter)
    print np.array(wordcount)
    #in wordcount[i] steht, wie oft der Text des i-ten Segments insgesamt im Dokument vorkommt (erleichtert die Evaluation)

    for word in range(segsForAllDocs.shape[0]):
        if wordcount[word] != 1:
            a = wordcount[word]
            count = 0
            similarWords = ""
            print 'Die Woerter der' , wordcount[word]-1, 'Segmente, die als dem Segment mit dem Wort "', segsForAllDocs[word][4], '" am aehnlichsten erkannt wurden: '
            for i in range(1, wordcount[word]):
                similarWords += str(segsForAllDocs[dist[word][i]][4]) + ", "
                if segsForAllDocs[dist[word][i]][4] == segsForAllDocs[word][4]:
                    count = count+1
            error = (float(count)/a)*100
            print similarWords
            print 'Das ergibt eine Erkennungsrate von', error, '%'
            print

def wordspottingOneDoc():
    doc = []    #hier stehen jetzt fuer jedes Segment die Informationen in der Form (xmin, xmax, ymin, ymax, text)
    obj = open("resources/GT/2700270.gtp", "r")
    image = Image.open("resources/pages/2700270.png")
    segs = []   #Liste mit Segementgrenzen und -texten
    for line in obj:
        xmin, ymin, xmax, ymax, text = line.split()
        doc.append(list((int(xmin), int(xmax), int(ymin), int(ymax), text)))
    step_size = 5
    cell_size = 10
    #pickle_densesift_fn = 'resources/Sift/2700270-full_dense-%d_sift-%d_descriptors.p' % (step_size, cell_size)
    #frames, desc = pickle.load(open(pickle_densesift_fn, 'rb'))
    im_arr = np.asarray(image, dtype='float32')
    #frames, desc = vlfeat.vl_dsift(im_arr, step=step_size, size=cell_size)
    #frames = frames.T
    #desc = np.array(desc.T, dtype=np.float)
    #pickle.dump( frames, open( "frames.p", "wb" ) )
    #pickle.dump( desc, open( "desc.p", "wb" ) )
    frames = pickle.load( open( "frames.p", "rb" ) )
    desc = pickle.load( open( "desc.p", "rb" ) )
    # Optional: SIFT nach Vorkommen in Segmenten filtern

    #===========================================================================
    # framesInSeq = []
    # descInSeq = []
    # for seg in doc:
    #     for i in range(len(frames)):  #Zentren der berechneten SIFT-Deskriporen durchgehen
    #         #Wenn Deskriptor im aktuellen Segment liegt, Index des Deskriptors abspeichern
    #         if seg[0] <= frames[i][0] and seg[1] >= frames[i][0] and seg[2] <= frames[i][1] and seg[3] >= frames[i][1]:
    #             framesInSeq.append(frames[i])
    #             descInSeq.append(desc[i])
    # print 'Segmente entfernt'
    # pickle.dump( framesInSeq, open( "framesInSeq.p", "wb" ) )
    # pickle.dump( descInSeq, open( "descInSeq.p", "wb" ) )
    #===========================================================================
    framesInSeq = pickle.load( open( "framesInSeq.p", "rb" ) )
    descInSeq = pickle.load( open( "descInSeq.p", "rb" ) )

    frames = framesInSeq
    desc = np.array(descInSeq)
    n_centroids = 256
    _,labels = kmeans2(desc,n_centroids,iter =20, minit='points')
    pickle.dump( labels, open( "labels.p", "wb" ) )
    print 'kmeans berechnet'

    """""
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
    """""

    #===========================================================================
    # siftsind = []
    # siftslinksind = []
    # siftsrechtsind = []
    # for seg in doc:    #Segmentgrenzen in Dokument durchgehen
    #     framesifts = [] #Indizes der SIFT-Deskriptoren, die zu aktuellem Segement gehoeren
    #     framesiftslinks = []
    #     framesiftsrechts = []
    #     for i in range(len(frames)):  #Zentren der berechneten SIFT-Deskriporen durchgehen
    #         #Wenn Deskriptor im aktuellen Segment liegt, Index des Deskriptors abspeichern
    #         if seg[0] <= frames[i][0] and seg[1] >= frames[i][0] and seg[2] <= frames[i][1] and seg[3] >= frames[i][1]:
    #             framesifts.append(i)
    #         #fuer linken Teil
    #         if seg[0] <= frames[i][0] and (seg[0]+((seg[1]-seg[0])/2)) >= frames[i][0] and seg[2] <= frames[i][1] and seg[3] >= frames[i][1]:
    #             framesiftslinks.append(i)
    #         #fuer rechten Teil
    #         if (seg[0]+((seg[1]-seg[0])/2)) < frames[i][0] and seg[1] >= frames[i][0] and seg[2] <= frames[i][1] and seg[3] >= frames[i][1]:
    #             framesiftsrechts.append(i)
    #     siftsind.append(framesifts)   #zu aktuellem Segment gehoerende Deskriptoren zu Liste mit Deskriptoren im Dokument hinzufuegen
    #     siftslinksind.append(framesiftslinks)
    #     siftsrechtsind.append(framesiftsrechts)
    # #print siftslinksind
    # #print siftsrechtsind
    # #in sifts[] stehen jetzt an i-ter Stelle die Indizes der Deskriptoren, die zum ganzen i-ten Segement im Dokument gehoeren
    # print 'Deskriptoren pro Segment gefiltert'
    # pickle.dump( siftsind, open( "siftsind.p", "wb" ) )
    # pickle.dump( siftslinksind, open( "siftslinksind.p", "wb" ) )
    # pickle.dump( siftsrechtsind, open( "siftsrechtsind.p", "wb" ) )
    #===========================================================================
    siftsind = pickle.load( open( "siftsind.p", "rb" ) )
    siftslinksind = pickle.load( open( "siftslinksind.p", "rb" ) )
    siftsrechtsind = pickle.load( open( "siftsrechtsind.p", "rb" ) )

    #analog stehen in siftslinks und siftsrechts die bei der Berechnung der Spatial Pyramid notwendigen Indizes der
    #Deskriptoren im linken und rechten Segmentausschnitt



    # Histogramm fuer jedes Segment mit bincount und bins=n_centroid
    # Spatial Pyramid: SIFT in ganzem, linken, rechten Segment zaehlen (Histogramm)
    # Bag-of-Features: Vektor mit 3*n Werten

    hist1 = []           #Histogramm fuer gesamte Segmente berechnen
    for seg in siftsind:
        segarr =[]
        for s in seg:
            segarr.append(labels[s])
        hist1.append(np.bincount(np.array(segarr), minlength = n_centroids))

    hist2 = []           #Histogramm fuer linken Segmentteil berechnen
    for seg in siftslinksind:
        segarr =[]
        for s in seg:
            segarr.append(labels[s])
        hist2.append(np.bincount(np.array(segarr), minlength = n_centroids))

    hist3 = []           #Histogramm fuer rechten Segmentteil berechnen
    for seg in siftsrechtsind:
        segarr =[]
        for s in seg:
            segarr.append(labels[s])
        hist3.append(np.bincount(np.array(segarr), minlength = n_centroids))

    bof = []
    #print type(hist1[0])
    for i in range(len(hist1)): #Histogramme zur BoF-Repraesentation zusammenfuehren
        bof.append(np.array(list(hist1[i]) + list(hist2[i]) + list(hist3[i])))

    bof = np.array(bof)
    print 'Spatial Pyramid erstellt'

    dist = pdist(bof, 'cosine')
    #print dist.shape
    dist = squareform(dist)
    dist = argsort(dist)
    #print dist
    print 'Distanzen berechnet'

    wordcount=[]
    for i in range(len(doc)):
        counter=0
        for j in range(len(doc)):
            if doc[i][4] == doc[j][4]:
                counter = counter + 1
        wordcount.append(counter)
    #print wordcount
    #in wordcount[i] steht, wie oft der Text des i-ten Segments insgesamt im Dokument vorkommt (erleichtert die Evaluation)
    
    ap = []
    for i in range(len(wordcount) ):
        if wordcount[i] != 1:
            result = []
            for j in range( len(wordcount)):
                if doc[dist[i][j]][4] == doc[i][4]:
                    result.append(1)
                else:
                    result.append(0)
            p = []
            print 'result fuer Wort', doc[i][4], result
            for k in range(0, len(result)):
                p.append((np.sum(result[:k])/k)*result[k])
            pk = sum(p)
            print 'pk fuer Wort', doc[i][4], pk
            rel2 = np.sum(result)
            print 'rel2 fuer Wort', doc[i][4], rel2
            ap.append(float(pk)/rel2)
    print ap
    map = np.sum(ap)/len(ap)
    print 'map', map
        
#===============================================================================
#     sumerror =0
#     counterrror =0
#     for word in range(len(doc)):
#         if wordcount[word] != 1:
# 
#             a = wordcount[word]-1
# 
#             count = 0
#             similarWords = ""
#             print 'Die Woerter der' , wordcount[word]-1, 'Segmente, die als dem Segment mit dem Wort "', doc[word][4], '" am aehnlichsten erkannt wurden: '
#             for i in range(1, a):
#                 similarWords += str(doc[dist[word][i]][4]) + ", "
#                 if doc[dist[word][i]][4] == doc[word][4]:
#                     count = count+1
#             error = (float(count)/a)*100
#             sumerror = sumerror+ error
#             counterrror =counterrror+1
#             print similarWords
#             print 'Das ergibt eine Erkennungsrate von', error, '%'
#             print
#             if error >=50:
# 		document_image_filename = 'resources/pages/2700270.png'
#     		image = Image.open(document_image_filename)
#     		im_arr = np.asarray(image, dtype='float32')
# 		#print im_arr
# 		queryimg_arr = im_arr[doc[word][2]:doc[word][3],doc[word][0]:doc[word][1]]
#     		for i in range(1, wordcount[word]):
# 			similarwordimg_arr = im_arr[doc[dist[word][i]][2]:doc[dist[word][i]][3],doc[dist[word][i]][0]:doc[dist[word][i]][1]]
# 			a = np.shape(queryimg_arr)[0]
# 			b = np.shape(similarwordimg_arr)[0]
# 			if a < b:
# 				queryimg_arr = np.vstack((queryimg_arr,np.full((b-a,np.shape(queryimg_arr)[1]),220)))
# 			if a > b:
# 				print np.shape(similarwordimg_arr)
# 				print np.shape(np.full((a-b,np.shape(similarwordimg_arr)[1]),220))
# 				similarwordimg_arr = np.vstack((similarwordimg_arr,np.full((a-b,np.shape(similarwordimg_arr)[1]),220)))
# 			print np.shape(queryimg_arr)
# 			print np.shape(similarwordimg_arr)
# 			queryimg_arr = np.hstack((queryimg_arr,np.zeros(((np.shape(queryimg_arr)[0]),10))))	
# 			queryimg_arr = np.hstack((queryimg_arr,similarwordimg_arr))
# 		print queryimg_arr[0]
#    		fig = plt.figure()
#   	  	ax = fig.add_subplot(111)
#     		ax.imshow(queryimg_arr, cmap=cm.get_cmap('Greys_r'))
#     		#ax.hold(True)
#     		ax.autoscale(enable=True)
# 		plt.show()
#     print "mean average precision:"
#     print sumerror/counterrror
#===============================================================================


    """
    npsiftslinks = np.asarray(siftslinks)
    npsiftsrechts = np.asarray(siftsrechts)
    np.shape(np.bincount(npsiftslinks,minlength=n_centroids))
    np.shape(np.bincount(npsiftsrechts,minlength=n_centroids))
    """
    # Rueckgabe: Matrix: Anzahl Segmente X (4096*3)

if __name__ == '__main__':
    wordspottingOneDoc()
