import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import PIL.Image as Image
import matplotlib
#import vlfeat
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import cdist, pdist, squareform
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D
from numpy import bincount, argsort
import scipy.sparse

np.set_printoptions(threshold=np.nan)


def wordspotting():

    # dataNames: alle Namen der Dateien ohne Endung, kann also fuer GT & pages genutzt werden
    dataNames = [ str(name)+"0"+str(name) for name in range(270,280)]+[str(name)+"0"+str(name) for name in range(300,310)]
    
    doc = []    #hier stehen jetzt fuer jedes Segment die Informationen in der Form (xmin, xmax, ymin, ymax, text)
    obj = open("resources/GT/2700270.gtp", "r")
    segs = []   #Liste mit Segementgrenzen und -texten
    for line in obj:
        xmin, ymin, xmax, ymax, text = line.split()
        doc.append(list((int(xmin), int(xmax), int(ymin), int(ymax), text)))
    step_size =25
    cell_size =5
    pickle_densesift_fn = 'resources/Sift/2700270-full_dense-%d_sift-%d_descriptors.p' % (step_size, cell_size)
    frames, desc = pickle.load(open(pickle_densesift_fn, 'rb'))
    frames = frames.T
    desc = np.array(desc.T, dtype=np.float)
    # Optional: SIFT nach Vorkommen in Segmenten filtern
    n_centroids = 2048
    _,labels = kmeans2(desc,n_centroids,iter =40, minit='points')

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

    siftsind = []
    siftslinksind = []
    siftsrechtsind = []
    for seg in doc:    #Segmentgrenzen in Dokument durchgehen
        framesifts = [] #Indizes der SIFT-Deskriptoren, die zu aktuellem Segement gehoeren
        framesiftslinks = []
        framesiftsrechts = []
        for i in range(len(frames)):  #Zentren der berechneten SIFT-Deskriporen durchgehen
            #Wenn Deskriptor im aktuellen Segment liegt, Index des Deskriptors abspeichern
            if seg[0] <= frames[i][0] and seg[1] >= frames[i][0] and seg[2] <= frames[i][1] and seg[3] >= frames[i][1]:
                framesifts.append(i)
            #fuer linken Teil
            if seg[0] <= frames[i][0] and (seg[0]+((seg[1]-seg[0])/2)) >= frames[i][0] and seg[2] <= frames[i][1] and seg[3] >= frames[i][1]:
                framesiftslinks.append(i)
            #fuer rechten Teil   
            if (seg[0]+((seg[1]-seg[0])/2)) < frames[i][0] and seg[1] >= frames[i][0] and seg[2] <= frames[i][1] and seg[3] >= frames[i][1]:
                framesiftsrechts.append(i)
        siftsind.append(framesifts)   #zu aktuellem Segment gehoerende Deskriptoren zu Liste mit Deskriptoren im Dokument hinzufuegen
        siftslinksind.append(framesiftslinks) 
        siftsrechtsind.append(framesiftsrechts) 
    print siftsind
    print siftslinksind
    print siftsrechtsind
    #in sifts[] stehen jetzt an i-ter Stelle die Indizes der Deskriptoren, die zum ganzen i-ten Segement im Dokument gehoeren
    
    #analog stehen in siftslinks und siftsrechts die bei der Berechnung der Spatial Pyramid notwendigen Indizes der
    #Deskriptoren im linken und rechten Segmentausschnitt
    
    

    # TODO: Spatial Pyramid fuer jedes Segment & Bag-of-Features
    # uebernimmt: blub
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
    #print bof
    #print bof[21]
    #print bof[22]
    #print bof.shape
    #print type(bof[0,0])
    
    dist = pdist(bof, 'euclidean')
    print dist.shape
    dist = squareform(dist)
    dist = argsort(dist)
    print dist
    
    wordcount=[]
    for i in range(len(doc)):
        counter=0
        for j in range(len(doc)):
            if doc[i][4] == doc[j][4]:
                counter = counter + 1
        wordcount.append(counter)
    print wordcount
    #in wordcount[i] steht, wie oft der Text des i-ten Segments insgesamt im Dokument vorkommt (erleichtert die Evaluation)
    
    for word in range(len(doc)):
        if wordcount[word] != 1:
            a = wordcount[word]
            count = 0
            similarWords = ""
            print 'Die Woerter der' , wordcount[word]-1, 'Segmente, die als dem Segment mit dem Wort "', doc[word][4], '" am aehnlichsten erkannt wurden: '
            for i in range(1, wordcount[word]):
                similarWords += str(doc[dist[word][i]][4]) + ", "
                if doc[dist[word][i]][4] == doc[word][4]:
                    count = count+1

            error = (float(count)/a)*100
	    print similarWords
            print 'Das ergibt eine Erkennungsrate von', error, '%'
            print
            if word % 100 == 0:
		document_image_filename = 'resources/pages/2700270.png'
    		image = Image.open(document_image_filename)
    		im_arr = np.asarray(image, dtype='float32')
		#print im_arr
		queryimg_arr = im_arr[doc[word][2]:doc[word][3],doc[word][0]:doc[word][1]]
    		for i in range(1, wordcount[word]):
			similarwordimg_arr = im_arr[doc[dist[word][i]][2]:doc[dist[word][i]][3],doc[dist[word][i]][0]:doc[dist[word][i]][1]]
			a = np.shape(queryimg_arr)[0]
			b = np.shape(similarwordimg_arr)[0]
			if a < b:
				queryimg_arr = np.vstack((queryimg_arr,np.full((b-a,np.shape(queryimg_arr)[1]),220)))
			if a > b:
				print np.shape(similarwordimg_arr)
				print np.shape(np.full((a-b,np.shape(similarwordimg_arr)[1]),220))
				similarwordimg_arr = np.vstack((similarwordimg_arr,np.full((a-b,np.shape(similarwordimg_arr)[1]),220)))
			print np.shape(queryimg_arr)
			print np.shape(similarwordimg_arr)
			queryimg_arr = np.hstack((queryimg_arr,np.zeros(((np.shape(queryimg_arr)[0]),10))))	
			queryimg_arr = np.hstack((queryimg_arr,similarwordimg_arr))
		print queryimg_arr[0]
   		fig = plt.figure()
  	  	ax = fig.add_subplot(111)
    		ax.imshow(queryimg_arr, cmap=cm.get_cmap('Greys_r'))
    		#ax.hold(True)
    		ax.autoscale(enable=True)
		plt.show()
		print doc[word][4]
            
            
    """""
    
    
    
    
    """""
    
    #Das codebook scheint die falschen deskriptoren zu enthalten
    #die deskriptoren aus sifts passen nicht zu denen im codebook
    """
    npsiftslinks = np.asarray(siftslinks)
    npsiftsrechts = np.asarray(siftsrechts)
    np.shape(np.bincount(npsiftslinks,minlength=n_centroids))
    np.shape(np.bincount(npsiftsrechts,minlength=n_centroids))
    """
    # Rueckgabe: Matrix: Anzahl Segmente X (4096*3)
    # Bitte Rueckgabe bof nennen!

    #bof = [[]]

    # TODO: Distanz des Inputs durch Cosinusdistanz
    # pdist, argsort,
    # uebernimmt: blub

    # Frage: Warum pdist? So wuerden wir ja jedes Vorkommen von einem Centroiden mit dem im anderen Segment vergleichen und nur die am naechsten zueinander stehenden Centroiden finden
    # Glaube daher cdist ist richtig

    #bofDist = cdist(bof,bof, 'cosine')
    #bofDistArgsort = (np.argsort(bofDist))[:,1:]

    # boolsche matrix/schleifen um Vorkommen zu identifizieren
    # uebernimmt: Jonas

    
    

    # TODO: Fehlerevaluierung
    # uebernimmt: recharge

if __name__ == '__main__':
    wordspotting()
