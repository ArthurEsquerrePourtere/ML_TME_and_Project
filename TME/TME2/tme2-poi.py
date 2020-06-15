import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

plt.ion()
parismap = mpimg.imread('data/paris-48.806-2.23--48.916-2.48.jpg')

## coordonnees GPS de la carte
xmin,xmax = 2.23,2.48   ## coord_x min et max
ymin,ymax = 48.806,48.916 ## coord_y min et max

def show_map():
    plt.imshow(parismap,extent=[xmin,xmax,ymin,ymax],aspect=1.5)
    ## extent pour controler l'echelle du plan

poidata = pickle.load(open("data/poi-paris.pkl","rb"))
## liste des types de point of interest (poi)
print("Liste des types de POI" , ", ".join(poidata.keys()))

## Choix d'un poi
typepoi = "night_club"

## Creation de la matrice des coordonnees des POI
geo_mat = np.zeros((len(poidata[typepoi]),2))
for i,(k,v) in enumerate(poidata[typepoi].items()):
    geo_mat[i,:]=v[0]

## Affichage brut des poi
show_map()
## alpha permet de regler la transparence, s la taille
plt.scatter(geo_mat[:,1],geo_mat[:,0],alpha=0.8,s=3)


###################################################
############## HISTOGRAMME ##############

# discretisation pour l'affichage des modeles d'estimation de densite
steps = 20
xx,yy = np.meshgrid(np.linspace(xmin,xmax,steps),np.linspace(ymin,ymax,steps))
grid = np.c_[xx.ravel(),yy.ravel()]

def histo(data,steps):
    borneX =np.linspace(xmin,xmax,steps+1)
    borneY = np.linspace(ymin,ymax,steps+1)
    newMap = np.zeros((steps,steps))
    for i in data :
        newMap[determinePos(borneY,i[0]),determinePos(borneX,i[1])] += 1
    return newMap/len(data[:,0])
    


def determinePos(borne,value):
    for i in range(len(borne)-1):
        if value >= borne[i] and value < borne[i+1] :
            return i

#res = np.random.random((steps,steps))
res = histo(geo_mat,steps).reshape(steps,steps)
plt.figure()
show_map()
plt.imshow(res,extent=[xmin,xmax,ymin,ymax],interpolation='none',\
               alpha=0.3,origin = "lower")
plt.colorbar()
plt.scatter(geo_mat[:,1],geo_mat[:,0],alpha=0.3,s=4)


############## PARZEN ##############

def Parzen(hn,matrice,n):
    
    borneX=np.linspace(xmin,xmax,n)
    borneY = np.linspace(ymin,ymax,n)
    
    newMatrice = np.zeros((n,n))
    
    for i in range(len(borneX)) :
        for j in range(len(borneY)) :
            newMatrice[j,i] = densiteX(matrice,[borneY[j],borneX[i]],hn)
    
    return newMatrice

def densiteX(matrice,I,hn):
    value = 0
    tupleI= tuple(I)
    for j in matrice :
        value += indicatriceHyperCube( (I-j)/hn )
    return value/(len(matrice[:,0])*hn**(len(tupleI)))

def indicatriceHyperCube(vectorX):
    for x in vectorX :
        if abs(x) > 1/2 :
            return 0
    return 1

res2 = Parzen(0.020,geo_mat,100)
plt.figure()
show_map()
plt.imshow(res2,extent=[xmin,xmax,ymin,ymax],interpolation='none',\
               alpha=0.3,origin = "lower")
plt.colorbar()
plt.scatter(geo_mat[:,1],geo_mat[:,0],alpha=0.3,s=4)

############## NOYAU GAUSSIEN ##############

def gaussien(alpha,matrice,n):
    
    borneX=np.linspace(xmin,xmax,n)
    borneY = np.linspace(ymin,ymax,n)
    
    newMatrice = np.zeros((n,n))
    
    for i in range(len(borneX)) :
        for j in range(len(borneY)) :
            newMatrice[j,i] = densiteX_gaussian(matrice,[borneY[j],borneX[i]],alpha)
    
    return newMatrice

def densiteX_gaussian(matrice,I,alpha):
    value = 0
    tupleI= tuple(I)
    for j in matrice :
        value += gaussianValue((I-j),alpha)
    return value/(len(matrice[:,0])*alpha**(len(tupleI)))

def gaussianValue(vectorX,alpha):
    distance=0
    for x in vectorX :
        distance+=x**2
    return (1/((alpha**2)*2*np.pi))*np.exp(-(distance)/(2*(alpha**2)))

res2 = gaussien(0.008,geo_mat,100)
plt.figure()
show_map()
plt.imshow(res2,extent=[xmin,xmax,ymin,ymax],interpolation='none',\
               alpha=0.3,origin = "lower")
plt.colorbar()
plt.scatter(geo_mat[:,1],geo_mat[:,0],alpha=0.3,s=4)

############## QUESTIONS ##############
# 1) Méthode des histogrammes :
# Si on discrètise trop (avec un steps trop petit), on obtient 
# de trop grandes "cases" et il y a une risque de sous-apprentissage.
# Au contraire si on ne discrètise pas assez (un nombre de steps grand),
# il y a un risque de sur-apprentissage.
#
# 2) Méthode à noyaux :
# Par la méthode à noyaux, nous avons 2 paramètres à chaque fois.
# Le paramètre n présent à la fois pour Parzen et pour le noyau gaussien
# sert en fait pour déterminer le nombre de points d'intérêt pour pouvoir
# les afficher. Il vaut mieux avoir une valeur plus élevée pour 
# que ça soit plus précis, mais attention le nombre de points d'intérêt
# est n**2, si n est trop grand le temps de calcul augmente beaucoup.
# Pour les fenetres de Parzen, le paramètre hn fait varier la taille 
# de l'hypercube. Si il est trop grand, on sous-apprend et s'il est 
# trop petit il y a un risque de sur-apprentissage, en particulier si on
# a peu de données.   
# Enfin, pour le noyau gaussien, on utilise le paramètre alpha, si 
# alpha est élevé, on prend plus en compte les points éloignés 
# et on risque de sous-apprendre et si alpha est faible les points éloignés
# sont peu pris en compte en on risque de sur-apprendre.
#
# 3) Pour la méthode des histogrammes et la méthode des noyaux, il faut choisir
# minutieusement les valeurs de ces hyper-paramètres pour éviter,
# ni de sous-apprendre, ni de sur-apprendre.
        
        
        
        
        
        
        
        
        
        
        
        
        
