#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 11:14:20 2020

@author: 28601744
"""

import math
import decisiontree as dtree
import pickle 
import numpy as np
import random
import matplotlib.pyplot as plt
try:
    import pydot  #pour l'affichage graphique d'arbres
except ImportError:
    print("Pydot non disponible pour l'affichage graphique, allez sur http://www.webgraphviz.com/ pour generer un apercu de l'arbre")


depth=16

[ data , id2titles , fields ]= pickle . load ( open ("imdb_extrait.pkl","rb"))
datax = data [: ,:32]
datay = np.array ([1 if x [33] >6.5 else -1 for x in data ])

entropy=dtree.entropy(datay)
print("Entropie : ",entropy)
for i in range(28):
    #print("entropy vect ",i,dt.entropy(datax[i]))
    entropy_cond=dtree.entropy_cond([datay[datax[:,i]==1],datay[datax[:,i]==0]])
    print("Entropie conditionelle ",i,", ",fields[i]," : ",entropy_cond)
    print("Différence : ",entropy-entropy_cond)
    
#Si la différence entre l'entropie et l'entropie conditionelle vaut 0,
#cela signifie que l'attribut ne permet pas de faire baisser l'entropie du tout,
#le "gain d'informations" est donc nul.
#Pour la première partition, le meilleur attribut est l'attribut "drama" car c'est
#qui a la plus grande différence entre l’entropie et l’entropie conditionnelle,
#autrement dit c'est l'attribut qui permet de minimiser le plus l'entropie
    
for i in range(0,depth):
    dt= dtree.DecisionTree()
    dt.max_depth = i # on fixe la taille de l ’ arbre a i (avec i entre 1 et 10)
    dt.min_samples_split = 2 # nombre minimum d ’ exemples pour spliter un noeud
    dt.fit ( datax , datay )
    dt.predict ( datax [:5 ,:])
    print ("Score pour la profondeur "+str(i)+" : "+str(dt.score( datax , datay )))
    # dessine l ’ arbre dans un fichier pdf si pydot est installe .
    #dt.to_pdf ("./tmp/testtree_deep_"+str(i)+".pdf", fields )
    # sinon utiliser http :// www . webgraphviz . com /
    #dt.to_dot ( fields )
    # ou dans la console
    #print ( dt.print_tree ( fields ))
    
###### Question 1.4
    
###### Question 1.5
#Les scores de bonnes classification augmentent au fur et à mesure que la
#profondeur augmente. C'est normal puisque ce score est obtenu en testant sur 
#les données utilisées pour l'apprentissage.
#Toutefois si on testait sur d'autres données, ce score n'augmenterait plus et
#diminuerait même au bout d'une certaine profondeur.
#Car si la profondeur est trop grande, l'algorithme va apprendre par coeur 
#les données et n'arrivera plus à généraliser.


########## Question 1.6
#    
#Ces scores ne sont pas un indicateur fiable du comportement de l'algorithme, 
#en effet on test notre algorithme sur les données d'apprentissage, 
#ce qu'il ne faut pas faire. 
#En testant de cette manière, on ne peut pas détecter le sur-apprentissage.
#
#Pour avoir un meilleur indicateur il faudrait séparer les données
#de test des données d'apprentissage. Si les données sont en nombre insuffisant,
#on pourrait utiliser la validation croisée.
    
########## Question 1.7
datax_train=np.empty((0,32))
datay_train=np.array([])
datax_test=np.empty((0,32))
datay_test=np.array([])

train_set_rates=[0.2,0.5,0.8]

for train_set_rate in train_set_rates:
    print("Train set rate : "+str(train_set_rate))
    for i in range(len(datax)):
        if random.random()<train_set_rate:
            datax_train=np.append(datax_train,datax[i][np.newaxis],axis=0)
            datay_train=np.append(datay_train,datay[i])
        else:
            datax_test=np.append(datax_test,datax[i][np.newaxis],axis=0)
            datay_test=np.append(datay_test,datay[i])
      
    scores=[]
    for i in range(0,depth):
        dt= dtree.DecisionTree()
        dt.max_depth = i # on fixe la taille de l ’ arbre a i (avec i entre 1 et 10)
        dt.min_samples_split = 2 # nombre minimum d ’ exemples pour spliter un noeud
        dt.fit ( datax_train , datay_train )
        dt.predict ( datax_train)
        score=dt.score( datax_train , datay_train )
        print ("Score pour la profondeur "+str(i)+" : "+str(score)," sur le jeu de train")
        scores.append(score)
        # dessine l ’ arbre dans un fichier pdf si pydot est installe .
        #dt.to_pdf ("./tmp/testtree_deep_"+str(i)+".pdf", fields )
        # sinon utiliser http :// www . webgraphviz . com /
        #dt.to_dot ( fields )
        # ou dans la console
        #print ( dt.print_tree ( fields ))
    plt.plot(scores, label = "Train set rate : "+str(train_set_rate)+" sur le jeu de train")
    
    scores=[]
    for i in range(0,depth):
        dt= dtree.DecisionTree()
        dt.max_depth = i # on fixe la taille de l ’ arbre a i (avec i entre 1 et 10)
        dt.min_samples_split = 2 # nombre minimum d ’ exemples pour spliter un noeud
        dt.fit ( datax_train , datay_train )
        dt.predict ( datax_test)
        score=dt.score( datax_test , datay_test )
        print ("Score pour la profondeur "+str(i)+" : "+str(score)," sur le jeu de test")
        scores.append(score)
        # dessine l ’ arbre dans un fichier pdf si pydot est installe .
        #dt.to_pdf ("./tmp/testtree_deep_"+str(i)+".pdf", fields )
        # sinon utiliser http :// www . webgraphviz . com /
        #dt.to_dot ( fields )
        # ou dans la console
        #print ( dt.print_tree ( fields ))
    plt.plot(scores, label = "Train set rate : "+str(train_set_rate)+" sur le jeu de test")

    plt.xlabel("Profondeur")
    plt.ylabel("Score")
    plt.legend(loc='best')
    plt.show()

########## Question 1.8
#L'algorithme fait beaucoup plus d'erreurs quand il y a peu d'exemples 
#d'apprentissage (quand train_set_rate vaut 0.2).
#On voit que le score semble atteindre son maximum à partir de la profondeur 3
#et il diminue lentement après.

#À contrario, lorsque l'ensemble d'apprentissage est plus grand, l'algorithme 
#fait moins d'erreurs et son score augmente lorsqu'on augmente la profondeur.

########## Question 1.9

###### VALIDATION CROISÉE

# On regroupe les x et y afin de ne pas perdre l'information lors de la permutation
newdata=np.append(datax,datay[:,np.newaxis],axis=1)
np.random.permutation(newdata)

# On choisit le nombre de partition que l'on souhaite
nbPartition = 10
Partition =[]
taillePartition = math.ceil(datax.shape[0]/nbPartition) 
scores=np.zeros((nbPartition,depth))

# Création de la partition
for i in range(nbPartition) :
    Partition.append(newdata[i*taillePartition:(i+1)*taillePartition,:])


for i in range(len(Partition)) :
    
    # On a créer les données d'apprentissage et de test
    Partition_copy = Partition.copy()
    datax_train=np.empty((0,32))
    datay_train=np.array([])
    datax_test=np.empty((0,32))
    datay_test=np.array([])
    
    data_test = Partition_copy.pop(i)
    datax_test = data_test[:,:32]
    datay_test = data_test[:,-1]
      
    for j in Partition_copy :
        datax_train = np.append(datax_train,j[:,:32],axis = 0)
        datay_train = np.append(datay_train,j[:,-1][np.newaxis])
        
    #On calcule le score de chaque partition et pour différentes profondeur    
    for j in range(0,depth):
        dt= dtree.DecisionTree()
        dt.max_depth = j # on fixe la taille de l ’ arbre a i (avec i entre 1 et 10)
        dt.min_samples_split = 2 # nombre minimum d ’ exemples pour spliter un noeud
        dt.fit ( datax_train , datay_train )
        dt.predict ( datax_test)
        score=dt.score( datax_test , datay_test )
        print ("Score pour la profondeur "+str(j)+" : "+str(score))
        scores[i,j] = score
        # dessine l ’ arbre dans un fichier pdf si pydot est installe .
        #dt.to_pdf ("./tmp/testtree_deep_"+str(i)+".pdf", fields )
        # sinon utiliser http :// www . webgraphviz . com /
        #dt.to_dot ( fields )
        # ou dans la console
        #print ( dt.print_tree ( fields ))

means=[]

for i in range(depth) :
    mean=np.mean(scores[:,i])
    means.append(mean)
    print("Erreur moyenne pour une profondeur de ",i," est de :",mean)
    
plt.plot(means)
plt.xlabel("Profondeur")
plt.ylabel("Score")
plt.show()
    
print("Le modèle le plus efficace est celui avec la profondeur : ",means.index(max(means)))







