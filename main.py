import pandas as pd 
import numpy as np
from sklearn.neighbors import NearestNeighbors

movies=pd.read_csv("data/movies.csv")
ratings=pd.read_csv("data/ratings.csv")

print(movies.head())
print(ratings.head())

#ici je cree une matrice pour que le knn l'utilise cad en utilisant le fichier rating.csv
#les lignes= movieid (film)
#colonne=userid (utilisateur)
#value=rating

movie_user_matrix=ratings.pivot_table(
    index="movieId",
    columns="userId",
    values="rating"
)

#ici on remplace nan par 0 car il ya des utilisateur qui non pas vu des film ou pas noté des film donc on met nan=0 pour que le knn comprenne que ya pas d'infos

movie_user_matrix=movie_user_matrix.fillna(0)

print(movie_user_matrix.shape)  #matrcie de (movieid, userId)
print(movie_user_matrix.head()) #affiche les 5 premiere ligne

#maintenant on applique le knn sur notre matric mais le pb elle est sous pandas
#mais sklearn travail avec les tableau numpy donc on transforme notre matrice en tableau numpy

data=movie_user_matrix.to_numpy()

knn=NearestNeighbors(metric="cosine")
knn.fit(data)


#ici on definit une fonction pour mettre l'input de l'utilisateur en minuscul et retirer les espaces inutil pour trouver le nom exact du film dans notre dataset
def clear_texte(s):
    s=s.lower().strip()  #strip() : enlève les espaces au début et à la fin 
    s=" ".join(s.split())   #enlève les espaces en trop
    return s



movie_user= input("ecrivez le nom du film ici : ")

G=clear_texte(movie_user)

movie_id_chosen=None



"""
ici: 
Je parcours tous les films

Je nettoie le titre du film

Je regarde si ce que l'utilisateur a tapé est dedans

Si oui :

je prends le movieId

j'affiche le titre

j'arrête la recherche

Si aucun film trouvé :

j'affiche “film introuvable”

j'arrête le programme"""

for i in range(len(movies)):
    title_csv=clear_texte(movies["title"][i])

    if G in title_csv:
        movie_id_chosen=movies["movieId"][i]
        print(f"Le film trouvé est : {movies['title'][i]}")
        break

if movie_id_chosen is None:
    print("Film introuvable. Essaye un mot-clé plus simple (ex: office).")
    exit()




"""
movie_id_chosen = ID du dataset (movies.csv)

movie_position_chosen = position dans la matrice numpy

KNN travaille avec positions

Les titres s'affichent avec movieId"""


#get_loc peut planter si le film n’a aucune note dans la matrice donc : 

if movie_id_chosen not in movie_user_matrix.index:
    print("Le film existe, mais il n'a pas assez de notes dans ratings.csv pour faire une reco.")
    exit()

#conversion du movieId en position pour le Knn 
movie_position_chosen = movie_user_matrix.index.get_loc(movie_id_chosen)


"""
# movie_index est la POSITION du film dans la matrice (pas le movieId)
# exemple : 0 = premier film, 1 = deuxième film, etc.
movie_position_chosen=1  #dans le tableau numpy pas dans la matrice user-movie apres on feras la conversion

#conversion de la position du film dans le tableau a movieId dans la matrice
movie_id_chosen = movie_user_matrix.index[movie_position_chosen]
"""
# on prend la ligne correspondante au film choisi
# data[movie_position_chosen] est en 1D, donc on le met dans une liste
# pour que sklearn le voie comme un tableau 2D (1 film)
movie_inex2D=[data[movie_position_chosen]]


# indices_movies = positions des films proches dans la matrice
d,indices_movies=knn.kneighbors(movie_inex2D,n_neighbors= 5)

#indices_movies[0] = les positions des films proches dans le tableau numpy
# le premier élément (12) est le film lui-même
# [1:] sert à supprimer ce premier élément
voisin_position =indices_movies[0][1:]


# liste vide qui va contenir les vrais movieId des films recommandés
movie_id_voisin=[]

for p in voisin_position:

    # on convertit la position en movieId réel
    # movie_user_matrix.index contient tous les movieId
    movie_id_voisin.append(movie_user_matrix.index[p])



#maintenant pour recuperé le nom du film a partire du movieId
#on cree un dictionnaire clé et titre

movie_id_to_title=dict(zip(movies["movieId"],movies["title"])) #on associe chaque movieId a son titre

#pour voir le film choisi
print(movie_id_to_title[movie_id_chosen])
print("#####################################")
#pour les film recommandé

for k in movie_id_voisin:
    print(movie_id_to_title[k])





