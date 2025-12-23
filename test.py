import pandas as pd 
import numpy as np
from sklearn.neighbors import NearestNeighbors


films=pd.read_csv("data/movies.csv")
notes=pd.read_csv("data/ratings.csv")

print(films.head())
print(notes.head())

matrice_knn=notes.pivot_table(
    index="movieId",
    columns="userId",
    values="rating"
)


matrice_knn=matrice_knn.fillna(0)
print(matrice_knn.shape)
print(matrice_knn)

data=matrice_knn.to_numpy()

knn=NearestNeighbors(metric="cosine")
knn.fit(data)




def clean_text(s) :
    s = s.lower().strip()
    s = " ".join(s.split())   # enlève les espaces en trop
    return s




user_title = input("Tape un film : ")

target = clean_text(user_title)
movie_id_chosen = None

for i in range(len(films)):
    title_csv = clean_text(films["title"][i])

    # recherche partielle : ce que tu tapes doit être contenu dans le titre
    if target in title_csv:
        movie_id_chosen = films["movieId"][i]
        print("✅ Trouvé :", films["title"][i])  # on affiche le vrai titre
        break

if movie_id_chosen is None:
    print("❌ Film introuvable. Essaye un mot-clé plus simple (ex: office).")
    exit()

"""
film_name_user=input("entrer le nom du fims ici : ")

film_id=None

for i in range(len(films)):
    if films["title"][i].lower()==film_name_user:  #movies["title"]
                                                    #👉 ça donne toute la colonne title
                                                    #(une liste de titres)
                                                    #movies["title"][i]
                                                    #👉 ça donne le titre à la ligne i
        
        film_id=films["movieId"][i]
        break

if film_id is None:
    print("❌ Film introuvable")
    exit()
else:
    print("✅ Film trouvé")

# conversion du movieId en position dans la matrice
movie_position_chosen = matrice_knn.index.get_loc(film_id)

#get_loc est utiliser dans pandas et veux dire : À quelle position se trouve ce movieId dans l’index ?
"""



#ici on fait ma conversion pour la matrice et non pas le tableau 



#ici on convertie en 2D

film1_tab_2D=[data[movie_id_chosen]]

d,indices=knn.kneighbors(film1_tab_2D,3)
print(indices)

voisin =indices[0][1:]
print(voisin)

#on convertie les position des voiisin en movieId

film_id_voisin=[]



for p in voisin:
    film_id_voisin.append(matrice_knn.index[p])


film_id_to_titre=dict(zip(films["movieId"],films["title"]))

for i in film_id_voisin:
    print(film_id_to_titre[i])