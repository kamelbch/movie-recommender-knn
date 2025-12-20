import pandas as pd 

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

print(movie_user_matrix.shape)
print(movie_user_matrix.head()) #affiche les 5 premiere ligne