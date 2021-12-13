from os import close
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data=pd.read_csv("movie_dataset.csv")

data.head()

data.shape

selected_features=["genres","keywords","tagline","cast","director"]
#print(selected_features)

for feature in selected_features:
    data[feature]=data[feature].fillna('')

combined_features=data["genres"]+' '+data["keywords"]+' '+data["tagline"] +' '+data["cast"] +' '+data["director"]    

#print(combined_features)

vectorizer=TfidfVectorizer()

feature_vectors=vectorizer.fit_transform(combined_features)

#print(feature_vectors)

similarity=cosine_similarity(feature_vectors)

movie_name=input("Enter your favourite movie name: ")

list_of_all_titles=data["title"].tolist()

find_close_match=difflib.get_close_matches(movie_name,list_of_all_titles)

close_match=find_close_match[0]
#print(close_match)

index_of_the_movie=data[data.title==close_match]["index"].values[0]
#print(index_of_the_movie)

similarity_score=list(enumerate(similarity[index_of_the_movie]))

sorted_similar_movies=sorted(similarity_score,key=lambda x:x[1],reverse=True)

i=1

for movie in sorted_similar_movies:
  index = movie[0]
  title_from_index = data[data.index==index]['title'].values[0]
  if (i<30):
    print(i, '.',title_from_index)
    i+=1