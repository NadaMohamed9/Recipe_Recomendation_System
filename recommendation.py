#!/usr/bin/env python
# coding: utf-8

# In[45]:


from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Load the recipe dataset
recipes_df = pd.read_csv(r"food_data.csv")

# Preprocess the data
recipes_df.drop_duplicates(inplace=True)
recipes_df.dropna(inplace=True)

# Feature extraction
vectorizer = TfidfVectorizer()
vectorized_recipes = vectorizer.fit_transform(recipes_df["ingredients"])

# Train the model
recipe_index = pd.Series(recipes_df.index, index=recipes_df["recipe_name"])
import joblib
joblib.dump(vectorizer, "vectorizer.joblib")
joblib.dump(vectorized_recipes, "vectorized_recipes.joblib")
joblib.dump(recipe_index, "recipe_index.joblib")

def recommend_recipes(ingredients):
    # Vectorize the user input ingredients
    recipe_idex = joblib.load('recipe_index.joblib')
    vectorized_recipes_1 = joblib.load('vectorized_recipes.joblib')
    vectorizer_1 = joblib.load('vectorizer.joblib')
    vectorized_input_1 = vectorizer_1.transform([ingredients])

    # Calculate cosine similarity between input and recipes
    cosine_similarities = cosine_similarity(vectorized_input_1, vectorized_recipes_1).flatten()

    # Get the top 5 recommended recipes
    for i in range(5):
        related_indices = cosine_similarities.argsort()[::-1][1:6]

    # print(type(related_indices.tolist()))
    recipes = []
    for i in related_indices.tolist():
        rec = []
        rec.append(recipes_df["recipe_name"].iloc[i])
        rec.append(recipes_df["methode"].iloc[i])
        rec.append(recipes_df["cook_time"].iloc[i])
        rec.append(recipes_df["nutrition"].iloc[i])
        rec.append(recipes_df["img_src"].iloc[i])
        recipes.append(rec)

    return recipes



# In[46]:


ingredients = "chicken, garlic, tomato, pasta"
recommendations = recommend_recipes(ingredients)
# print(len(recommendations))
# print(len(recommendations))

count = 0
for i in recommendations:
    count = count + 1
    print('the info for the recipe number: ', count, 'is shown bellow:')
    print('Name: ', i[0])
    print('Methode: ', i[1])
    print('Cook Time: ', i[2])
    print('Nutrition: ', i[3])
    print('Image: ', i[4])
    print('************************')


# In[ ]:




