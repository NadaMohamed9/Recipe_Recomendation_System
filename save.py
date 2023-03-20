from recommendation import recommend_recipes,vectorizer,vectorized_recipes,recipe_index
import joblib
import joblib

# Save the model

# Save the model
joblib.dump(recommend_recipes, "model.joblib")
print("saved")
