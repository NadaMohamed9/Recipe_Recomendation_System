from flask import Flask, render_template, request
from flask import jsonify


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    recipes = []
    if request.method == 'POST':
        ingredients = request.form['ingredients']
        import joblib

        # Load the saved model
        model = joblib.load('model.joblib')

        # Use the loaded model to make predictions
        recipes = model(ingredients)

    return render_template('index.html', recipes=recipes)

if __name__ == '__main__':
    app.run(debug=True)
