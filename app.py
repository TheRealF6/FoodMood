from flask import Flask, render_template, request
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

app = Flask(__name__, static_folder='static')

# Fuzzy Logic Variables
mood = ctrl.Antecedent(np.arange(0, 11, 1), 'mood')  # Rentang nilai diubah menjadi 0 sampai 10
recommendation = ctrl.Consequent(np.arange(0, 101, 1), 'recommendation')

# Membership functions for mood
mood['negative'] = fuzz.trimf(mood.universe, [0, 0, 5])
mood['neutral'] = fuzz.trimf(mood.universe, [0, 5, 10])
mood['positive'] = fuzz.trimf(mood.universe, [5, 10, 10])

# Membership functions for recommendation
recommendation['low'] = fuzz.trimf(recommendation.universe, [0, 0, 50])
recommendation['medium'] = fuzz.trimf(recommendation.universe, [0, 50, 100])
recommendation['high'] = fuzz.trimf(recommendation.universe, [50, 100, 100])

# Rule Base
rule1 = ctrl.Rule(mood['negative'], recommendation['low'])
rule2 = ctrl.Rule(mood['neutral'], recommendation['medium'])
rule3 = ctrl.Rule(mood['positive'], recommendation['high'])

# Control System
recommendation_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
recommendation_system = ctrl.ControlSystemSimulation(recommendation_ctrl)

# List of Foods
foods = {
    'low': ['Es krim', 'Sayur Sop', 'Sup Krim', 'Bubur Ayam', 'Roti Panggang'],
    'medium': ['Nasi Goreng', 'Mie Goreng', 'Nasi Padang', 'Soto Ayam', 'Pecel Lele'],
    'high': ['Burger', 'Steak', 'Martabak', 'Sate Kambing', 'Kambing Guling']
}

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST', 'GET'])
def recommend():
    if request.method == 'POST':
        user_mood = request.form['mood']

        # Fuzzy Logic Calculation
        recommendation_system.input['mood'] = int(user_mood)
        recommendation_system.compute()

        # Get the recommendation
        food_recommendation_score = recommendation_system.output['recommendation']

        # Determine the category (low, medium, high) based on the recommendation score
        category = None
        if food_recommendation_score <= 48:
            category = 'low'
        elif food_recommendation_score <= 56:
            category = 'medium'
        else:
            category = 'high'

        # Get the list of recommended foods
        recommended_foods = foods[category]

        return render_template('result.html', mood=user_mood, recommendation=food_recommendation_score, foods=recommended_foods)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
