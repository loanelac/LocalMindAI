from flask import Flask, render_template, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Utiliser le modèle BART pour la tâche de résumé
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Route pour la page d'accueil et l'interface de chat
@app.route('/')
def index():
    return render_template('index.html')

# Route pour gérer les interactions de résumé
@app.route('/chat', methods=['POST'])
def handle_chat():
    user_input = request.form['user_input']
    response = generate_summary(user_input)
    return jsonify({'response': response})

# Fonction pour générer un résumé en utilisant le modèle AI
def generate_summary(user_input):
    try:
        summary = summarizer(user_input, max_length=130, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        # Log the error for debugging (in real applications, consider using logging module)
        print(f"Error generating summary: {e}")
        return "Désolé, une erreur s'est produite lors de la génération du résumé."

if __name__ == "__main__":
    app.run(debug=True)
