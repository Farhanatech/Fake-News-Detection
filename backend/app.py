from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re

app = Flask(__name__)
CORS(app)

# Loading the trained model and vectorizer
model = joblib.load('fake_news_model.pkl')
vector = joblib.load('tfidf_vectorizer.pkl')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    news_text = data['news_text']
    
    # Preprocessing and Vectorization
    cleaned_text = preprocess_text(news_text)
    vectorized_text = vector.transform([cleaned_text])
    
    # Prediction
    prediction = model.predict(vectorized_text)
    result = "Real News" if prediction[0] == 1 else "Fake News"
    
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True, port=5000)