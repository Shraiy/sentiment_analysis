from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
model_path = r"D:\sentiment_analysis\saved_models\distilbert-base-uncased"  # use raw string or escape backslashes
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Label mapping
label_map = {
    0: "Positive",
    1: "Neutral",
    2: "Negative"
}

# Flask app
app = Flask(__name__)

# Route to serve the frontend HTML page
@app.route('/')
def home():
    return render_template('index.html')

# API endpoint to get sentiment prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    review_summary = data.get('review_summary', '')
    review_text = data.get('review_text', '')
    rating = data.get('rating', '')

    # Combine all inputs into one string
    full_text = f"{review_summary} {review_text} Rating: {rating}"

    # Tokenize and prepare inputs
    inputs = tokenizer(full_text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Predict sentiment
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()

    return jsonify({
        "sentiment": label_map.get(prediction, "Unknown")
    })

if __name__ == '__main__':
    app.run(debug=True)
