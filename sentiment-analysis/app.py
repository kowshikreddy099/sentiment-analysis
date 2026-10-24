"""Gradio web interface for Real-World Review Sentiment Analysis.
Works out of the box and seamlessly switches to your locally trained models.
"""
import os
import sys
import gradio as gr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from predict import TransformerPredictor, LSTMPredictor

# Initialize predictors
transformer_pred = TransformerPredictor()
lstm_pred = None
if os.path.exists("models/lstm/model.pt") and os.path.exists("models/lstm/vocab.json"):
    try:
        lstm_pred = LSTMPredictor()
    except Exception as e:
        print(f"Note: LSTM model present but failed to load: {e}")

def predict_sentiment(review_text, model_choice):
    if not review_text or not review_text.strip():
        return {"positive": 0.0, "negative": 0.0}
    
    if model_choice == "BiLSTM (Fast / Lightweight)" and lstm_pred is not None:
        return lstm_pred(review_text)
    return transformer_pred(review_text)

available_models = ["Transformer / DeBERTa / BERT (State of the Art)"]
if lstm_pred is not None:
    available_models.append("BiLSTM (Fast / Lightweight)")

demo = gr.Interface(
    fn=predict_sentiment,
    inputs=[
        gr.Textbox(
            lines=4,
            label="Customer / Product / Movie Review",
            placeholder="Type or paste any real-world review here (e.g. Amazon product, Yelp restaurant, IMDb movie)..."
        ),
        gr.Radio(
            choices=available_models,
            value=available_models[0],
            label="Model Architecture"
        )
    ],
    outputs=gr.Label(num_top_classes=2, label="Predicted Sentiment & Confidence"),
    title="Real-World Review Sentiment Analyzer",
    description=(
        "Deep Learning sentiment classifier trained on multi-domain real-world reviews "
        "(Amazon, Yelp, and IMDb). Handles nuanced phrasing, negations, and colloquial review slang."
    ),
    examples=[
        ["I honestly didn't expect much for the price, but it completely exceeded my expectations!", available_models[0]],
        ["Delivery took over three weeks, customer support was rude, and the item arrived broken.", available_models[0]],
        ["The cinematography and score were breathtaking, although the plot dragged slightly in the middle.", available_models[0]],
        ["Not bad at all for a first-generation product, though battery life could certainly be better.", available_models[0]],
        ["Total waste of hard-earned cash. It stopped working after only two days.", available_models[0]],
    ],
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch()
