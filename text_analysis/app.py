import gradio as gr
from transformers import pipeline
import requests
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Load the Sentiment Analysis pipeline...
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# Define your Telegram bot token and chat ID
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Function to send messages to the Telegram bot
def send_to_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, data=payload)
    except requests.exceptions.RequestException as e:
        print(f"Failed to send message to Telegram: {e}")

# Define the prediction function...
def sentiment_predictor(text):
    if not text:
        return "Please enter some text.", 0.0

    result = classifier(text)[0]
    label = result['label']
    score = result['score']

    output_text = f"Predicted Sentiment: **{label}**"

    # Send input and output to the Telegram bot
    message = (
        f"*New Sentiment Analysis Result:*\n"
        f"*Input:* {text}\n"
        f"*Output:* {output_text}\n"
        f"*Confidence Score:* {score:.2f}"
    )
    send_to_telegram(message)

    return output_text, score

# Create the Gradio Interface
iface = gr.Interface(
    fn=sentiment_predictor,
    inputs=gr.Textbox(lines=5, placeholder="Type a sentence here...", label="Enter Text for Sentiment Analysis"),
    outputs=[
        gr.Markdown(label="Analysis Result"),
        gr.Number(label="Confidence Score")
    ],
    title="🤗 Simple Sentiment Analyzer on Hugging Face Spaces",
    description="A demonstration of deploying a DistilBERT-based model for sentiment classification using Gradio and Hugging Face Spaces. Type in any sentence and see the prediction!",
    # The allow_flagging argument is now obsolete and removed.
)

# Launch the interface
iface.launch()