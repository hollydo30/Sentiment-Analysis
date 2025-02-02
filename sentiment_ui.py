
import requests
import pandas as pd
from bs4 import BeautifulSoup
from newspaper import Article
from transformers import pipeline
import os
import gradio as gr

# Initialize the sentiment analysis pipeline using DistilBERT
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased")

# Load k-shot labeled data from a CSV
def load_k_shot_data(csv_path):
    df = pd.read_csv(csv_path)
    return df

# Adjust classification based on k-shot learning examples
def classify_sentiment_with_k_shot(text, k_shot_data):
    model_prediction = classifier(text)
    predicted_label = model_prediction[0]['label']  # 'LABEL_0' or 'LABEL_1'
    
    # Convert DistilBERT labels to our labels
    predicted_label = "bad" if predicted_label == "LABEL_0" else "good"

    # Check the k-shot dataset for similar samples
    similar_texts = k_shot_data[k_shot_data["text"].str.contains(text[:50], na=False)]
    
    if not similar_texts.empty:
        # If a similar example exists in the dataset, use its label instead
        return similar_texts["label"].values[0]
    
    return predicted_label  # Otherwise, return model prediction

# Function to scrape news from a URL
def get_news_text(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text

# Preprocess the text to clean it
def preprocess_text(text):
    return text.replace('\n', ' ').replace('\r', ' ').strip()

# Function to classify news sentiment from a URL
def analyze_stock_news(url, k_shot_data):
    # Step 1: Scrape the article
    news_text = get_news_text(url)
    
    # Step 2: Preprocess the article text
    cleaned_text = preprocess_text(news_text)
    
    # Step 3: Get sentiment analysis using k-shot learning
    sentiment = classify_sentiment_with_k_shot(cleaned_text, k_shot_data)
    
    # Step 4: Display the result
    return "Good news for stock!" if sentiment == "good" else "Bad news for stock!"

# Set CSV path
csv_path = "C:/Users/holly/OneDrive/Desktop/k_shot_data.csv"

# Check if the CSV file exists
if os.path.exists(csv_path):
    print("CSV file found!")
else:
    print("CSV file not found. Check the path.")

# Load k-shot data from CSV
k_shot_data = load_k_shot_data(csv_path)

# Test with a news URL
url = 'https://www.reddit.com/r/AMD_Stock/comments/1i6vjn6/trump_to_announce_up_to_500_billion_in_private/'
result = analyze_stock_news(url, k_shot_data)
print(result)

df = pd.read_csv(csv_path) # dictionary of {"text":"label"}
print(df["label"].value_counts())  # Check how many "good" vs "bad" examples


# Chatbot function (Fixed)
def chat_with_bot(url):
    summary = analyze_stock_news(url, k_shot_data)  # Pass k_shot_data
    return f"This is \n{summary}"

# Gradio UI
demo = gr.Interface(
    fn=chat_with_bot,
    inputs=gr.Textbox(label="Enter URL"),
    outputs="text",
    title="Reddit News Sentiment Classifier"
)

# Launch local server
demo.launch()

