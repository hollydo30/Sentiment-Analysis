
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
    similar_texts = k_shot_data[k_shot_data["text"].str.contains(text[:30], na=False)]
    
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


# Initialize weight factor (starts at 0.5, meaning equal trust in model and k-shot)
w = 0.5  

# Reinforcement Learning: Update Weight
def update_weight(correct_label, model_label):
    """Adjusts weight factor (w) based on correct feedback."""
    global w
    alpha = 0.2  # Learning rate (increase for faster updates)
    print(w)

    # Convert labels to numeric scores
    label_map = {"good": 1, "bad": 0}
    correct_score = label_map[correct_label]
    model_score = label_map[model_label]
    print(correct_score)
    print(model_score)

    # Adjust weight based on correctness
    if correct_score == model_score:
        w += alpha  # Decrease trust in model_score
    else:
        w -= alpha  # Increase trust in model_score

    print(w)
    # Keep w within [0,1] range
    w = max(0, min(1, w))


# Classification with RL-based weighting
def classify_sentiment_with_rl(text, k_shot_data):
    """Uses model prediction + k-shot with adaptive weighting."""
    
    # Get model prediction
    model_prediction = classifier(text)
    model_label = "bad" if model_prediction[0]['label'] == "LABEL_0" else "good"

    # Get k-shot label (if available)
    similar_texts = k_shot_data[k_shot_data["text"].str.contains(text[:30], na=False)]
    k_shot_label = similar_texts["label"].values[0] if not similar_texts.empty else model_label
    print(model_label)
    print(k_shot_label)

    # Weighted decision-making
    return model_label if w >= 0.5 else k_shot_label


# Example usage
text_sample = "Trump to announce up to $500 billion in private sector AI infrastructure investment Start doing due diligence on AI infrastructure stocks. There will be many benefitting from this. (large caps and small caps). AI infrastructure needs to be upgraded, energy grids need to be updated to be able to keep up with artificial intelligence."

# Initial prediction
print("Initial Prediction:", classify_sentiment_with_rl(text_sample, k_shot_data))

# Simulating RL update (user provides correct label)
update_weight("good", "bad")  # Adjust based on correct feedback

# Prediction after RL adjustment
print("Updated Prediction after RL:", classify_sentiment_with_rl(text_sample, k_shot_data))


"""
# Dictionary to store feedback (acting as reinforcement learning memory)
rl_feedback_memory = {}

# Function to collect user feedback
def update_feedback(text, correct_label):
    #Store user feedback to reinforce learning.
    rl_feedback_memory[text] = correct_label

# Modify classification function to incorporate RL feedback
def classify_sentiment_with_rl(text, k_shot_data):
    #Uses model prediction, k-shot learning, and reinforcement feedback.
    
    # Check RL feedback memory first
    if text in rl_feedback_memory:
        return rl_feedback_memory[text]  # Use corrected label if available
    
    # Use existing k-shot and model classification logic
    return classify_sentiment_with_k_shot(text, k_shot_data)

# Example usage
text_sample = "Stock market is rising due to strong earnings."
print("Initial Prediction:", classify_sentiment_with_rl(text_sample, k_shot_data))

# Simulating user correction (Reinforcement Learning Step)
update_feedback(text_sample, "good")  # User corrects the label

# The next time the same text appears, RL memory applies the learned correction
print("Updated Prediction after RL:", classify_sentiment_with_rl(text_sample, k_shot_data))
"""


"""
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
"""
