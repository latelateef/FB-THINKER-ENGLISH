from transformers import AutoModelForSequenceClassification, AutoTokenizer

def load_bert_model(model_name = "nlptown/bert-base-multilingual-uncased-sentiment"):
    try:
        tokenizer = AutoTokenizer.from_pretrained("./nlp_bert/nlptown_bert_model")
        model = AutoModelForSequenceClassification.from_pretrained("./nlp_bert/nlptown_bert_model")
        return model, tokenizer
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)

def output_sentiment(model, tokenizer, text):
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = logits.argmax(dim=-1).item()
        return predicted_class
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)

def review_class(rating):
    if rating == 0:
        return "Terrible"
    elif rating == 1:
        return "Poor"
    elif rating == 2:
        return "Average"
    elif rating == 3:
        return "Good"
    elif rating == 4:
        return "Excellent"
    else:
        return "Invalid rating. Please enter a rating between 1 and 5."


def get_sentiment(text):
    try:
        model, tokenizer = load_bert_model()
        sentiment = output_sentiment(model, tokenizer, text)
        return review_class(sentiment)
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)