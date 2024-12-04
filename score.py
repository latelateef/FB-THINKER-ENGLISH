import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModel
import torch
from sentiment import get_sentiment

# Paths to BERT model and tokenizer
bert_model_path = "./nlp_bert/nlptown_bert_model"
tokenizer_path = "./nlp_bert/nlptown_bert_model"

# Load BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
bert_model = AutoModel.from_pretrained(bert_model_path)


def calculate_bert_score(generated, reference, model, tokenizer):
    gen_tokens = tokenizer(generated, return_tensors="pt", padding=True, truncation=True)
    ref_tokens = tokenizer(reference, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        gen_embeddings = model(**gen_tokens).last_hidden_state.mean(dim=1)
        ref_embeddings = model(**ref_tokens).last_hidden_state.mean(dim=1)
    cosine_similarity = torch.nn.functional.cosine_similarity(gen_embeddings, ref_embeddings)
    return cosine_similarity.item()


def calculate_scores(reference_summary, generated_summary, bert_model, tokenizer):
    # Step 1: BLEU Score
    smooth_func = SmoothingFunction().method1
    bleu_score = sentence_bleu([reference_summary.split()], generated_summary.split(), smoothing_function=smooth_func)

    # Step 2: ROUGE Score
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = scorer.score(reference_summary, generated_summary)

    # Step 3: METEOR Score
    meteor_score = single_meteor_score(word_tokenize(reference_summary), word_tokenize(generated_summary))

    # Step 4: BERTScore
    bert_score = calculate_bert_score(generated_summary, reference_summary, bert_model, tokenizer)

    return {
        "BLEU Score": bleu_score,
        "ROUGE-1": rouge_scores["rouge1"].fmeasure,
        "ROUGE-2": rouge_scores["rouge2"].fmeasure,
        "ROUGE-L": rouge_scores["rougeL"].fmeasure,
        "METEOR Score": meteor_score,
        "BERTScore": bert_score,
    }


def process_json(input_json_path, output_json_path, output_text_path):
    # Load JSON file
    with open(input_json_path, "r") as file:
        data = json.load(file)

    # Initialize score accumulators
    score_totals = {
        "BLEU Score": 0.0,
        "ROUGE-1": 0.0,
        "ROUGE-2": 0.0,
        "ROUGE-L": 0.0,
        "METEOR Score": 0.0,
        "BERTScore": 0.0,
    }
    num_summaries = len(data)

    # Loop through all objects in JSON
    for key, value in data.items():
        reference_summary = " ".join(value["summary_list"])
        generated_summary = value["generated_summary"]

        # Calculate all scores
        scores = calculate_scores(reference_summary, generated_summary, bert_model, tokenizer)

        # Perform sentiment analysis
        sentiment = get_sentiment(generated_summary)

        # Update JSON object
        data[key]["scores"] = scores
        data[key]["sentiment"] = sentiment

        # Accumulate scores for averaging
        for score_key in scores:
            score_totals[score_key] += scores[score_key]

    # Calculate averages
    score_averages = {key: total / num_summaries for key, total in score_totals.items()}

    # Save the updated JSON to the output file
    with open(output_json_path, "w") as file:
        json.dump(data, file, indent=4)

    # Write final results to a text file
    with open(output_text_path, "w") as file:
        file.write("Average Scores Across All Summaries:\n")
        for score_key, average in score_averages.items():
            file.write(f"{score_key}: {average:.4f}\n")
        file.write("\nResults saved to updated_test_results.json.\n")


# Input and output file paths
input_json_path = "test_results.json"
output_json_path = "updated_test_results.json"
output_text_path = "test_results.txt"

# Process the JSON file and compute averages
process_json(input_json_path, output_json_path, output_text_path)
