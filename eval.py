import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import single_meteor_score
from initial_summary import create_pipeline
from sentiment import get_sentiment
from generate_summary import generate_summary1
import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("./nlp_bert/nlptown_bert_model")
model = AutoModel.from_pretrained("./nlp_bert/nlptown_bert_model")

def calculate_bert_score(generated, reference, model, tokenizer):
        gen_tokens = tokenizer(generated, return_tensors='pt', padding=True, truncation=True)
        ref_tokens = tokenizer(reference, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            gen_embeddings = model(**gen_tokens).last_hidden_state.mean(dim=1)
            ref_embeddings = model(**ref_tokens).last_hidden_state.mean(dim=1)
        cosine_similarity = torch.nn.functional.cosine_similarity(gen_embeddings, ref_embeddings)
        return cosine_similarity.item()
# Function to compute BLEU, ROUGE, and METEOR scores
def calculate_scores(reference, generated):
    # BLEU score
    smooth_func = SmoothingFunction().method1
    bleu_score = sentence_bleu([reference.split()], generated.split(), smoothing_function=smooth_func)
    
    # ROUGE score
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference, generated)
    
    # METEOR score
    reference_tokens = word_tokenize(reference)
    generated_tokens = word_tokenize(generated)
    meteor_score = single_meteor_score(reference_tokens, generated_tokens)

    #Bert score
    bert_score = calculate_bert_score(generated, reference, model, tokenizer)
    
    return bleu_score, rouge_scores, meteor_score ,bert_score


pipeline = create_pipeline("./llama_model")

# Path to input JSON file
input_file_path = './dataset/training1/product_csum_test.json'

# Read the test data
with open(input_file_path, 'r', encoding='UTF-8') as file:
    test_data = json.load(file)

# Initialize cumulative scores
total_bleu = 0
total_rouge1 = 0
total_rouge2 = 0
total_rougeL = 0
total_meteor = 0
total_bert = 0
num_samples = 0

import json
test_results = test_data.copy()

# Generate summaries and calculate scores
for key, data in test_results.items():
    text = data["text"]
    reference_summary = " ".join(data["summary_list"])  # Combine summary list into a single string
    # Generate the summary
    print("Generating Summary for:",key)
    generated_summary = generate_summary1(pipeline, text)
    if generated_summary.lower().startswith("summarized review:"):
        generated_summary = generated_summary[len("Summarized Review:"):].strip()
    
    # Add generated summary to the JSON structure
    test_results[key]["generated_summary"] = generated_summary
    test_results[key]["sentiment"] = get_sentiment(generated_summary)

    bleu, rouge, meteor,bert = calculate_scores(reference_summary, generated_summary)
    total_bleu += bleu
    total_rouge1 += rouge['rouge1'].fmeasure
    total_rouge2 += rouge['rouge2'].fmeasure
    total_rougeL += rouge['rougeL'].fmeasure
    total_meteor += meteor
    total_bert += bert
    num_samples += 1
    print("done",key)

# Save the modified test data with generated summaries to a JSON file
output_json_path = './test_results.json'
try:
    with open(output_json_path, 'w', encoding='UTF-8') as json_file:
        json.dump(test_results, json_file, indent=4, ensure_ascii=False)
    print(f"Generated summaries saved to {output_json_path}")
except Exception as e:
    print(f"An error occurred while saving the JSON file: {e}")

# Calculate average scores 
avg_bleu = total_bleu / num_samples
avg_rouge1 = total_rouge1 / num_samples
avg_rouge2 = total_rouge2 / num_samples
avg_rougeL = total_rougeL / num_samples
avg_meteor = total_meteor / num_samples
avg_bert = total_bert / num_samples


# Save evaluation results to a text file 
evaluation_text = f"""
Average scores:
  BLEU: {avg_bleu:.4f}
  ROUGE-1: {avg_rouge1:.4f}
  ROUGE-2: {avg_rouge2:.4f}
  ROUGE-L: {avg_rougeL:.4f}
  METEOR: {avg_meteor:.4f}
  BERT_SCORE: {avg_bert:.4f}
"""

output_eval_path = './evaluation.txt'
try:
    with open(output_eval_path, 'w', encoding='UTF-8') as file:
        file.write(evaluation_text)
    print(f"Evaluation results saved to {output_eval_path}")
except Exception as e:
    print(f"An error occurred while saving the evaluation results: {e}")
