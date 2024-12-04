from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import single_meteor_score
from transformers import AutoTokenizer, AutoModel
import torch

def calculate_scores(reference_summary, generated_summary):
    """
    Calculates BLEU, ROUGE, METEOR, and BERTScore between a reference and a generated summary.
    
    Args:
        reference_summary (str): The reference text.
        generated_summary (str): The generated text.

    Returns:
        dict: A dictionary containing BLEU, ROUGE-1, ROUGE-2, ROUGE-L, METEOR, and BERTScore.
    """
    # Step 1: BLEU Score
    smooth_func = SmoothingFunction().method1
    bleu_score = sentence_bleu([reference_summary.split()], generated_summary.split(), smoothing_function=smooth_func)

    # Step 2: ROUGE Score
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference_summary, generated_summary)

    # Step 3: METEOR Score
    reference_summary_tokens = word_tokenize(reference_summary)
    generated_summary_tokens = word_tokenize(generated_summary)
    meteor_score = single_meteor_score(reference_summary_tokens, generated_summary_tokens)

    # Step 4: BERTScore using local BERT model
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

    bert_score = calculate_bert_score(generated_summary, reference_summary, model, tokenizer)
    
    # Return all scores as a dictionary
    return {
        "BLEU Score": bleu_score,
        "ROUGE-1": rouge_scores["rouge1"].fmeasure,
        "ROUGE-2": rouge_scores["rouge2"].fmeasure,
        "ROUGE-L": rouge_scores["rougeL"].fmeasure,
        "METEOR Score": meteor_score,
        "BERTScore": bert_score,
    }

# reference_summary = "The advantages of Werland Double Engine four -wheel drive cars are good power, low fuel consumption, good chassis texture, good steering feel, and exquisite appearance, but there are unreasonable configurations, obvious vibrations when the engine startsSmall shortcomings."
# generated_summary = "The Toyota RAV4 Hybrid AWD offers excellent fuel efficiency, responsive power, and a refined chassis, making it ideal for urban use. The ECO mode ensures quiet and efficient driving, while Sport mode enhances performance. Its sleek design, good steering feedback, and low-speed light handling add to the appeal.However, drawbacks include a manual co-pilot seat, noticeable engine vibration at startup, inconvenient touch controls for volume, and undersized 18-inch wheels that compromise aesthetics and handling slightly. Overall, it’s a solid hybrid SUV with room for improvement in practicality and comfort."
# result1 = calculate_scores(reference_summary,generated_summary)
# print(result1)