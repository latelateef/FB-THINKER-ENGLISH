
# FB-THINKER-ENGLISH: Product Review Summarization and Classification  

**Product Review Summarization and Classification** leverages the **FB-Thinker framework** for effective summarization and **BERT-based classification models** for categorizing product reviews. This project addresses key challenges such as factual accuracy, aspect coverage, and content relevance to help consumers make informed purchasing decisions.

## Features  
- **Summarization**: Enhances large language models (LLMs) using multi-objective forward reasoning and multi-reward backward refinement with the FB-Thinker framework.  
- **Classification**: Utilizes fine-tuned BERT models for accurate product review categorization.  
- **Datasets**: Curated datasets, including Product-CSum, and additional datasets using the Gemini framework, for training and evaluation.  

---

## Datasets  
- **Product-CSum**: Reviews sourced from online forums.   
- **Gemini Framework Datasets**: Used to fine-tune LongFormer models as reward models and and BERT model as classifier.

### Dataset Structure  
- **Input**: Raw product reviews.  
- **Output**: Summaries and classification labels.

---

## Setup  

### Clone the Repository  
```bash
git clone https://github.com/latelateef/FB-THINKER-ENGLISH.git  
cd FB-THINKER-ENGLISH
```

### Download Models
1. Download the Llm for summarization, for llama the code is available in `downloading_llama.py`.
2. Download the LongFormer models for backward refinement, for finetuning and saving the code is available in `train_longformer.ipynb` notebook.
3. Download the BERT model for classification from hugging face.

### Generating Summaries

To generate summaries, run the `app.py` script:
```sh
python app.py
```

---

### Files and Directories
- `app.py:` Main application script.
- `generate_summary.py:` Contains functions to generate summaries.
- `initial_summary.py:` Contains functions to generate initial summaries.
- `eval.py:` Script to evaluate summaries.
- `evaluation.py:` Contains functions to evaluate summaries.
- `score.py:` Contains functions to calculate BLEU, ROUGE, METEOR, and BERTScore.
- `sentiment.py:` Contains functions for sentiment analysis.
- `templates.py:` Contains functions to generate templates for text generation.
- `using_reward_models.py:` Contains functions to use reward models for text refinement.
- `downloading_llama.py:` Script to download and save the LLM model.
- `train_longformer.ipynb:` Jupyter notebook to fine-tune and save LongFormer models.
- `language_translate/:` Directory containing code for dataset translation from chinese to english.
- `template_files/:` Directory containing template files for text generation.

---