from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json

def load_reward_model(reward):
    try:
        model_path = f"./reward_models/{reward}"
        tokenizer = AutoTokenizer.from_pretrained(model_path, clean_up_tokenization_spaces=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
        return model, tokenizer
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)

def predict_label(model, tokenizer, input_text):
    try:
        inputs = tokenizer(input_text, return_tensors="pt", padding="max_length", max_length=1024, truncation=True)
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = logits.argmax(dim=-1).item()
        return predicted_class
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)

def prompt_for_refinement(input_text, initial_summary, name):
    try:
        factual_model, factual_tokenizer = load_reward_model("factual")
        compre_model, compre_tokenizer = load_reward_model("compre")
        relate_model, relate_tokenizer = load_reward_model("relate")

        factual_label = predict_label(factual_model, factual_tokenizer, initial_summary)
        compre_label = predict_label(compre_model, compre_tokenizer, initial_summary)
        relate_label = predict_label(relate_model, relate_tokenizer, initial_summary)

        hard_templates_path = "./template_files/improve_templates.json"
        with open(hard_templates_path,'r') as f:
            hard_templates = json.load(f)
            
        if factual_label==1 and compre_label==0 and relate_label==0:
            prompt = hard_templates['fact_template'].format(
                text=input_text, summary=initial_summary, name=name
            )
        elif factual_label==0 and compre_label==1 and relate_label==0:
            prompt = hard_templates['compre_template'].format(
                text=input_text, summary=initial_summary, name=name
            )
        elif factual_label==0 and compre_label==0 and relate_label==1:
            prompt = hard_templates['relate_template'].format(
                text=input_text, summary=initial_summary, name=name
            )
        elif factual_label==1 and compre_label==1 and relate_label==0:
            prompt = hard_templates['fact_compre_template'].format(
                text=input_text, summary=initial_summary, name=name
            )
        elif factual_label==0 and compre_label==1 and relate_label==1:
            prompt = hard_templates['compre_relate_template'].format(
                text=input_text, summary=initial_summary, name=name
            )
        elif factual_label==1 and compre_label==0 and relate_label==1:
            prompt = hard_templates['relate_fact_template'].format(
                text=input_text, summary=initial_summary, name=name
            )
        elif factual_label==1 and compre_label==1 and relate_label==1:
            prompt = hard_templates['tri_template'].format(
                text=input_text, summary=initial_summary, name=name
            )
        elif factual_label==0 and compre_label==0 and relate_label==0:
            return True
        
        return prompt
    
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)