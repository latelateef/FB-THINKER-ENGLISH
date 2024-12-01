import transformers
import torch

def create_pipeline(model_path="./llama_model"):
    """
    Creates a pipeline for the given model path.
    """
    try:
        # model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_path,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        return pipeline
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)
        
def generate_aspect_polarity_sentence(pipeline, prompt):
    """
    Generates aspect-polarity-supporting sentences from the given reviews.
    """
    try:
        messages = [
            {"role": "system", "content": "You are a chatbot who extracts aspect-polarity-supporting sentences from the given reviews."},
            {"role": "user", "content": f"{prompt}"},
        ]
        outputs = pipeline(messages, max_new_tokens=512)
        return outputs[0]["generated_text"][-1]['content']
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)

# Extracts Positive/Neutral and Negative/Neutral summaries from the provided text.
# def extract_summaries(output_text):
#     try:
#         sections = output_text.split("###")
#         summaries = {
#             "positive_neutral": "",
#             "negative_neutral": ""
#         }
        
#         for section in sections:
#             if "Positive/Neutral Summary" in section:
#                 summaries["positive_neutral"] = section.split("Positive/Neutral Summary:")[1].strip()
            
#             elif "Negative/Neutral Summary" in section:
#                 summaries["negative_neutral"] = section.split("Negative/Neutral Summary:")[1].strip()
        
#         return summaries
    
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         exit(1)

def generate_summary(pipeline, prompt):
    """
    Generates summaries from the given reviews with provided prompt.
    """
    try:
        messages = [
            {"role": "system", "content": "You are a chatbot who provides summaries of the given reviews."},
            {"role": "user", "content": f"{prompt}"},
        ]
        outputs = pipeline(messages, max_new_tokens=512)
        summary = outputs[0]["generated_text"][-1]['content']
        # summaries = extract_summaries(summary)
        return summary
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)
    
    
def extract_entity_name(pipeline, review):
    """ 
    Extracts entity names from the given review.
    """
    try:
        prompt = f"Please provide only the entity name about which the given review is: {review}"
        messages = [
            {"role": "system", "content": "You are a chatbot who extract entity from a given review."},
            {"role": "user", "content": f"{prompt}"},
        ]
        outputs = pipeline(messages, max_new_tokens=512)
        return outputs[0]["generated_text"][-1]['content']
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)
