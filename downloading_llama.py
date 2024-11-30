from huggingface_hub import login
import transformers
import torch

try:
    with open("hugging_face_token.txt") as f:
        token = f.read()
        login(token=token)
except Exception as e:
    print(f"An error occurred: {e}")
    exit(1)
    
    
try:
    # change with your model id from huggingface.co
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    # Saving the model after use
    save_directory = "llama_model"  # Specify your desired save path
    pipeline.model.save_pretrained(save_directory)  # Save model weights
    pipeline.tokenizer.save_pretrained(save_directory)  # Save tokenizer
except Exception as e:
    print(f"An error occurred: {e}")
    exit(1)
