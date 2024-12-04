from initial_summary import *
from using_reward_models import *
from templates import *
from sentiment import *
def generate_summary1(pipeline, input_text):
    """
    Generates a summary using the initialized pipeline and the given prompt.

    Parameters:
        pipeline: The text generation pipeline instance.
        prompt (str): The input prompt to generate the summary.

    Returns:
        str: The generated summary.
    """
    try:
        topic = extract_entity_name(pipeline, input_text)

        aspect_polarity_sentence_prompt = aspect_polarity_template().format(topic=topic, review=input_text)
        aspect_polarity_sentence_prompt = f"{aspect_polarity_sentence_prompt}\n Topic is :{topic} \n Review is:{input_text}"


        aspect_polarity_sentence = generate_aspect_polarity_sentence(pipeline, aspect_polarity_sentence_prompt)

        initial_summary_prompt = initial_summary_template().format(topic=topic, text=aspect_polarity_sentence)

        initial_summary = generate_summary(pipeline, initial_summary_prompt)

        prompt = prompt_for_refinement(input_text, initial_summary, topic)

        if prompt == True:
            summary = initial_summary 
        else:
            print(f"Refinement prompt: {prompt}")
            summary = generate_summary(pipeline, prompt)

        
        return summary.strip()
    
    except Exception as e:
        print(f"An error occurred in generate_summary: {e}")
        return ""
