import streamlit as st
from initial_summary import *
from using_reward_models import *
from templates import *
from sentiment import *
from evaluation import calculate_scores  # Importing the calculate_scores function

# Title
st.title("User Review Summarizer and Classifier")

# Step 1: Input File Upload
uploaded_file = st.file_uploader("Upload your input text file:", type=["txt"])
reference_summary = st.text_area("Please enter the reference summary for evaluation:", height=150)

if uploaded_file:
    # Read the input file
    input_text = uploaded_file.read().decode("utf-8").strip()
    st.write("### Input Text")
    st.text_area("Uploaded Text", input_text, height=200)
    
    try:
        #Step 2: Create Pipeline
        st.write("## Step 1: Creating Pipeline...")
        pipeline = create_pipeline("./llama_model")
        st.success("Pipeline created successfully.")

        # Step 3: Extract Entity Name (Topic)
        st.write("## Step 2: Extracting Entity Name...")
        topic = extract_entity_name(pipeline, input_text)
        st.write(f"Extracted Topic: {topic}")

        # Step 4: Generate Aspect Polarity Sentence Prompt
        st.write("## Step 3: Generating Aspect Polarity Sentence...")
        aspect_polarity_sentence_prompt = aspect_polarity_template().format(topic=topic, review=input_text)
        aspect_polarity_sentence_prompt += f"\nTopic: {topic}\nReview: {input_text}"
        aspect_polarity_sentence = generate_aspect_polarity_sentence(pipeline, aspect_polarity_sentence_prompt)
        st.text_area("Aspect Polarity Sentence", aspect_polarity_sentence, height=200)

        # Step 5: Generate Initial Summary Prompt
        st.write("## Step 4: Generating Initial Summary...")
        initial_summary_prompt = initial_summary_template().format(topic=topic, text=aspect_polarity_sentence)
        initial_summary = generate_summary(pipeline, initial_summary_prompt)
        st.text_area("Initial Summary", initial_summary, height=200)

        # Step 6: Refine Prompt (if needed)
        st.write("## Step 5: Refinement Check...")
        prompt = prompt_for_refinement(input_text, initial_summary, topic)
        if prompt:
            st.success("Initial summary is correct. No refinement needed.")
            final_summary = initial_summary
        else:
            st.write("Refinement needed. Generating final summary...")
            final_summary = generate_summary(pipeline, prompt)
        st.text_area("Final Summary", final_summary, height=200)

        # Step 7: Sentiment Analysis
        st.write("## Step 6: Sentiment Analysis...")
        review_sentiment = get_sentiment(final_summary)
        st.write(f"**Sentiment:** {review_sentiment}")

        # Step 8: Save Final Summary
        st.write("## Step 7: Save Summary")
        output_file_path = './summary_output.txt'
        with open(output_file_path, 'w', encoding='UTF-8') as f:
            f.write(final_summary)
        st.success(f"Summary has been saved to {output_file_path}")

        # Allow user to download the file
        with open(output_file_path, 'rb') as f:
            st.download_button(
                label="Download Summary",
                data=f,
                file_name="summary_output1.txt",
                mime="text/plain",
            )

        # STEP 9: Get Reference Summary from User and Calculate the Scores
        st.write("## Step 8: Evaluation Scores")

        # Ask user to input the reference summary
        
        if reference_summary:
                # Calculate the evaluation scores using the reference summary provided by the user
                try:
                    # Calculate the evaluation scores using the reference summary and generated summary
                    scores = calculate_scores(reference_summary, final_summary)  # Call the calculate_scores function
                    st.write("### Evaluation Scores")
                    for metric, value in scores.items():
                        st.write(f"**{metric}:** {value:.4f}")

                except Exception as e:
                    st.error(f"An error occurred: {e}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
