import streamlit as st
import time
import torch
from transformers import pipeline
from transformers import BloomTokenizerFast, BloomForCausalLM
from spellchecker import SpellChecker

spell = SpellChecker()
# Add custom words to the spell checker's dictionary
custom_words = ["negamax", "ssd", "neat", "algorithm", 'explain']
spell.word_frequency.load_words(custom_words)

name = "D:\Repos\My-Digital-Clone\Models\Clone"
# Loading the fine-tuned model
tokenizer = BloomTokenizerFast.from_pretrained(name)
model = BloomForCausalLM.from_pretrained(name)
# Initialize SpellChecker
spell = SpellChecker()

generator = pipeline('text-generation',
                     model=model,
                     tokenizer=tokenizer,
                     do_sample=True,
                     temperature=1,  # Adjust temperature for creativity
                     max_length=512,
                     truncation=True)


def answer(user_input):
    try:
        # Correct spelling mistakes in the user input
        corrected_input = ' '.join([spell.correction(word) for word in user_input.split()])

        # Generate model response
        prompt = f"Given the question {corrected_input}, what is the answer? Answer: "
        result = generator(prompt)

        generated_answer = result[0]['generated_text'][len(prompt):].strip()
        return generated_answer
    except Exception as e:
        print(f"An error occurred: {e}")
        return "An error occurred while generating the answer. Please try again."


# Initialize Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = answer(prompt)
        answer = ""

        # Split response into words and simulate typing
        words = full_response.split()
        typed_response = ""
        for word in words:
            typed_response += word + " "
            message_placeholder.markdown(typed_response + "|")
            time.sleep(0.1)

        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
