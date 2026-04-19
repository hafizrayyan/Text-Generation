import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Page config ---
st.set_page_config(page_title="Text Generator", page_icon="✍️", layout="centered")

st.title(" Text Generator")
st.caption("Built By Hafiz Rayyan Asif")

# --- Load model & tokenizer ---
@st.cache_resource
def load_artifacts():
    model = load_model("text_gen_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

try:
    model, tokenizer = load_artifacts()
except Exception as e:
    st.error(f"Could not load model or tokenizer: {e}")
    st.stop()

# --- Helpers ---
def generate_text(seed_text, next_words, max_sequence_len, temperature=1.0):
    output = seed_text
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([output])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding="pre")
        predictions = model.predict(token_list, verbose=0)[0]

        # Temperature scaling
        predictions = np.log(predictions + 1e-7) / temperature
        predictions = np.exp(predictions) / np.sum(np.exp(predictions))

        predicted_index = np.random.choice(len(predictions), p=predictions)

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break

        output += " " + output_word
    return output

# --- UI ---
seed_text = st.text_input("Seed text", placeholder="Enter a starting phrase...")

col1, col2 = st.columns(2)
with col1:
    next_words = st.slider("Words to generate", min_value=5, max_value=100, value=20, step=5)
with col2:
    temperature = st.slider("Creativity (temperature)", min_value=0.1, max_value=2.0, value=1.0, step=0.1)

max_seq_len = model.input_shape[1] + 1  # infer from model

if st.button("Generate", type="primary"):
    if not seed_text.strip():
        st.warning("Please enter a seed text first.")
    else:
        with st.spinner("Generating..."):
            result = generate_text(seed_text, next_words, max_seq_len, temperature)
        st.markdown("### Output")
        st.success(result)
        st.caption(f"Seed: *{seed_text}* · Words generated: {next_words} · Temperature: {temperature}")