import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the fine-tuned model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    model_path = './poem_model'  # Path to the fine-tuned model
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# Function to generate poem
def generate_poem(prompt, max_length=50):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(input_ids, max_length=max_length, num_return_sequences=1)
    generated_poem = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_poem

# Streamlit UI
st.set_page_config(page_title="Poem Generator", page_icon="📜", layout="wide")

# App title and description
st.title("Poetry Generator with GPT-2")
st.markdown("""
    Welcome to the Poetry Generator app! ✨

    This app uses a fine-tuned GPT-2 model to generate poems based on your input.
    Simply type a prompt, and the model will generate a creative poem for you.
""")

# Text input for the prompt
prompt = st.text_area("Enter the first few words or line of the poem:", 
                      "The sky is filled with", height=150)

# Slider for maximum length of the poem
max_length = st.slider("Select the length of the poem:", min_value=10, max_value=200, value=50, step=10)

# Button to generate poem
if st.button("Generate Poem"):
    with st.spinner("Generating poem..."):
        poem = generate_poem(prompt, max_length)
    st.markdown("### Generated Poem:")
    st.write(poem)

# Customize the page with some styling
st.markdown("""
    <style>
        .stTextArea textarea {
            font-family: 'Courier New', monospace;
            font-size: 20px;
        }
        .stButton button {
            background-color: #FF6347;
            color: white;
            font-size: 16px;
            font-weight: bold;
        }
        .stButton button:hover {
            background-color: #FF4500;
        }
    </style>
""", unsafe_allow_html=True)
