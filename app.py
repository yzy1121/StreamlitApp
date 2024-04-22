from transformers import pipeline
import streamlit as st


summarizer = pipeline(task="summarization", model="Falconsai/text_summarization")

print("Pipeline device:", summarizer.device)

st.title("💬Summarize Your Text")
st.caption("🚀 A streamlit application powered by HuggingFace Model")

# Set up session state to manage the app's state
if "summarization_result" not in st.session_state:
    st.session_state["summarization_result"] = ""

input_text = st.text_area("Enter your text", height=300, help = "enter the text you want to summarize")

if st.button("Summarize"):
    if input_text:
        summarization = summarizer(input_text, max_length=50, min_length=20, do_sample=False)
        
        st.session_state["summarization_result"] = summarization[0]['summary_text']
    else:
        st.warning("Please enter some text to summarize.")

# Display the summarization result
st.subheader("Text Summarization Result")
st.write(st.session_state["summarization_result"])