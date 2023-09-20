import streamlit as st
from util import generate

st.title("Service Description Generator")

# Get user input for OpenAI API key and service title
openai_key = st.text_input("Enter your OpenAI API key", type="password")
service_title = st.text_input("Enter the service title")
# lang = st.selectbox("Select the language", ("English", "Spanish", "French", "Italian", "Japanese", "Korean"))
# minWords = st.number_input("Enter the minimum number of words", value=2048, max_value=5000)

# Button to generate service description
if st.button("Generate"):
    # Execute the 'generate' function if both inputs are provided

    if openai_key and service_title:
        service_description = generate(serviceTitle=service_title, openai_api_key=openai_key)

        # Display the generated service description
        st.subheader("Generated Service Description")
        st.text_area("Copy the service description", value=service_description, height=200)
