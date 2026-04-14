from dotenv import load_dotenv
import streamlit as st
import os

# Providers
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# Load env variables
load_dotenv()

# ------------------- UI -------------------
st.set_page_config(page_title="Multi-Model Chatbot", page_icon="🤖")
st.title("🗪 Multi-Model GenAI Chatbot")
if st.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()

# ------------------- PROVIDER + MODEL -------------------
provider = st.selectbox(
    "Select Provider",
    ["Groq", "OpenRouter", "Gemini"]
)

model_dict = {
    "Groq": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
    "OpenRouter": [
        "openai/gpt-oss-120b:free",
        "nvidia/nemotron-3-super-120b-a12b:free"
    ],
    "Gemini": ["gemini-2.5-flash"]  # only flash-2.5 as 2.5 pro is not available in free tier
}

model = st.selectbox(
    "Select Model",
    model_dict[provider],
    index=0  # ensures flash is default for Gemini
)

# ------------------- SESSION STATE -------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ------------------- DISPLAY CHAT -------------------
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ------------------- LLM INIT -------------------
def get_llm(provider, model):
    if provider == "Groq":
        return ChatGroq(
            model=model,
            temperature=0.3,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
    
    elif provider == "OpenRouter":
        return ChatOpenAI(
            model=model,
            temperature=0.3,
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
    )

    elif provider == "Gemini":
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=0.3,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

llm = get_llm(provider, model)

# ------------------- CHAT INPUT -------------------
user_prompt = st.chat_input("Type your message...")

if user_prompt:
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append(
        {"role": "user", "content": user_prompt}
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        *st.session_state.chat_history
    ]

    response = llm.invoke(messages)
    assistant_response = response.content

    ##For streaming reponses, uncomment below and comment out the above 2 lines 
    # assistant_response = ""

    # with st.chat_message("assistant"):
    #     message_placeholder = st.empty()

    #     try:
    #         for chunk in llm.stream(messages):
    #             if hasattr(chunk, "content") and chunk.content:
    #                 assistant_response += chunk.content
    #                 message_placeholder.markdown(assistant_response + "▌")

    #         message_placeholder.markdown(assistant_response)

    #     except Exception as e:
    #         assistant_response = "⚠️ Error generating response."
    #         message_placeholder.markdown(assistant_response)

    st.session_state.chat_history.append(
        {"role": "assistant", "content": assistant_response}
    )

    with st.chat_message("assistant"):
        st.markdown(assistant_response)