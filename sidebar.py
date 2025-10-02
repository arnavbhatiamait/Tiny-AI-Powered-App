import streamlit as st
import pandas as pd
import datetime
import os
import subprocess
from langchain_core.prompts import ChatPromptTemplate
import requests
from langchain.tools import Tool, tool
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAI
from langchain.agents import Tool
from langchain_ollama import ChatOllama
import google.generativeai as genai
def get_ollama_models():
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')
        models = []
        for line in lines[1:]:
            parts = line.split()
            if parts:
                models.append(parts[0])
        return models
    except subprocess.CalledProcessError as e:
        print("Error running 'ollama list':", e)
        print("Output:", e.stdout)
        print("Error Output:", e.stderr)

def get_all_groq_models(api_key: str) -> list:
    url = "https://api.groq.com/openai/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        models_data = response.json()
        return [model["id"] for model in models_data.get("data", [])]
    except requests.RequestException as e:
        print(f"Error fetching Groq models: {e}")
        return []

def get_all_gemini_models(api_key):
    try:
        genai.configure(api_key=api_key)
        models = genai.list_models()
        gemini_models = [model.name for model in models if "gemini" in model.name.lower()]
        return gemini_models
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def sidebar():
    with st.sidebar:
        st.title("Model Selection")
        model_option = None
        openai_api_key = " "
        model_list = [
            "Ollama",
                       "Open AI", "Groq", "Gemini"]
        model = st.selectbox("Select Model", model_list)

        if model == "Open AI":
            openai_api_key = st.text_input("Enter your OpenAi API key", type="password")
            st.divider()
            if openai_api_key:
                model_option = st.selectbox(
                    "Select AI Model",
                    ["gpt-3.5-turbo (Fast)", "gpt-4o (High Quality)"],
                    help="GPT 3.5 is faster than GPT 4"
                )
                llm = ChatOpenAI(model=model_option, api_key=openai_api_key, verbose=1, temperature=0.1)
            else:
                st.warning("enter API Key")

        elif model == "Ollama":
            ollama_list = get_ollama_models()
            model_option = st.selectbox("Select AI Model", ollama_list)
            llm = ChatOllama(model=model_option, verbose=1, temperature=0.1)

        elif model == "Groq":
            groqapi_key = st.text_input("Enter your Groq API key", type="password")
            st.divider()
            openai_api_key = groqapi_key
            if groqapi_key:
                groq_list = get_all_groq_models(api_key=groqapi_key)
                model_option = st.selectbox("Select AI Model", groq_list)
                llm = ChatGroq(model=model_option, api_key=groqapi_key, verbose=1, temperature=0.1)
            else:
                st.warning("enter API Key")

        elif model == "Gemini":
            Geminiapi_key = st.text_input("Enter your Gemini API key", type="password")
            st.divider()
            openai_api_key = Geminiapi_key
            if Geminiapi_key:
                gemini_list = get_all_gemini_models(api_key=Geminiapi_key)
                model_option = st.selectbox("Select AI Model", gemini_list)
                llm = GoogleGenerativeAI(api_key=Geminiapi_key, verbose=1, temperature=0.1, model=model_option)
            else:
                st.warning("enter API Key")

    return llm