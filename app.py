from sidebar import sidebar, get_all_gemini_models, get_all_groq_models, get_ollama_models
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
# st.set_page_config("wide")
from langchain.chains.summarize import load_summarize_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.document_loaders import UnstructuredURLLoader
from pypdf import PdfReader
from datetime import datetime
import pandas as pd
st.set_page_config(layout="wide", page_title="Tiny AI-Powered App")

def text_summarizer(text, summary_size, llm, system_prompt):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    docs_full = text_splitter.split_text(text)
    docs = [Document(page_content=doc) for doc in docs_full]
    system_prompt="""You are a heplful assistant. Your task is to summarize the given text in a concise manner. The summary should capture the main points and essence of the original text while being significantly shorter. Ensure that the summary is coherent and easy to understand. The summary should be approximately  {summary_size} words long. Provide the summary in markdown format.
    The text to summarize is as follows:
    {text}
    
    """
    map_prompt = PromptTemplate(
        input_variables=["text"],
        template=system_prompt,
        partial_variables={"summary_size": summary_size},
    )
    chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=map_prompt,verbose=True)
    result = chain.invoke({"input_documents": docs})
    summary = result['output_text']
    st.markdown("**Summary:**")
    st.markdown(summary)

def read_pdf(uploaded_file):
    
    if uploaded_file is not None:
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""  # Safely handle pages with no text
        return text.strip()
    return ""

if st.session_state.get("history") is None:
    st.session_state["history"] = []
st.title("Tiny AI-Powered App")

def webpage_summarizer(url, summary_size, llm, system_prompt):
   loader=UnstructuredURLLoader(urls=[url])
   documents=loader.load()
   if not documents:
        st.warning("No content found at the provided URL.")
        return
   else:
        st.success("Content successfully retrieved from the URL.")
   text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
   docs = text_splitter.split_documents(documents)
   content=docs[0].page_content
   text_summarizer(content, summary_size, llm, system_prompt)

llm=sidebar()
system_prompt="you are a helpful assistant Your name is Tiny AI and your role is to provide detailed and accurate information to users based on their queries. You should be friendly, approachable, and professional in your responses. Always strive to understand the user's intent and provide relevant answers. If you don't know the answer, it's okay to admit it rather than providing incorrect information. Provide examples wherever possible to illustrate your points and the response should be strictly in markdown format."
st.session_state.history.insert(0, SystemMessage(content=system_prompt))
tab1, tab2 ,tab3= st.tabs(["Chat", "Text Summarizer","Personal Expense Tracker"])
with tab1:

    user_input = st.text_area("Your Message ", placeholder="Type your message here...", key="input")
    if st.button("Send", key="send"):
        if user_input and llm:
            st.session_state.history.append(HumanMessage(content=user_input))
            with st.spinner("AI is typing..."):
                response = llm.invoke(st.session_state.history)
            st.session_state.history.append(AIMessage(content=response.content))
        else:
            st.warning("Please enter a message and ensure the model is selected.")
    for msg in st.session_state.history:
        if isinstance(msg, HumanMessage):
            st.chat_message("user").markdown(f"**You:** {msg.content}")
        elif isinstance(msg, AIMessage):
            st.chat_message("assistant").markdown(f"**AI:** {msg.content}")

with tab2:


    t1,t2,t3=st.tabs(["Summarize Text","Summarize PDF","Summarize Webpage"])
    with t1:
        st.header("Text Summarizer")
        text_input = st.text_area("Enter text to summarize", placeholder="Type or paste text here...", key="text_input")
        summary_size=st.slider("Select summary size", min_value=100, max_value=2000, value=500, step=10)
        if st.button("Summarize", key="summarize"):
            text_summarizer(text_input, summary_size, llm, system_prompt)
    with t2:
        st.header("PDF Summarizer")
        uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
        summary_size_pdf=st.slider("Select summary size", min_value=100, max_value=2000, value=500, step=10,key="pdf")
        if st.button("Summarize PDF", key="summarize_pdf"):
            if uploaded_file is not None:
                pdf_text= read_pdf(uploaded_file)
                if pdf_text:
                    text_summarizer(pdf_text, summary_size_pdf, llm, system_prompt)
                else:
                    st.warning("The uploaded PDF contains no extractable text.")
    with t3:
        st.header("Webpage Summarizer")
        url = st.text_input("Enter the URL of the webpage to summarize", placeholder="abc.com", key="url_input")
        url.strip()
        summary_size_web=st.slider("Select summary size", min_value=100, max_value=2000, value=500, step=10,key="web")
        if st.button("Summarize Webpage", key="summarize_webpage"):
            webpage_summarizer(url, summary_size_web, llm, system_prompt)


with tab3:
    st.header("Personal Expense Tracker")
    if 'expenses' not in st.session_state:
        st.session_state.expenses = []
    if 'history_t3' not in st.session_state:
        st.session_state.history_t3 = []

    col1, col2, col3 = st.columns(3)
    with col1:
        date = st.date_input("Date", value=datetime.date(datetime.now()))
    with col2:
        description = st.text_input("Description")
    with col3:
        amount = st.number_input("Amount", min_value=0)

    if st.button("Add Expense"):
        if description and amount > 0:
            st.session_state.expenses.append({
                "date": date,
                "description": description,
                "amount": amount
            })
            st.success("Expense added!")
        else:
            st.warning("Please provide a valid description and amount.")

    if st.session_state.expenses:
        df = pd.DataFrame(st.session_state.expenses)
        total_expense = df['amount'].sum()
        st.subheader("Expense History")
        st.dataframe(df)
        st.markdown(f"**Total Expense:** ${total_expense:.2f}")
        if st.button("Clear Expenses"):
            st.session_state.expenses = []
            st.success("All expenses cleared.")
        if st.button("Download Expenses as CSV"):
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name='expenses.csv',
                mime='text/csv',
            )
            st.success("Expenses downloaded successfully.")
    else:
        st.info("No expenses recorded yet.")
    system_prompt_expense="""You are a helpful assistant. Your task is to help the user analyze their personal expenses based on the provided data. You should provide insights into spending patterns, suggest ways to save money, and help categorize expenses. Always strive to understand the user's intent and provide relevant answers. If you don't know the answer, it's okay to admit it rather than providing incorrect information. Provide examples wherever possible to illustrate your points and the response should be strictly in markdown format. Provide tables and summaries wherever possible.: The expense data is as follows:
    {expense_data}"""
    st.session_state.history_t3.insert(0, system_prompt_expense)
    user_input_t3 = st.text_area("Ask about your expenses", placeholder="Type your message here...", key="input_t3")
    if st.button("Send", key="send_t3"):
        if user_input_t3 and llm:
            expense_data = pd.DataFrame(st.session_state.expenses).to_csv(index=False)
            st.session_state.history_t3.append(HumanMessage(content=user_input_t3+f"\n Here is the expense data:\n {expense_data}"))
            # expense_summary = expense_data.to_string()
            # st.session_state.history_t3.append(AIMessage(content=f"Here is the expense data summary:\n{expense_summary}"))
            # with st.spinner("AI is typing..."):
            response_t3 = llm.invoke(st.session_state.history_t3)
            st.session_state.history_t3.append(AIMessage(content=response_t3.content))
            st.chat_message("user").markdown(f"**You:** {user_input_t3}")
            st.chat_message("assistant").markdown(f"**AI:** {response_t3.content}")
        else:
            st.warning("Please enter a message and ensure the model is selected.")
    

