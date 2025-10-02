
# 🤖 Tiny AI-Powered App: Q&A Bot with Streamlit
## Video Link : [TinyAI](https://www.youtube.com/watch?v=3bHTJk03FO0)
## 🎯 Project Overview
This repository contains a simple, AI-powered application developed as a solution to the intern assignment.  
The implemented project is an **AI Q&A Bot**.

The core function is to take a natural language question from a user and generate a relevant, coherent answer using a large language model API.  
To go beyond the minimum requirement, a simple, interactive **web user interface** was built using **Streamlit**.

---

## 🛠️ Technology Stack
- **Language:** Python 3.x  
- **Web Framework:** [Streamlit](https://streamlit.io/)  
- **AI Backend:** [OpenAI API](https://platform.openai.com/) (requires an API Key)  
- **Environment Management:** `venv` and `pip`  

---

## 🚀 Getting Started

Follow these steps to set up and run the application locally.

### 1. Prerequisites
- Python **3.8+** installed on your machine.  
- An **OpenAI API key**.

---

### 2. Clone the Repository
```bash
# Clone the repository to your local machine
git clone [YOUR-REPO-URL-HERE]
cd tiny-ai-app
````

---

### 3. Environment Setup

It’s recommended to use a virtual environment to manage dependencies.

```bash
# Create and activate the virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt


### 4. API Key Configuration

The application requires an API key for the LLM service.

1. Create a file named `.env` in the **root directory** of the project.
2. Add your API key to this file. Example:

   ```env
   OPENAI_API_KEY="sk-************************************"
   ```

---

### 5. Run the Application

Launch the Streamlit app from your terminal:

```bash
streamlit run app.py
```

The app will automatically open in your default browser (usually at **[http://localhost:8501](http://localhost:8501)**).

---

## 🧐 Development Journey & Documentation

### 1. Project Selection and Initial Design

* **Choice:** I chose the AI Q&A Bot because it directly demonstrates AI capability.
* **Initial Structure:** Started with a command-line prototype (`cli_prototype.py`) to validate API connectivity before building the UI.

### 2. Implementing the Streamlit UI

* **Why Streamlit?** Simple, fast, and perfect for data-driven apps.
* **Enhancements:**

  * Replaced `input()`/`print()` with `st.text_input` and `st.write`.
  * Added `st.spinner` for real-time loading feedback.
  * Used markdown formatting for clean output display.

### 3. Challenges & Solutions

| **Challenge**           | **Attempted Solution / Search**                                                   | **Final Solution**                                               |
| ----------------------- | --------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| Secure API Key Handling | Hardcoded key at first; searched *"python use environment variables for api key"* | Used `python-dotenv` to load keys from `.env` file               |
| Real-Time Feedback      | App froze during API call; searched *"streamlit show loading message"*            | Implemented `st.spinner('Thinking...')`                          |
| Error Handling          | API calls failed sometimes                                                        | Wrapped calls in `try/except` with `st.error` for clear messages |

### 4. Code Quality & Organization

* Functions separated for **UI** and **API logic**.
* Included **requirements.txt** and **clear setup instructions**.

---

## 💡 Future Enhancements

If continued, possible improvements include:

* **Deployment:** Deploy to [Hugging Face Spaces](https://huggingface.co/spaces), [Render](https://render.com/), or Streamlit Cloud.
* **Conversation History:** Use Streamlit `session_state` to store history and allow contextual Q&A.
* **Model Configuration:** Add sidebar options for temperature, max tokens, and model selection.

---

## 📂 Repository Structure (Example)

```
tiny-ai-app/
│── app.py               # Main Streamlit app
│── requirements.txt     # Dependencies
│── .env.example         # Example env file (without secrets)
│── README.md            # Project documentation
```

---

## 🙌 Acknowledgements

* [Streamlit](https://streamlit.io/) for the easy-to-build UI.
* [OpenAI](https://platform.openai.com/) for the LLM API.
* Inspiration from community tutorials and official docs.

---


