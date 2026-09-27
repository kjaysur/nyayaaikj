import streamlit as st
from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun

# 1. Page Configuration
st.set_page_config(page_title="NyayaAI: Legal Agent by KJ", page_icon="⚖️", layout="wide")

# 2. Sidebar Configuration
st.sidebar.title("⚖️ NyayaAI Settings")
enable_web_search = st.sidebar.checkbox("🌐 Enable Web Search Fallback", value=True)

# Active Groq Models
model_choice = st.sidebar.selectbox(
    "Select Groq Model",
    ["openai/gpt-oss-20b", "qwen/qwen3.8-27b", "openai/gpt-oss-120b"],
    index=0
)

# API Key loaded directly from Streamlit Secrets (completely hidden from UI)
raw_key = st.secrets.get("GROQ_API_KEY", "")
groq_api_key = raw_key.strip().strip('"').strip("'")

# 3. Load Vector Database & Search Tool
@st.cache_resource
def load_resources():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    return db.as_retriever(search_kwargs={"k": 2})

retriever = load_resources()
web_search_tool = DuckDuckGoSearchRun()

today_str = datetime.now().strftime("%A, %B %d, %Y")

# Unified Single-Stage Legal Verification Prompt
unified_prompt = ChatPromptTemplate.from_messages([
    ("system", f"""Today's date is {today_str}.
You are an expert Indian Legal AI Assistant and Auditor (NyayaAI).
Answer legal queries accurately based on the provided statutory context and strict statutory realities:

1. OFFENCE DATE CUTOFF (July 1, 2024): Offences committed before July 1, 2024 must be charged under IPC (Article 20(1) Ex Post Facto protection), NOT BNS.
2. PROCEDURAL LAW: Investigations or FIRs registered on or after July 1, 2024 follow BNSS (Section 173 for FIRs).
3. Generate a clear, structured final output. Always conclude with:
'Disclaimer: This response is for educational purposes and does not constitute formal legal advice.'

Statutory Context:
{{context}}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

def format_docs(docs):
    return "\n\n".join(f"[{doc.metadata.get('act_name', 'Act')} - Page {doc.metadata.get('page', 'N/A')}]: {doc.page_content}" for doc in docs)

# 4. UI Header & Presentation
st.title("⚖️ NyayaAI: Legal Agent by KJ")
st.caption("Intelligent Legal Assistant Grounded on Indian Law (Constitution, BNS, BNSS, BSA)")

# Initialize Memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Render Past Chat
for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(msg.content)

# 5. Chat Execution Loop
if user_query := st.chat_input("Ask a legal question or scenario..."):
    if not groq_api_key or not groq_api_key.startswith("gsk_"):
        st.error("⚠️ Missing or invalid Groq API Key in Streamlit Secrets. Please verify `GROQ_API_KEY` is set under App Settings.")
        st.stop()

    try:
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model=model_choice,
            temperature=0.1,
            max_tokens=1024
        )
        legal_chain = unified_prompt | llm | StrOutputParser()
    except Exception as e:
        st.error(f"Failed to initialize Groq client: {e}")
        st.stop()

    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing legal query and verifying provisions..."):
            docs = retriever.invoke(user_query)
            context_str = format_docs(docs)

            is_latest_query = any(w in user_query.lower() for w in ["latest", "recent", "2025", "2026", "news", "supreme court"])
            if enable_web_search and (is_latest_query or len(context_str.strip()) < 100):
                try:
                    web_results = web_search_tool.invoke(f"Indian Law {user_query}")
                    context_str = f"--- LIVE WEB RESULTS ---\n{web_results}\n\n--- LOCAL STATUTE CONTEXT ---\n{context_str}"
                except Exception:
                    pass

            try:
                final_answer = legal_chain.invoke({
                    "context": context_str,
                    "chat_history": st.session_state.chat_history,
                    "input": user_query
                })
            except Exception as err:
                st.error(f"Groq API Error: {err}")
                st.info("Tip: Try switching models in the sidebar dropdown.")
                st.stop()

        st.write(final_answer)

        with st.expander("🔍 View Referenced Legal Sources"):
            for i, doc in enumerate(docs):
                act = doc.metadata.get("act_name", "Legal Act")
                page = doc.metadata.get("page", "N/A")
                st.markdown(f"**Source {i+1}: {act} (Page {page})**")
                st.caption(doc.page_content[:250] + "...")

    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=final_answer))