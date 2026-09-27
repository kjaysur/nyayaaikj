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

# Verification Feature Toggle
enable_web_verification = st.sidebar.checkbox(
    "🔎 Double-Check with Live Web Search", 
    value=True,
    help="Cross-checks ChromaDB results with live web search before generating the final legal response."
)

# Active Groq Models
model_choice = st.sidebar.selectbox(
    "Select Groq Model",
    ["openai/gpt-oss-20b", "qwen/qwen3.8-27b", "openai/gpt-oss-120b"],
    index=0
)

# API Key loaded directly from Streamlit Secrets (hidden from UI)
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

# Unified Cross-Verification System Prompt (Cleaned for End-Users)
cross_verify_prompt = ChatPromptTemplate.from_messages([
    ("system", f"""Today's date is {today_str}.
You are an expert Indian Legal AI Assistant (NyayaAI).

Your task is to answer legal queries by synthesizing statutory data and live verification data into a clear, concise, and professional legal summary.

STRICT OUTPUT & FORMATTING RULES:
1. NO TECHNICAL JARGON OR INTERNAL MECHANICS:
   - NEVER mention "ChromaDB", "Local Statutory Context", "Live Web Verification", "Web Search", "database", or "Cross-verification" in your final response text.
   - Do NOT write sections detailing where the data was retrieved from. Synthesize everything cleanly into a single authoritative explanation.
2. PURE MARKDOWN FORMATTING (NO HTML/BR TAGS):
   - NEVER use HTML tags such as `<br>`, `<b>`, `<i>`, or `<ul>` anywhere in the response or table cells.
   - Separate multiple items using simple commas or standard Markdown bullet points (`- `).
3. STRICT SECTION NUMBERING:
   - Ensure exact BNS, BNSS, or BSA section numbers are stated (e.g., Murder = Section 103 BNS; Theft = Section 303 BNS; Snatching = Section 302 BNS).
   - NEVER default to old IPC section numbers when BNS applies.
4. OFFENCE DATE CUTOFF (July 1, 2024):
   - Offences before July 1, 2024 -> Charged under IPC (Article 20(1) Ex Post Facto protection).
   - Offences on/after July 1, 2024 -> Charged under BNS / BNSS.

Always conclude with:
'Disclaimer: This response is for educational purposes and does not constitute formal legal advice.'

--- STATUTORY CONTEXT ---
{{rag_context}}

--- VERIFICATION DATA ---
{{web_context}}"""),
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
        st.error("⚠️ Missing or invalid Groq API Key in Streamlit Secrets.")
        st.stop()

    try:
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model=model_choice,
            temperature=0.1,
            max_tokens=1024
        )
        legal_chain = cross_verify_prompt | llm | StrOutputParser()
    except Exception as e:
        st.error(f"Failed to initialize Groq client: {e}")
        st.stop()

    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Step 1/2: Searching ChromaDB... Step 2/2: Double-checking with live web search..."):
            
            # Step 1: Retrieve local statutory passages from ChromaDB
            docs = retriever.invoke(user_query)
            rag_context = format_docs(docs)

            # Step 2: Double-check with targeted live web search
            web_context = "Web verification toggled off."
            if enable_web_verification:
                try:
                    # Targeted search query designed for verification
                    verification_query = f"Bharatiya Nyaya Sanhita section punishment {user_query}"
                    raw_web_res = web_search_tool.invoke(verification_query)
                    # Truncate to 1,200 chars to protect token budget
                    web_context = raw_web_res[:1200]
                except Exception as err:
                    web_context = f"Live verification search unavailable: {err}"

            # Step 3: Execute Cross-Verification Chain in a single LLM call
            try:
                final_answer = legal_chain.invoke({
                    "rag_context": rag_context,
                    "web_context": web_context,
                    "chat_history": st.session_state.chat_history,
                    "input": user_query
                })
            except Exception as err:
                st.error(f"Groq API Error: {err}")
                st.info("Tip: Try switching models in the sidebar dropdown.")
                st.stop()

        st.write(final_answer)

        # Expanders for transparent debugging
        with st.expander("🔍 View Referenced Sources & Verification Data"):
            st.subheader("Local Statutory Context (ChromaDB)")
            for i, doc in enumerate(docs):
                act = doc.metadata.get("act_name", "Legal Act")
                page = doc.metadata.get("page", "N/A")
                st.markdown(f"**Source {i+1}: {act} (Page {page})**")
                st.caption(doc.page_content[:250] + "...")
            
            if enable_web_verification:
                st.subheader("Live Web Verification Snippet")
                st.caption(web_context)

    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=final_answer))