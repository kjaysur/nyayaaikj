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

enable_web_verification = st.sidebar.checkbox(
    "🔎 Double-Check with Live Web Search & Case Law", 
    value=True,
    help="Cross-checks statutory database with live web search and retrieves Supreme Court / High Court case laws."
)

model_choice = st.sidebar.selectbox(
    "Select Groq Model",
    ["openai/gpt-oss-20b", "qwen/qwen3.8-27b", "openai/gpt-oss-120b"],
    index=0
)

# API Key loaded directly from Streamlit Secrets
raw_key = st.secrets.get("GROQ_API_KEY", "")
groq_api_key = raw_key.strip().strip('"').strip("'")

# 3. Load Vector Database (Expanded Retrieval k=5)
@st.cache_resource
def load_resources():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    return db.as_retriever(search_kwargs={"k": 5})

retriever = load_resources()
web_search_tool = DuckDuckGoSearchRun()

today_str = datetime.now().strftime("%A, %B %d, %Y")

# 4. Anti-Hallucination System Prompt
cross_verify_prompt = ChatPromptTemplate.from_messages([
    ("system", f"""Today's date is {today_str}.
You are an expert Indian Legal AI Assistant (NyayaAI) grounded strictly in the body of Indian Statutory Law (849 Central Acts) and authentic Case Law from the Supreme Court and High Courts of India.

STRICT ZERO-HALLUCINATION & GROUNDING RULES:
1. STRICT CONTEXT GROUNDING:
   - Base your answer ONLY on active, enacted Indian legislation and authentic verified precedents provided in the context below.
   - If a specific provision or judgment is NOT present in the statutory context or live web verification data, explicitly state: 'The exact statutory section/precedent for this scenario was not found in the verified legal sources.' NEVER invent section numbers or case titles.

2. UNIVERSAL STATUTORY ACCURACY:
   - Identify and cite the EXACT Act and Section governing the query (e.g., Companies Act 2013, BNS 2023, Consumer Protection Act 2019, Hindu Marriage Act 1955, etc.).
   - NEVER classify civil, corporate, or personal laws under BNS or IPC.

3. CASE LAW CITATION RULE:
   - ONLY cite case laws if an authentic case name (e.g., 'X v. Y') appears directly in the VERIFICATION & CASE LAW DATA snippet.
   - Format citations clearly: *Case Name v. Opposing Party* (Year) [Court], with a concise 1-2 sentence legal ratio. NEVER cite non-existent cases, lapsed Bills, or draft proposals.

4. PURE MARKDOWN FORMATTING (NO HTML/BR TAGS):
   - NEVER use HTML tags like `<br>`, `<b>`, `<i>`, or `<ul>` anywhere in your output.
   - Use standard Markdown bullets (`- `) or simple commas.

5. NO TECHNICAL MECHANICS:
   - NEVER mention "ChromaDB", "Local Statutory Context", "Live Web Verification", "database", or RAG mechanics in your final answer.

Always conclude with:
'Disclaimer: This response is for educational purposes and does not constitute formal legal advice.'

--- STATUTORY CONTEXT (849 Central Acts) ---
{{rag_context}}

--- VERIFICATION & CASE LAW DATA ---
{{web_context}}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

def format_docs(docs):
    return "\n\n".join(f"[{doc.metadata.get('act_name', 'Act')} - Page {doc.metadata.get('page', 'N/A')}]: {doc.page_content}" for doc in docs)

# 5. UI Header & Presentation
st.title("⚖️ NyayaAI: Legal Agent by KJ")
st.caption("Intelligent Legal Assistant Grounded on 849 Indian Central Acts & Verified Judicial Precedents")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(msg.content)

# 6. Chat Execution Loop
if user_query := st.chat_input("Ask any legal question or request judgments/case laws..."):
    if not groq_api_key or not groq_api_key.startswith("gsk_"):
        st.error("⚠️ Missing or invalid Groq API Key in Streamlit Secrets.")
        st.stop()

    try:
        # Temperature set to 0.0 for zero creativity / strict factual determinism
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model=model_choice,
            temperature=0.0,
            max_tokens=1024
        )
        legal_chain = cross_verify_prompt | llm | StrOutputParser()
    except Exception as e:
        st.error(f"Failed to initialize Groq client: {e}")
        st.stop()

    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Searching 849 Central Acts & verifying against authoritative legal databases..."):
            
            # Step 1: Retrieve statutory passages across all 849 Acts
            docs = retriever.invoke(user_query)
            rag_context = format_docs(docs)

            # Step 2: Domain-Restricted Web Verification (Domain filtering to block web noise)
            web_context = "Web verification toggled off."
            if enable_web_verification:
                try:
                    case_keywords = ["judgment", "judgement", "case law", "precedent", "supreme court", "high court", "ruling", "landmark case", "vs", "v."]
                    is_case_req = any(kw in user_query.lower() for kw in case_keywords)

                    if is_case_req:
                        # Target Indian Kanoon, SC Observer, and Supreme Court Official site
                        verification_query = f"site:indiankanoon.org OR site:scobserver.in OR site:main.sci.gov.in landmark judgment {user_query}"
                    else:
                        # Target Gazette of India and Indian Kanoon
                        verification_query = f"site:indiankanoon.org OR site:egazette.gov.in statutory section {user_query}"

                    raw_web_res = web_search_tool.invoke(verification_query)
                    web_context = raw_web_res[:1500]
                except Exception as err:
                    web_context = f"Live verification search unavailable: {err}"

            # Step 3: Execute Chain
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

        with st.expander("🔍 View Referenced Sources & Verification Data"):
            st.subheader("Statutory Context (Retrieved Chunks)")
            for i, doc in enumerate(docs):
                act = doc.metadata.get("act_name", "Legal Act")
                page = doc.metadata.get("page", "N/A")
                st.markdown(f"**Source {i+1}: {act} (Page {page})**")
                st.caption(doc.page_content[:250] + "...")
            
            if enable_web_verification:
                st.subheader("Live Verification Snippets (Domain Filtered)")
                st.caption(web_context)

    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=final_answer))