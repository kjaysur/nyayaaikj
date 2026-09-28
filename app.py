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

raw_key = st.secrets.get("GROQ_API_KEY", "")
groq_api_key = raw_key.strip().strip('"').strip("'")

# 3. Load Vector Database (k=6 Retrieval Depth)
@st.cache_resource
def load_resources():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    return db.as_retriever(search_kwargs={"k": 6})

retriever = load_resources()
web_search_tool = DuckDuckGoSearchRun()

today_str = datetime.now().strftime("%A, %B %d, %Y")

# 4. Pure Universal Extraction Prompt (Zero Specific Law Hardcoding)
cross_verify_prompt = ChatPromptTemplate.from_messages([
    ("system", f"""Today's date is {today_str}.
You are an expert Indian Legal AI Assistant (NyayaAI) operating under a STRICT ZERO-HALLUCINATION POLICY.

CORE ARCHITECTURAL RULE:
You are an extraction and synthesis engine. You must answer questions using ONLY the facts explicitly stated in the provided STATUTORY CONTEXT and VERIFICATION DATA. You are strictly forbidden from using internal pre-training memory to invent, guess, or extrapolate section numbers, offences, penalties, or case names.

STRICT OPERATIONAL DIRECTIVES:
1. MANDATORY SOURCE CITATIONS:
   - Every legal claim, section, or penalty stated MUST be directly backed by the provided STATUTORY CONTEXT or VERIFICATION DATA.
   - Attach inline source references matching the context headers, e.g., '[Source: Act Name, Page X]'.

2. HONEST FALLBACK ON MISSING DATA:
   - If the exact section number or legal detail for a query is NOT present in the retrieved STATUTORY CONTEXT or VERIFICATION DATA, state:
     "The exact statutory section or provision for this scenario is not present in the retrieved database context."
   - NEVER guess single-digit section numbers, invent section mappings, or combine old/new statutes from memory.

3. DYNAMIC ACT IDENTIFICATION:
   - Cite the exact Act name as present in the source chunks (e.g., Companies Act, Hindu Marriage Act, BNS, IPC, Income Tax Act). Never force criminal statutes (BNS/IPC) onto civil or corporate queries.

4. PURE MARKDOWN FORMATTING:
   - Use Markdown tables ONLY when comparing multiple provisions or penalties.
   - NEVER use HTML tags like `<br>`, `<b>`, `<i>`, or `<ul>`.

5. NO TECHNICAL MECHANICS:
   - Do NOT mention "ChromaDB", "Vector Database", "RAG", "Retrieved Chunks", or internal code mechanics.

Always conclude with:
'Disclaimer: This response is for educational purposes and does not constitute formal legal advice.'

--- STATUTORY CONTEXT (Retrieved Documents) ---
{{rag_context}}

--- VERIFICATION & CASE LAW DATA ---
{{web_context}}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

def format_docs(docs):
    formatted = []
    for doc in docs:
        act = doc.metadata.get('act_name', 'Indian Statute')
        page = doc.metadata.get('page', 'N/A')
        content = doc.page_content.strip()
        formatted.append(f"--- DOCUMENT CHUNK [{act} - Page {page}] ---\n{content}")
    return "\n\n".join(formatted)

def sanitize_web_context(raw_res):
    # Filter out non-legal web noise
    legal_indicators = ["section", "act", "court", "judgment", "sanhita", "code", "punishment", "held", "vs", "v."]
    lines = raw_res.split(". ")
    filtered_lines = [line for line in lines if any(ind in line.lower() for ind in legal_indicators)]
    if filtered_lines:
        return ". ".join(filtered_lines)[:1500]
    return "No authoritative legal web snippets found."

# 5. UI Header & Presentation
st.title("⚖️ NyayaAI: Legal Agent by KJ")
st.caption("Factual Legal Assistant Grounded on 849 Indian Central Acts")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(msg.content)

# 6. Chat Execution Loop
if user_query := st.chat_input("Ask any legal question..."):
    if not groq_api_key or not groq_api_key.startswith("gsk_"):
        st.error("⚠️ Missing or invalid Groq API Key in Streamlit Secrets.")
        st.stop()

    try:
        # Temperature 0.0 guarantees maximum factual determinism
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
        with st.spinner("Searching statutory database & verifying facts..."):
            
            # Step 1: Retrieve statutory passages from ChromaDB
            docs = retriever.invoke(user_query)
            rag_context = format_docs(docs)

            # Step 2: Web verification with domain filtering & noise reduction
            web_context = "Web verification toggled off."
            if enable_web_verification:
                try:
                    case_keywords = ["judgment", "judgement", "case law", "precedent", "supreme court", "high court"]
                    is_case_req = any(kw in user_query.lower() for kw in case_keywords)

                    if is_case_req:
                        verification_query = f"site:indiankanoon.org OR site:scobserver.in Supreme Court landmark judgment {user_query}"
                    else:
                        verification_query = f"site:indiankanoon.org OR site:egazette.gov.in statutory section {user_query}"

                    raw_web_res = web_search_tool.invoke(verification_query)
                    web_context = sanitize_web_context(raw_web_res)
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

        with st.expander("🔍 View Raw Retrieved Context (Source Grounding)"):
            st.subheader("Statutory Context (ChromaDB Chunks)")
            for i, doc in enumerate(docs):
                act = doc.metadata.get("act_name", "Legal Act")
                page = doc.metadata.get("page", "N/A")
                st.markdown(f"**Chunk {i+1}: {act} (Page {page})**")
                st.caption(doc.page_content[:300] + "...")
            
            if enable_web_verification:
                st.subheader("Sanitized Web Verification Snippet")
                st.caption(web_context)

    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=final_answer))