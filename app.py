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

# 3. Load Vector Database (Expanded Retrieval k=5 for All 849 Acts)
@st.cache_resource
def load_resources():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    return db.as_retriever(search_kwargs={"k": 5})

retriever = load_resources()
web_search_tool = DuckDuckGoSearchRun()

today_str = datetime.now().strftime("%A, %B %d, %Y")

# 4. System Prompt with Case Law & Precedents Integration
cross_verify_prompt = ChatPromptTemplate.from_messages([
    ("system", f"""Today's date is {today_str}.
You are an expert Indian Legal AI Assistant (NyayaAI) grounded in the complete body of Indian Statutory Law (849 Central Acts) and Case Law / Precedents from the Supreme Court and High Courts of India.

Your objective is to provide precise, authoritative, and factually grounded legal answers.

STRICT GROUNDING & RESPONSE RULES:
1. UNIVERSAL STATUTORY ACCURACY:
   - Identify and cite the EXACT Act and Section governing the user's query (e.g., Companies Act 2013, Arbitration Act 1996, Income Tax Act 1961, BNS 2023, Consumer Protection Act 2019, Hindu Marriage Act 1955, etc.).
   - NEVER assume an offence or civil matter falls under BNS or IPC if it is governed by another specific Act.

2. JUDGMENTS & CASE LAW PRECEDENTS:
   - If the user explicitly asks for judgments, case laws, or precedents, OR if interpreting the statute requires judicial clarification, include a dedicated section titled '### Relevant Judgments & Judicial Precedents'.
   - Format citations clearly: *Case Name v. Union of India / Opposing Party* (Year) [Supreme Court / High Court], along with a concise 1-2 sentence summary of the core ratio decidendi (legal holding).

3. ZERO HALLUCINATION & GROUNDING:
   - Base your response strictly on active, enacted Indian legislation and authentic judicial precedents. NEVER cite lapsed Bills, draft proposals, or non-existent cases/sections.
   - If a specific judgment or provision is not present in the provided statutory context or web search data, explicitly state that rather than inventing case names or section numbers.

4. FLEXIBLE PRESENTATION:
   - Use Markdown tables ONLY when comparing multiple statutory provisions, listing distinct penalties, or comparing old vs. new laws.
   - For general questions, case law summaries, or procedural advice, use clear headings, concise paragraphs, and bullet points.

5. PURE MARKDOWN FORMATTING (NO HTML/BR TAGS):
   - NEVER use HTML tags like `<br>`, `<b>`, `<i>`, or `<ul>` anywhere in the response.

6. NO TECHNICAL MECHANICS:
   - NEVER mention "ChromaDB", "Local Statutory Context", "Live Web Verification", "database", or internal RAG mechanics in your final answer.

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
st.caption("Intelligent Legal Assistant Grounded on 849 Indian Central Acts & Judicial Precedents")

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
        with st.spinner("Searching 849 Central Acts & retrieving relevant judgments..."):
            
            # Step 1: Retrieve statutory passages from ChromaDB
            docs = retriever.invoke(user_query)
            rag_context = format_docs(docs)

            # Step 2: Dynamic targeted web search for Judgments & Precedents
            web_context = "Web verification toggled off."
            if enable_web_verification:
                try:
                    # Detect if user specifically requested case laws or judgments
                    case_keywords = ["judgment", "judgement", "case law", "precedent", "supreme court", "high court", "ruling", "landmark case", "vs", "v."]
                    is_case_req = any(kw in user_query.lower() for kw in case_keywords)

                    if is_case_req:
                        verification_query = f"Supreme Court India landmark judgment precedent {user_query}"
                    else:
                        verification_query = f"Indian law statutory section landmark judgment {user_query}"

                    raw_web_res = web_search_tool.invoke(verification_query)
                    web_context = raw_web_res[:1500]  # Expanded to capture case citation details
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

        with st.expander("🔍 View Referenced Sources & Web Verification Data"):
            st.subheader("Statutory Context (Retrieved Chunks)")
            for i, doc in enumerate(docs):
                act = doc.metadata.get("act_name", "Legal Act")
                page = doc.metadata.get("page", "N/A")
                st.markdown(f"**Source {i+1}: {act} (Page {page})**")
                st.caption(doc.page_content[:250] + "...")
            
            if enable_web_verification:
                st.subheader("Live Verification & Judgment Snippets")
                st.caption(web_context)

    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=final_answer))