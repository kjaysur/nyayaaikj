import streamlit as st
import json
import re
from datetime import datetime
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun

# 1. Page Configuration
st.set_page_config(page_title="NyayaAI: Hybrid Legal Engine", page_icon="⚖️", layout="wide")

# 2. Sidebar Configuration
st.sidebar.title("⚖️ NyayaAI Settings")

enable_web_verification = st.sidebar.checkbox(
    "🔎 Double-Check with Live Web Search", 
    value=True,
    help="Cross-checks statutory database with live web search."
)

model_choice = st.sidebar.selectbox(
    "Select Groq Model",
    ["openai/gpt-oss-20b", "qwen/qwen3.8-27b", "openai/gpt-oss-120b"],
    index=1 # Recommend 70B for strict JSON extraction and hybrid synthesis
)

# API Key loaded directly from Streamlit Secrets
raw_key = st.secrets.get("GROQ_API_KEY", "")
groq_api_key = raw_key.strip().strip('"').strip("'")

# 3. Load Vector Database (CUDA locally / CPU fallback on Streamlit Cloud)
@st.cache_resource
def load_resources():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': device}
    )
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    return db.as_retriever(search_kwargs={"k": 15}) 

retriever = load_resources()
web_search_tool = DuckDuckGoSearchRun()
today_str = datetime.now().strftime("%A, %B %d, %Y")

# 4. PHASE 1: LLM Fact & Intent Extraction
def extract_facts(llm, user_query):
    extractor_prompt = ChatPromptTemplate.from_template("""
    You are a legal intent and data extraction bot. Analyze the user query and extract entities into a valid JSON object.
    Output ONLY a valid JSON object. Do not include markdown formatting or explanations.
    
    Query: {query}
    
    Schema:
    {{
        "intent_type": "criminal_analysis" | "civil_contract_drafting" | "other",
        "victim_age": int or null,
        "accused_age": int or null,
        "incident_date": "YYYY-MM-DD" or null,
        "alleged_crime": "string (e.g., 'murder', 'attempt to murder', 'drug trafficking', 'rape', 'theft', 'none')",
        "key_terms": "space separated statutory search keywords"
    }}
    """)
    try:
        chain = extractor_prompt | llm | StrOutputParser()
        raw_json = chain.invoke({"query": user_query})
        clean_json = re.sub(r'```json|```', '', raw_json).strip()
        return json.loads(clean_json)
    except Exception:
        return {"intent_type": "criminal_analysis", "victim_age": None, "accused_age": None, "incident_date": None, "alleged_crime": "other", "key_terms": user_query}

# 5. PHASE 2: Python Deterministic Rule Engine (Handles Civil vs Criminal Branching)
def apply_legal_rules(facts):
    mandatory_directives = []
    search_boosts = []
    
    intent = facts.get("intent_type", "criminal_analysis")

    # BRANCH A: Civil Contract Drafting (Rent Agreements, Deeds, etc.)
    if intent == "civil_contract_drafting":
        mandatory_directives.append("MANDATORY CONTRACT RULE: This is a civil contract/agreement drafting request. DO NOT cite criminal penal codes like BNS or IPC. Contracts are governed by civil law, the Indian Contract Act, and local state stamp/tenancy rules.")
        mandatory_directives.append("MANDATORY FORMAT: Generate a comprehensive, professional, ready-to-use legal agreement template using clean Markdown clauses and tables. Fill placeholders realistically based on local urban norms (e.g., Ahmedabad, Gujarat).")
        search_boosts.append("Leave and License Agreement Gujarat stamp duty residential flat")
        return mandatory_directives, " ".join(search_boosts)

    # BRANCH B: Criminal Law Analysis (Your Existing Robust Logic)
    alleged_crime = str(facts.get("alleged_crime", "")).lower()

    # Rule 1: Temporal Law Application (IPC vs BNS)
    is_legacy_law = False
    if facts.get("incident_date"):
        try:
            inc_date = datetime.strptime(facts["incident_date"], "%Y-%m-%d")
            if inc_date < datetime(2024, 7, 1):
                is_legacy_law = True
        except:
            pass

    if is_legacy_law:
        mandatory_directives.append("MANDATORY RULE: The incident occurred before July 1, 2024. You MUST cite the Indian Penal Code (IPC). DO NOT cite BNS.")
        search_boosts.append("IPC Indian Penal Code 1860")
    else:
        mandatory_directives.append("MANDATORY RULE: Incident is post-July 2024 (or date unstated). You MUST cite the Bharatiya Nyaya Sanhita (BNS). DO NOT cite the repealed IPC.")
        search_boosts.append("Bharatiya Nyaya Sanhita BNS 2023")

    # Rule 2: Base Law Routing
    if "attempt to murder" in alleged_crime:
        search_boosts.append("BNS Section 109 attempt to murder punishment")
    elif "murder" in alleged_crime:
        search_boosts.append("BNS Section 103 punishment for murder")
    elif "rape" in alleged_crime or "sexual assault" in alleged_crime:
        search_boosts.append("BNS Section 63 Section 64 general rape punishment")
    elif "theft" in alleged_crime:
        search_boosts.append("BNS Section 303 punishment for theft")

    # Rule 3: POCSO & Victim Age Math
    if ("sexual assault" in alleged_crime or "rape" in alleged_crime) and facts.get("victim_age") is not None:
        v_age = facts["victim_age"]
        search_boosts.append("POCSO Act 2012 penetrative sexual assault Special Court")
        
        mandatory_directives.append("MANDATORY RULE: Section 42 of the POCSO Act governs statutory overlap. If an offense falls under both BNS and POCSO, the court must apply the statute carrying the greater punishment. Trial MUST take place in a POCSO Special Court.")
        
        if v_age < 12:
            mandatory_directives.append("MANDATORY RULE: Victim is under 12. BNS Section 65(2) strictly applies.")
            mandatory_directives.append("MANDATORY RULE: Victim is under 12. POCSO Section 5(m) classifies the crime as 'Aggravated Penetrative Sexual Assault' SOLELY because the victim is under 12. Section 6 governs the penalty.")
            mandatory_directives.append("MANDATORY RULE: Under BNS 65(2) and POCSO Sec 6, the punishment is a MINIMUM of 20 years rigorous imprisonment, extending to natural life imprisonment, OR the death penalty. Death is a discretionary maximum, NOT a mandatory alternative.")
            search_boosts.append("Section 65(2) Section 5(m) Section 6 death penalty")
        elif v_age < 16:
            mandatory_directives.append("MANDATORY RULE: Victim is under 16. BNS Section 65(1) strictly applies.")
            mandatory_directives.append("MANDATORY RULE: Victim is under 16. POCSO Section 3 and 4 strictly apply.")
        elif v_age < 18:
            mandatory_directives.append("MANDATORY RULE: Victim is under 18. POCSO Act applies. Ensure age-appropriate BNS sections are cited.")

    # Rule 4: Dynamic Juvenile Justice
    if facts.get("accused_age") is not None and facts.get("accused_age") < 18:
        acc_age = facts.get("accused_age")
        search_boosts.append("Juvenile Justice Act 2015 Section 15 Section 18 Section 21 Childrens Court Place of Safety Special Home")

        if acc_age >= 16:
            mandatory_directives.append(
                "MANDATORY JUVENILE RULE: The accused is 16-18. Check the retrieved statutes for the alleged crime. "
                "IF the statute dictates a MINIMUM mandatory punishment of 7 years or more, it is a 'Heinous Offence'. "
                "Apply JJA Section 15 (JJB preliminary assessment) and Section 18(3) (can be transferred to Children's Court to be tried as an adult)."
            )
            mandatory_directives.append(
                "MANDATORY SENTENCING RULE (CHILDREN'S COURT): If convicted as an adult under JJA Section 21, the juvenile CAN face custodial imprisonment "
                "(Place of Safety until 21, then adult prison). The judge retains full discretion up to life imprisonment with possibility of release. "
                "Do NOT cap the punishment at the statutory minimum. Only death and life without release are prohibited."
            )
            mandatory_directives.append(
                "MANDATORY NON-HEINOUS RULE: IF the statute has NO minimum, or a minimum under 7 years (even if the maximum is 7+ years, like Attempt to Murder), "
                "it is classified as a 'Serious Offence' under the JJA 2021 Amendment (codifying Shilpa Mittal). "
                "Adult trial is STRICTLY PROHIBITED. The JJB retains exclusive jurisdiction."
            )
            mandatory_directives.append(
                "MANDATORY JJB SENTENCING CEILING: Under Section 18(1)(g) of the JJA, the MAXIMUM custodial order the JJB can pass is 3 years in a Special Home. "
                "You are STRICTLY FORBIDDEN from applying the BNS maximum penalty (e.g., 10 years) or transferring the juvenile to an adult prison."
            )
        else:
            mandatory_directives.append(
                "MANDATORY RULE: The accused is under 16. Regardless of the crime's severity, the Juvenile Justice Board (JJB) has exclusive jurisdiction. "
                "Adult courts cannot try them. The maximum custodial stay is 3 years in a Special Home under Section 18(1)(g)."
            )

    return mandatory_directives, " ".join(search_boosts)

# 6. PHASE 3: Synthesis Prompt (Unified for Criminal and Civil Intent)
synthesis_prompt = ChatPromptTemplate.from_template("""
Today's date is {today_str}.
You are NyayaAI, an Indian Legal AI grounded strictly in Central and State Acts.

You must format the final legal response using the retrieved context, verification data, AND the mandatory rules below.

--- MANDATORY PYTHON-GENERATED RULES ---
{mandatory_rules}
(If the above says 'None', rely purely on the retrieved statutory context and your universal operational rules below).

--- UNIVERSAL OPERATIONAL RULES ---
1. GROUNDING: Base all facts, penalties, and citations strictly on the provided context. Cite source headers (e.g., [SOURCE 1]).
2. INTENT SEPARATION: If the intent is civil contract drafting, output a complete, professional legal agreement template (e.g., Leave & License Agreement) without referencing criminal codes. If criminal analysis, apply rigorous penal rules.
3. SPECIFICITY: Always match the facts of the query to the narrowest/most specific provision available in the context.
4. EXACT SECTION MATCHING (ANTI-OVERLAP RULE): Do not confuse general crimes with special categories unless explicitly stated.
5. PENALTIES: List exact minimums, maximums, and fine structures. Do not invent alternatives.
6. FORMATTING: Use clean Markdown. NEVER use HTML tags like `<br>`. Use tables for side-by-side law comparisons or structured agreement clauses.
7. JOINT TRIALS: If a scenario involves both adult and juvenile co-accused, explicitly state that Section 23 of the Juvenile Justice Act prohibits joint trials.

--- RETRIEVED STATUTORY CONTEXT ---
{rag_context}

--- VERIFICATION DATA ---
{web_context}

User Query: {input}

Conclude your response EXACTLY with:
'Disclaimer: This response is for educational purposes and does not constitute formal legal advice.'
""")

def format_docs(docs):
    formatted = []
    for i, doc in enumerate(docs):
        act_name = doc.metadata.get('act_name') or doc.metadata.get('source') or 'Indian Statute'
        act_name = str(act_name).replace('./pdfs\\', '').replace('./pdfs/', '').replace('.txt', '').replace('_', ' ')
        page = doc.metadata.get('page', 'N/A')
        header = f"[SOURCE {i+1}: {act_name} | Page {page}]"
        formatted.append(f"{header}\n{doc.page_content.strip()}")
    return "\n\n".join(formatted)

def sanitize_web_context(raw_res):
    legal_indicators = ["section", "act", "court", "judgment", "agreement", "license", "stamp", "duty", "gujarat", "ahmedabad", "punishment"]
    lines = raw_res.split(". ")
    filtered = [line for line in lines if any(ind in line.lower() for ind in legal_indicators)]
    return ". ".join(filtered)[:1200] if filtered else "No authoritative legal web snippets found."

# 7. UI Presentation
st.title("⚖️ NyayaAI: Legal Agent by KJ")
st.caption("Hybrid Deterministic AI Grounded on 849 Central Acts")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 8. Execution Loop
if user_query := st.chat_input("Ask a legal question or request document drafting..."):
    if not groq_api_key or not groq_api_key.startswith("gsk_"):
        st.error("⚠️ Missing or invalid Groq API Key in Streamlit Secrets.")
        st.stop()

    llm = ChatGroq(groq_api_key=groq_api_key, model=model_choice, temperature=0.0)
    
    with st.chat_message("user"):
        st.write(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        with st.spinner("Step 1: Extracting Intent & Applying Python Legal Logic..."):
            
            # Phase 1: LLM extracts JSON intent & facts
            extracted_facts = extract_facts(llm, user_query)
            
            # Phase 2: Python applies intent branching & strict legal math
            mandatory_rules_list, search_boosts = apply_legal_rules(extracted_facts)
            mandatory_rules_text = "\n".join(f"- {rule}" for rule in mandatory_rules_list)
            
            # Execute ChromaDB Retrieval
            optimized_query = f"{user_query} {extracted_facts.get('key_terms', '')} {search_boosts}"
            docs = retriever.invoke(optimized_query)
            rag_context = format_docs(docs)

            # Web Verification
            web_context = "Web verification toggled off."
            if enable_web_verification:
                try:
                    case_keywords = ["judgment", "judgement", "case law", "precedent", "supreme court", "high court"]
                    if extracted_facts.get("intent_type") == "civil_contract_drafting":
                        query_type = "Gujarat leave and license rent agreement rules"
                    else:
                        query_type = "Supreme Court landmark judgment" if any(kw in user_query.lower() for kw in case_keywords) else "statutory section"
                    
                    raw_web_res = web_search_tool.invoke(f"site:indiankanoon.org OR site:egazette.gov.in OR site:gujarat.gov.in {query_type} {user_query}")
                    web_context = sanitize_web_context(raw_web_res)
                except Exception as err:
                    web_context = f"Live verification search unavailable: {err}"

        with st.spinner("Step 2: Synthesizing Final Output..."):
            # Phase 3: Final LLM Generation locked by Python Rules
            chain = synthesis_prompt | llm | StrOutputParser()
            try:
                final_answer = chain.invoke({
                    "today_str": today_str,
                    "mandatory_rules": mandatory_rules_text if mandatory_rules_text else "None.",
                    "rag_context": rag_context,
                    "web_context": web_context,
                    "input": user_query
                })
            except Exception as err:
                st.error(f"Groq API Error: {err}")
                st.stop()

        # Print Final Output
        st.write(final_answer)

        # Transparency UI for Debugging & Trust
        with st.expander("⚙️ View Middleware Engine Data (Zero-Hallucination Pipeline)"):
            st.markdown("**1. Extracted JSON Intent & Facts:**")
            st.json(extracted_facts)
            st.markdown("**2. Hardcoded Python Directives Applied:**")
            if mandatory_rules_list:
                for rule in mandatory_rules_list:
                    st.success(rule)
            else:
                st.info("No hardcoded thresholds triggered. Relying on standard vector retrieval.")
            st.markdown("**3. Optimized ChromaDB Search Query:**")
            st.code(optimized_query)
            
        with st.expander("🔍 View Raw Retrieved Context & Verification Data"):
            st.subheader("Statutory Context (Retrieved Chunks)")
            for i, doc in enumerate(docs):
                act = doc.metadata.get("act_name") or doc.metadata.get("source") or "Legal Act"
                act = str(act).replace('./pdfs\\', '').replace('./pdfs/', '').replace('.txt', '').replace('_', ' ')
                st.markdown(f"**Chunk {i+1}: {act}**")
                st.caption(doc.page_content[:300] + "...")
            if enable_web_verification:
                st.subheader("Sanitized Verification Snippet")
                st.caption(web_context)

    st.session_state.chat_history.append({"role": "assistant", "content": final_answer})