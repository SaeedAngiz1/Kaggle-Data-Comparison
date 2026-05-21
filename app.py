import html
import streamlit as st
import pandas as st_pandas
import pandas as pd
import numpy as np
import os
import json
import requests
from thefuzz import fuzz
from thefuzz import process
import shutil
import plotly.graph_objects as go
import base64
import io
import PyPDF2
import docx
import urllib.parse
import socket
import ipaddress

def is_safe_url(url):
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ('http', 'https'):
            return False
        hostname = parsed.hostname
        if not hostname:
            return False

        ip_info = socket.getaddrinfo(hostname, None)
        for info in ip_info:
            ip_addr = info[4][0]
            ip = ipaddress.ip_address(ip_addr)
            if ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_unspecified:
                return False
        return True
    except Exception:
        return False

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Comparative Analysis with Kaggle",
    page_icon="assets/kaggle-icon.png",
    layout="wide"
)

# --- INJECT BACKGROUND ---
@st.cache_data
def add_bg_from_local(image_file):
    try:
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url(data:image/png;base64,{encoded_string.decode()});
                background-size: cover;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            /* Optional: Add a slight dark overlay to ensure text is readable */
            .stApp::before {{
                content: "";
                position: absolute;
                top: 0; left: 0; right: 0; bottom: 0;
                background-color: rgba(0,0,0,0.4);
                z-index: -1;
            }}
            /* Make info/alert boxes less transparent for better readability */
            [data-testid="stAlert"] {{
                background-color: rgba(28, 131, 225, 0.9) !important;
                color: white !important;
                border: none !important;
            }}
            [data-testid="stAlert"] p {{
                color: white !important;
                font-weight: 500;
            }}
            [data-testid="stAlert"] svg {{
                fill: white !important;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        pass

add_bg_from_local('assets/kaggle-background.png')

# --- INITIALIZATION & UTILS ---
@st.cache_resource
def init_kaggle(username, key):
    os.environ['KAGGLE_USERNAME'] = username
    os.environ['KAGGLE_KEY'] = key
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        return api
    except Exception as e:
        st.error(f"Failed to authenticate with Kaggle: {e}")
        return None

def fetch_kaggle_secrets():
    # Try Streamlit Secrets First
    try:
        if "KAGGLE_USERNAME" in st.secrets and "KAGGLE_KEY" in st.secrets:
            return st.secrets["KAGGLE_USERNAME"], st.secrets["KAGGLE_KEY"]
    except Exception:
        pass
    
    # Try OS Env variables
    if "KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ:
        return os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"]
    
    return None, None

@st.cache_data
def download_and_extract_kaggle_dataset(_api, dataset_ref):
    download_path = "./kaggle_downloads"
    os.makedirs(download_path, exist_ok=True)
    
    with st.spinner(f"Downloading {dataset_ref}..."):
        try:
            _api.dataset_download_files(dataset_ref, path=download_path, unzip=True)
            # Find the largest CSV file in the directory
            csv_files = []
            for root, dirs, files in os.walk(download_path):
                for file in files:
                    if file.endswith(".csv"):
                        csv_files.append(os.path.join(root, file))
            
            if not csv_files:
                st.error("No CSV files found in the downloaded dataset.")
                return None
                
            # Get the largest file (assuming it's the main dataset)
            largest_csv = max(csv_files, key=os.path.getsize)
            return pd.read_csv(largest_csv)
        except Exception as e:
            st.error(f"Error downloading dataset: {e}")
            return None

@st.cache_data
def semantic_match_columns_with_llm(user_cols, kaggle_cols, llm_provider, llm_url, model_name, api_key=None, proxies=None):
    prompt = f"""
    Map the following user data headers to the most semantically similar headers from a Kaggle dataset.
    Account for domain-specific units, abbreviations, and synonyms (e.g., 'Heart Rate' to 'bpm', 'Heat' to 'Temp', 'rpm' to 'Engine_Speed').
    User Headers: {user_cols}
    Kaggle Headers: {kaggle_cols}
    
    Output ONLY a JSON object where keys are User Headers and values are Kaggle Headers.
    If no reasonable match exists for a column, omit it.
    Example: {{"Heat": "Coolant_Temp", "Speed": "Velocity_KMH"}}
    """
    try:
        raw = call_llm(prompt, llm_provider, llm_url, model_name, api_key, proxies, system_prompt="You are an expert data engineering assistant specializing in schema mapping and semantic alignment. Output ONLY valid JSON.")
        cleaned = raw.replace('```json', '').replace('```', '').strip()
        return json.loads(cleaned)
    except:
        return {}

def fuzzy_match_columns(user_cols, kaggle_cols, threshold=80):
    mapping = {}
    for uc in user_cols:
        match = process.extractOne(uc, kaggle_cols, scorer=fuzz.token_sort_ratio)
        if match:
            k_col, score = match[0], match[1]
            if score >= threshold:
                mapping[uc] = k_col
    return mapping

def call_llm(prompt, llm_provider, llm_url, model_name, api_key, proxies, system_prompt=None):
    headers = {"Content-Type": "application/json"}
    if api_key: headers["Authorization"] = f"Bearer {api_key}"
    
    if "Anthropic" in llm_provider:
        endpoint = f"{llm_url.rstrip('/')}/messages"
        headers["x-api-key"] = api_key
        headers["anthropic-version"] = "2023-06-01"
        if "Authorization" in headers: del headers["Authorization"]
        payload = {
            "model": model_name,
            "max_tokens": 1024,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}]
        }
        if system_prompt:
            payload["system"] = system_prompt
            
    elif "Gemini" in llm_provider:
        endpoint = f"{llm_url.rstrip('/')}/models/{model_name}:generateContent?key={api_key}"
        if "Authorization" in headers: del headers["Authorization"]
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0}
        }
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
            
    elif "OpenAI" in llm_provider or "OpenRouter" in llm_provider or "Custom REST" in llm_provider:
        endpoint = f"{llm_url.rstrip('/')}/chat/completions"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": 0,
            "stream": False
        }
    else:
        # Ollama format
        endpoint = f"{llm_url.rstrip('/')}/api/generate"
        if "Authorization" in headers: del headers["Authorization"]
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0}
        }
        if system_prompt:
            payload["system"] = system_prompt
        
    if not is_safe_url(endpoint):
        raise ValueError(f"Unsafe or invalid URL provided: {endpoint}")

    response = requests.post(endpoint, json=payload, headers=headers, timeout=300, proxies=proxies)
    response.raise_for_status()
    
    resp_json = response.json()
    if "Anthropic" in llm_provider:
        return resp_json["content"][0]["text"].strip()
    elif "Gemini" in llm_provider:
        return resp_json["candidates"][0]["content"]["parts"][0]["text"].strip()
    elif "OpenAI" in llm_provider or "OpenRouter" in llm_provider or "Custom REST" in llm_provider:
        return resp_json["choices"][0]["message"]["content"].strip()
    else:
        return resp_json.get("response", "").strip()

@st.cache_data
def generate_expanded_search_queries(user_input, headers, category, llm_provider, ollama_url, model_name, api_key=None, proxies=None):
    system_prompt = (
        "You are an expert data scientist. Before searching Kaggle, you must categorize the user's data.\n"
        "Determine the Industry, Specific Machine/Body Part, and Physical Units involved.\n"
        "Then, generate 3 'Search Identities' to find tabular datasets on Kaggle.\n"
        "CRITICAL: Keep each identity very short (2-4 words maximum) to work with keyword search.\n"
        "Output ONLY valid JSON with no markdown formatting containing these keys:\n"
        "- 'industry': The general industry (e.g., 'Industrial / Automotive')\n"
        "- 'machine_or_body_part': Specific machine or body part (e.g., 'Engine', 'Heart')\n"
        "- 'physical_units': Expected physical units (e.g., 'RPM, Celsius', 'BPM, mmHg')\n"
        "- 'identity_a_technical': SHORT technical keywords (e.g., 'engine telemetry')\n"
        "- 'identity_b_result': (use this) SHORT result keywords (e.g., 'predictive maintenance failure')\n"
        "- 'identity_c_synonym': SHORT synonym keywords (e.g., 'vibration thermal logs')\n"
    )
    prompt = "Context:\n"
    if category: prompt += f"Category: {category}\n"
    if user_input: prompt += f"User Issue/Description: {user_input}\n"
    if headers: prompt += f"Dataset Headers: {', '.join(headers)}\n"
    
    try:
        raw = call_llm(prompt, llm_provider, ollama_url, model_name, api_key, proxies, system_prompt=system_prompt)
        cleaned = raw.replace('```json', '').replace('```', '').strip()
        return json.loads(cleaned)
    except Exception as e:
        st.error(f"LLM Error generating expansion: {e}")
        return {
            "industry": "General", 
            "identity_a_technical": "general dataset", 
            "identity_b_result": "tabular data", 
            "identity_c_synonym": "csv data"
        }

@st.cache_data
def rank_datasets_with_llm(user_input, dataset_keys, llm_provider, ollama_url, model_name, api_key=None, proxies=None):
    system_prompt = "You are an expert data science assistant. Select the SINGLE dataset from the list that is most relevant to the user's problem. Output ONLY the exact matching dataset title from the list."
    prompt = f"User's Problem: {user_input}\n\nDatasets Available:\n"
    for k in dataset_keys:
        prompt += f"- {k}\n"
    prompt += "\nWhich dataset is the absolute best match? Output ONLY the exact string from the list above."
    try:
        raw = call_llm(prompt, llm_provider, ollama_url, model_name, api_key, proxies, system_prompt=system_prompt)
        for k in dataset_keys:
            if k in raw or raw in k:
                return k
        return raw.strip()
    except Exception:
        return None

@st.cache_data
def generate_text_comparison_insights(user_text, dataset_summary, kaggle_dataset_ref, llm_provider, ollama_url, model_name, api_key=None, proxies=None):
    system_prompt = "You are an expert diagnostic AI providing comparative medical, mechanical, or financial insights."
    prompt = f"The user has provided the following description/symptoms: '{user_text}'.\n\nI have fetched a relevant Kaggle dataset ('{kaggle_dataset_ref}') with the following statistical summary and structure:\n{dataset_summary}\n\nPlease analyze the user's input against the typical patterns found in this dataset. Provide a comparative diagnostic insight."
    
    try:
        return call_llm(prompt, llm_provider, ollama_url, model_name, api_key, proxies, system_prompt=system_prompt)
    except Exception as e:
        return f"Error connecting to LLM: {str(e)}"

@st.cache_data
def generate_llm_insights(stats_summary, llm_provider, ollama_url, model_name, api_key=None, proxies=None):
    system_prompt = "You are an expert data analyst providing professional, concise narrative summaries of statistical anomalies."
    prompt = f"""
    I have compared user telemetry data against a global Kaggle benchmark dataset.
    Here is the statistical summary of the differences (Z-scores and anomaly detection):
    
    {json.dumps(stats_summary, indent=2)}
    
    Please provide a concise, precise narrative answer. For example, 'Your engine shows 15% higher friction than the 5,000 similar units on Kaggle.'
    Keep it professional, insightful, and highlight any critical anomalies detected.
    """
    
    try:
        return call_llm(prompt, llm_provider, ollama_url, model_name, api_key, proxies, system_prompt=system_prompt)
    except Exception as e:
        return f"Error connecting to LLM: {str(e)}"

def fetch_top_datasets(api, query):
    try:
        # Strategy 1: Explicit file_type filter (Most accurate if indexed correctly)
        datasets = api.dataset_list(search=query, sort_by='votes', file_type='csv')
        
        # Strategy 2: If no results, try without filter but with 'csv' in string
        if not datasets:
            datasets = api.dataset_list(search=f"{query} csv", sort_by='votes')
            
        # Strategy 3: If still no results, try without 'csv' at all
        if not datasets:
            datasets = api.dataset_list(search=query, sort_by='votes')

        high_usability = [d for d in datasets if getattr(d, 'usability_rating', 0) >= 0.7]
        if not high_usability:
            high_usability = datasets
        return high_usability[:10]
    except Exception as e:
        return []

def is_valid_dataset(d, user_context):
    # Filter out image/cv datasets unless user specifically mentions it
    title = getattr(d, 'title', '') or ''
    description = getattr(d, 'description', '') or ''
    title_desc_tags = (title + " " + description).lower()
    if hasattr(d, 'tags') and d.tags:
        title_desc_tags += " " + " ".join([str(t).lower() for t in d.tags if t is not None])
    
    # Negative signals (Image/CV/Audio)
    img_keywords = ['computer vision', 'image dataset', 'pixels', 'resnet', 'cnn', 'spectrogram', 'audio classification']
    user_wants_img = any(k in (user_context or "").lower() for k in ['image', 'vision', 'picture', 'photo'])
    
    # Positive signals (Tabular)
    tabular_keywords = ['csv', 'tabular', 'spreadsheet', 'dataframe', 'timeseries', 'telemetry', 'sensor data', 'clinical records']
    is_likely_tabular = any(k in title_desc_tags for k in tabular_keywords)

    if not user_wants_img:
        # If it has heavy image keywords but no tabular indicators, reject
        if any(k in title_desc_tags for k in img_keywords) and not is_likely_tabular:
            return False
            
    return True

def waterfall_kaggle_search(api, user_text, headers, id_a, id_b, id_c, industry=""):
    queries = []
    
    # Tier 1 (Most precise AI inference): Technical Categorical Identity
    if id_a: queries.append(f"{id_a} csv")
    
    # Tier 2: Result-Oriented Categorical Identity
    if id_b: queries.append(f"{id_b} csv")
    
    # Tier 3 (Domain-Guard): Industry + Exact headers
    if headers and len(headers) > 0:
        domain_prefix = f"{industry} " if industry and industry != "Unknown" else ""
        queries.append(f"{domain_prefix}{' '.join(headers[:3])} dataset csv")
        
    # Tier 4: Synonym-heavy Identity (Broadest AI inference)
    if id_c: queries.append(f"{id_c} csv")
    
    for q in queries:
        datasets = fetch_top_datasets(api, q)
        # Filter datasets further for non-tabular items if any slip through
        valid_datasets = [d for d in datasets if is_valid_dataset(d, user_text)]
        if valid_datasets:
            return valid_datasets
    return []

# --- MAIN APP ---
st.title("📊 Comparative Analysis with Kaggle")
st.markdown("Automated expert audits comparing your data against global Kaggle datasets.")

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")

# Kaggle Auth
kaggle_user, kaggle_key = fetch_kaggle_secrets()

if not kaggle_user or not kaggle_key:
    st.sidebar.error("Kaggle credentials not found in `.streamlit/secrets.toml`. Please configure them to use the app.")
    st.stop()
else:
    st.sidebar.success("Created by Mohammad Saeed Angiz")

# LLM Config
st.sidebar.subheader("🤖 LLM Settings")

providers = [
    "Ollama (Local)", 
    "OpenRouter", 
    "OpenAI", 
    "Anthropic (Claude)", 
    "Google Gemini", 
    "Custom REST (User Input)"
]
llm_provider = st.sidebar.selectbox("LLM Provider", providers)

if llm_provider == "Ollama (Local)":
    default_url = "http://localhost:11434"
    default_model = "llama3"
elif llm_provider == "OpenRouter":
    default_url = "https://openrouter.ai/api/v1"
    default_model = "anthropic/claude-3-opus"
elif llm_provider == "OpenAI":
    default_url = "https://api.openai.com/v1"
    default_model = "gpt-4o"
elif llm_provider == "Anthropic (Claude)":
    default_url = "https://api.anthropic.com/v1"
    default_model = "claude-3-5-sonnet-20240620"
elif llm_provider == "Google Gemini":
    default_url = "https://generativelanguage.googleapis.com/v1beta"
    default_model = "gemini-1.5-pro"
else:
    default_url = ""
    default_model = ""

llm_url = st.sidebar.text_input("Endpoint URL", value=default_url)
llm_model = st.sidebar.text_input("Model Name", value=default_model)
llm_key = st.sidebar.text_input("API Key (if required)", type="password")

with st.sidebar.expander("📖 Setup Guide: Supported LLMs"):
    st.markdown("""
    **Local Privacy (Ollama)**
    1. Run `ollama run llama3`. Leave API Key blank.
    
    **Cloud Providers**
    - **OpenRouter**: Get an API key from OpenRouter.ai. 
    - **OpenAI**: Use `gpt-4o` or `gpt-3.5-turbo`.
    - **Anthropic**: Use `claude-3-5-sonnet-20240620` or `claude-3-opus-20240229`.
    - **Gemini**: Use `gemini-1.5-pro`.
    
    **Custom REST**
    Use this if you have a private OpenAI-compatible endpoint (like LM Studio or vLLM). Input your custom URL.
    """)

with st.sidebar.expander("🌐 Advanced: Proxy Layer"):
    st.caption("Only enable if running a local proxy tool like Burp Suite.")
    use_proxy = st.checkbox("Enable Proxy for API Requests", value=False)
    proxies = None
    if use_proxy:
        http_proxy = st.text_input("HTTP Proxy", value="http://127.0.0.1:8080")
        https_proxy = st.text_input("HTTPS Proxy", value="http://127.0.0.1:8080")
        proxies = {
            "http": http_proxy,
            "https": https_proxy,
        }
        st.info("Proxy Layer Active")
if st.sidebar.button("Clear Cache"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.sidebar.success("Cache cleared!")

# --- STEP 1: DYNAMIC INPUT & INGESTION ---
st.header("1. Input Context & Data")

col_cat, col_input = st.columns([1, 2])
with col_cat:
    category = st.selectbox("Category Context", ["Medical Diagnostics", "Mechanical Engineering", "Financial Analysis", "Other / Custom"])
with col_input:
    user_text = st.text_area("Manual Input / Symptoms (e.g., 'I'm feeling dizzy after meals')", height=100)

uploaded_file = st.file_uploader("Upload CSV, Excel, JSON, PDF, or DOCX (Optional)", type=["csv", "xlsx", "json", "pdf", "docx", "doc"])

user_df = None
user_numeric_cols = []
headers = []

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            user_df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            user_df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith(".json"):
            user_df = pd.read_json(uploaded_file)
        elif uploaded_file.name.endswith(".pdf"):
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            extracted_pages = []
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    extracted_pages.append(text)
            extracted_text = "\n".join(extracted_pages)
            user_text += "\n\n--- Extracted from PDF ---\n" + extracted_text.strip()
            st.success(f"Successfully extracted text from {uploaded_file.name}.")
        elif uploaded_file.name.endswith(".docx") or uploaded_file.name.endswith(".doc"):
            doc = docx.Document(uploaded_file)
            extracted_text = "\n".join([para.text for para in doc.paragraphs])
            user_text += "\n\n--- Extracted from Document ---\n" + extracted_text.strip()
            st.success(f"Successfully extracted text from {uploaded_file.name}.")
        if user_df is not None:
            st.success(f"Successfully loaded {len(user_df)} rows and {len(user_df.columns)} columns.")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()
        
with st.expander("⌨️ Manual Telemetry Entry (Data Editor)"):
    st.write("Enter or paste your numeric data directly below if you don't have a file.")
    # Create a template df if no file is uploaded
    template_df = pd.DataFrame(columns=["Sensor_A", "Sensor_B", "Metric_C"]) if user_df is None else user_df.head(0)
    manual_df = st.data_editor(template_df, num_rows="dynamic")
    if not manual_df.empty and user_df is None:
        user_df = manual_df
        st.info("Using manually entered data.")

if user_df is not None:
    with st.expander("Preview Data"):
        st.dataframe(user_df.head())
    user_numeric_cols = user_df.select_dtypes(include=[np.number]).columns.tolist()
    headers = user_df.columns.tolist()
elif user_text:
    with st.expander("Preview Extracted Text"):
        st.text(user_text)

if not user_text and user_df is None:
    st.info("Please provide a text description or upload a dataset to proceed.")
    st.stop()

# --- STEP 2: KAGGLE SEARCH ---
st.header("2. Benchmark Dataset (Kaggle)")

api = init_kaggle(kaggle_user, kaggle_key)
if not api:
    st.stop()
    
# Helper to process and rank options
def process_and_rank_datasets(datasets, context_text):
    if not datasets:
        return {}
    options = {f"{d.title} ({d.ref})": d.ref for d in datasets}
    best_match = rank_datasets_with_llm(context_text, list(options.keys()), llm_provider, llm_url, llm_model, llm_key, proxies)
    
    if best_match and best_match in options:
        ordered_options = {f"⭐ AI Best Match: {best_match}": options[best_match]}
        for k, v in options.items():
            if k != best_match:
                ordered_options[k] = v
        return ordered_options
    return options

# Manual Kaggle search – user provides keywords directly
col_search1, col_search2 = st.columns([2, 1])
with col_search1:
    search_query = st.text_input("Search Kaggle Datasets", value="")
    if st.button("Search Kaggle"):
        with st.spinner("Searching and ranking datasets..."):
            datasets = fetch_top_datasets(api, search_query)
            valid_datasets = [d for d in datasets if is_valid_dataset(d, user_text if user_text else search_query)]
            
            if not valid_datasets:
                st.warning("Exact match failed or returned image datasets. Trying Semantic Expansion...")
                expanded = generate_expanded_search_queries(search_query, headers, category, llm_provider, llm_url, llm_model, llm_key, proxies)
                valid_datasets = waterfall_kaggle_search(api, user_text if user_text else search_query, headers, expanded.get("identity_a_technical"), expanded.get("identity_b_result"), expanded.get("identity_c_synonym"), expanded.get("industry"))
            
            if valid_datasets:
                context = user_text if user_text else search_query
                st.session_state['dataset_options'] = process_and_rank_datasets(valid_datasets, context)
            else:
                st.session_state['dataset_options'] = {}
                st.warning("No Kaggle datasets could be found for any query.")

with col_search2:
    st.write(" ")
    st.write(" ")
    if st.button("Get related datasets from Kaggle"):
        with st.spinner("Generating Semantic Expansion queries..."):
            expanded = generate_expanded_search_queries(user_text, headers, category, llm_provider, llm_url, llm_model, llm_key, proxies)
            id_a = html.escape(expanded.get("identity_a_technical", ""))
            id_b = html.escape(expanded.get("identity_b_result", ""))
            id_c = html.escape(expanded.get("identity_c_synonym", ""))
            industry = html.escape(expanded.get('industry', 'Unknown'))
            st.markdown(
                f"""
                <div style="background-color: rgba(128, 128, 128, 0.8); padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
                    <strong>Categorized as:</strong> {industry}<br><br>
                    <strong>Expanded Search Identities:</strong>
                    <ul style="margin-bottom: 0;">
                        <li><strong>Technical</strong>: {id_a}</li>
                        <li><strong>Result-Oriented</strong>: {id_b}</li>
                        <li><strong>Synonym-Heavy</strong>: {id_c}</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            with st.spinner("Executing Waterfall Search..."):
                datasets = waterfall_kaggle_search(api, user_text, headers, id_a, id_b, id_c, expanded.get("industry"))
                if datasets:
                    with st.spinner("AI ranking best match..."):
                        context = user_text if user_text else f"{id_a} {id_b}"
                        st.session_state['dataset_options'] = process_and_rank_datasets(datasets, context)
                else:
                    st.warning("No tabular datasets found on Kaggle using semantic expansion.")
            
if 'dataset_options' in st.session_state and st.session_state['dataset_options']:
    selected_dataset_name = st.selectbox("Select a benchmark dataset:", list(st.session_state['dataset_options'].keys()))
    selected_dataset_ref = st.session_state['dataset_options'][selected_dataset_name]
    
    if st.button("Harvest & Compare"):
        st.session_state['harvested_dataset_ref'] = selected_dataset_ref
        
    if st.session_state.get('harvested_dataset_ref') == selected_dataset_ref:
        kaggle_df = download_and_extract_kaggle_dataset(api, selected_dataset_ref)
        
        if kaggle_df is not None:
            st.success(f"Loaded Kaggle Dataset: {len(kaggle_df)} rows.")
            st.write("**Kaggle Dataset Headers:**", ", ".join(kaggle_df.columns.astype(str).tolist()))
            with st.expander("Preview Kaggle Data"):
                st.dataframe(kaggle_df.head())
            
            # --- STEP 3: HARVEST & COMPARISON ---
            st.header("3. AI Comparison & Synthesis")
            
            # If user provided text, do Text Comparison
            if user_text:
                st.subheader("Text Diagnosis vs Kaggle Dataset")
                with st.spinner("AI evaluating symptoms against dataset structure..."):
                    summary_json = kaggle_df.describe().to_json()
                    insights = generate_text_comparison_insights(user_text, summary_json, selected_dataset_ref, llm_provider, llm_url, llm_model, llm_key, proxies)
                    safe_insights = html.escape(insights)
                    st.markdown(
                        f"""
                        <div style="background-color: rgba(128, 128, 128, 0.8); padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; border: 1px solid rgba(128, 128, 128, 0.4);">
                            {safe_insights}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            # If user provided data, do Statistical Comparison
            if user_df is not None:
                st.subheader("Statistical Data Alignment & Mapping")
                kaggle_numeric_cols = kaggle_df.select_dtypes(include=[np.number]).columns.tolist()
                
                mapping = fuzzy_match_columns(user_numeric_cols, kaggle_numeric_cols, threshold=80)
                
                if len(mapping) < len(user_numeric_cols):
                    with st.spinner("Lexical match incomplete. Attempting AI Semantic Mapping..."):
                        unmapped = [c for c in user_numeric_cols if c not in mapping]
                        semantic_map = semantic_match_columns_with_llm(unmapped, kaggle_numeric_cols, llm_provider, llm_url, llm_model, llm_key, proxies)
                        mapping.update(semantic_map)
                
                if not mapping:
                    st.error("Could not find matching numeric columns between your data and the Kaggle dataset. (Confidence < 80%)")
                else:
                    st.write("### Matched Columns (100% or >80% confidence)")
                    match_df = pd.DataFrame(list(mapping.items()), columns=["Your Column", "Kaggle Column"])
                    st.table(match_df)
                    
                    st.write("### Tabular Delta (Schema Differences)")
                    unmatched_user = [c for c in headers if c not in mapping.keys()]
                    unmatched_kaggle = [c for c in kaggle_df.columns if c not in mapping.values()]
                    
                    delta_col1, delta_col2 = st.columns(2)
                    with delta_col1:
                        st.write("**Variables only in Your Data**")
                        if unmatched_user:
                            st.write(", ".join(unmatched_user))
                        else:
                            st.write("None")
                    with delta_col2:
                        st.write("**Variables only in Kaggle Data**")
                        if unmatched_kaggle:
                            st.write(", ".join(unmatched_kaggle))
                        else:
                            st.write("None")
                    
                    stats_results = []
                    processed_data = user_df.copy()
                    
                    for user_col, kaggle_col in mapping.items():
                        st.write(f"**Comparison: `{user_col}` vs `{kaggle_col}`**")
                        
                        user_data_numeric = pd.to_numeric(user_df[user_col], errors='coerce')
                        user_data = user_data_numeric.dropna()
                        kagg_data = pd.to_numeric(kaggle_df[kaggle_col], errors='coerce').dropna()
                        
                        user_mean = user_data.mean()
                        kagg_mean = kagg_data.mean()
                        kagg_std = kagg_data.std()
                        
                        if kagg_std == 0 or pd.isna(kagg_std):
                            continue
                            
                        # Calculate Z-scores for user data against kaggle distribution
                        z_scores = (user_data - kagg_mean) / kagg_std
                        anomalies = z_scores[np.abs(z_scores) > 2]
                        
                        # Calculate Percentile
                        user_percentile = (kagg_data < user_mean).mean() * 100
                        
                        # Add z-score to processed dataframe
                        # processed_data is a copy of user_df, so user_col has the same values
                        processed_data[f"{user_col}_zscore_vs_kaggle"] = (user_data_numeric - kagg_mean) / kagg_std
                        processed_data[f"{user_col}_is_anomaly"] = np.abs(processed_data[f"{user_col}_zscore_vs_kaggle"]) > 2
                        
                        diff_pct = ((user_mean - kagg_mean) / kagg_mean) * 100 if kagg_mean != 0 else 0
                        
                        stats_results.append({
                            "column": user_col,
                            "matched_kaggle_column": kaggle_col,
                            "user_mean": float(user_mean),
                            "kaggle_mean": float(kagg_mean),
                            "difference_percent": float(diff_pct),
                            "user_percentile": float(user_percentile),
                            "total_anomalies": len(anomalies)
                        })
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Your Mean", f"{user_mean:.2f}")
                        col2.metric("Kaggle Mean", f"{kagg_mean:.2f}", f"{diff_pct:.1f}%")
                        col3.metric("Z-Score (Mean)", f"{(user_mean - kagg_mean)/kagg_std:.2f}")
                        col4.metric("Anomalies Detected", f"{len(anomalies)} / {len(user_data)}")
                        
                        # Visualizations
                        vcol1, vcol2 = st.columns(2)
                        
                        with vcol1:
                            fig = go.Figure()
                            fig.add_trace(go.Histogram(x=kagg_data, name='Kaggle Benchmark', opacity=0.5, histnorm='probability density'))
                            fig.add_trace(go.Histogram(x=user_data, name='Your Data', opacity=0.7, histnorm='probability density'))
                            fig.update_layout(barmode='overlay', title=f"Distribution Comparison: {user_col}")
                            st.plotly_chart(fig, use_container_width=True)
                            
                        with vcol2:
                            fig_box = go.Figure()
                            fig_box.add_trace(go.Box(y=kagg_data, name='Kaggle Benchmark'))
                            fig_box.add_trace(go.Box(y=user_data, name='Your Data'))
                            fig_box.update_layout(title=f"Box Plot: {user_col}")
                            st.plotly_chart(fig_box, use_container_width=True)
                    
                    if st.button("Generate Statistical Insights with LLM"):
                        with st.spinner("Analyzing statistics..."):
                            stat_insights = generate_llm_insights(stats_results, llm_provider, llm_url, llm_model, llm_key, proxies)
                            safe_stat_insights = html.escape(stat_insights)
                            st.markdown(
                                f"""
                                <div style="background-color: rgba(128, 128, 128, 0.8); padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; border: 1px solid rgba(128, 128, 128, 0.4);">
                                    {safe_stat_insights}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            
                    # --- EXPORT ---
                    st.header("4. Export Processed Data")
                    st.write("Download your enriched data in any of the supported formats:")
                    
                    csv_data = processed_data.to_csv(index=False).encode('utf-8')
                    json_data = processed_data.to_json(orient='records').encode('utf-8')
                    
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        processed_data.to_excel(writer, index=False)
                    excel_data = excel_buffer.getvalue()
                    
                    col_dl1, col_dl2, col_dl3 = st.columns(3)
                    with col_dl1:
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv_data,
                            file_name='comparative_analysis_results.csv',
                            mime='text/csv',
                            use_container_width=True
                        )
                    with col_dl2:
                        st.download_button(
                            label="📥 Download Excel",
                            data=excel_data,
                            file_name='comparative_analysis_results.xlsx',
                            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                            use_container_width=True
                        )
                    with col_dl3:
                        st.download_button(
                            label="📥 Download JSON",
                            data=json_data,
                            file_name='comparative_analysis_results.json',
                            mime='application/json',
                            use_container_width=True
                        )
        # Cleanup downloaded files
        if os.path.exists("./kaggle_downloads"):
            shutil.rmtree("./kaggle_downloads")
