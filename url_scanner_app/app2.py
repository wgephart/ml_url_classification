import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import tensorflow as tf
import tldextract
import Levenshtein
import re
import math
from collections import Counter
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- 1. SETUP & CONFIGURATION ---
st.set_page_config(page_title="Malicious URL Detector", page_icon="🛡️")
st.header("Malicious URL Detector")

# Allowlist for major sites (Bypasses AI to prevent false positives)
SAFE_DOMAINS = [
    'google.com', 'www.google.com', 'youtube.com', 'facebook.com', 
    'amazon.com', 'wikipedia.org', 'instagram.com', 'twitter.com', 
    'linkedin.com', 'netflix.com', 'microsoft.com', 'apple.com',
    'github.com', 'nytimes.com', 'cnn.com'
]

BRANDS = ['google', 'paypal', 'microsoft', 'apple', 'amazon', 'netflix', 'facebook', 'bank', 'adobe']
SUSPICIOUS_WORDS = ['login', 'signin', 'verify', 'update', 'account', 'secure', 'banking', 'confirm']

# --- CRITICAL: DISABLE MAC GPU ---
try:
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    pass

# --- 2. LOAD ARTIFACTS ---
@st.cache_resource
def load_models():
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    
    scaler = joblib.load('scaler.pkl')
    encoder = joblib.load('label_encoder.pkl')
    #xgb_model = joblib.load('xgb_gatekeeper.pkl') # Make sure this name matches your file!
    nn_model = tf.keras.models.load_model('hybrid_model.h5')
    
    return tokenizer, scaler, encoder, nn_model

try:
    tokenizer, scaler, encoder, nn_model = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# --- 3. HELPER FUNCTIONS ---
def calc_entropy(text):
    if not text: return 0
    entropy = 0
    for x, n in Counter(str(text)).items():
        p = n / len(text)
        entropy -= p * math.log2(p)
    return entropy

def check_brand_usage(row):
    # This expects 'clean_url' logic inside the row data if possible, 
    # or we handle it inside extract_features
    return 0 

def is_whitelisted(url):
    try:
        obj = tldextract.extract(url)
        full_domain = f"{obj.domain}.{obj.suffix}"
        return full_domain in SAFE_DOMAINS or url in SAFE_DOMAINS
    except:
        return False

# --- 4. EXTRACTOR (THE HYBRID LOGIC) ---
def extract_features(raw_url):
    # 1. CLEANING (Match Training Logic)
    clean_url_text = re.sub(r'^https?://', '', raw_url)
    clean_url_text = re.sub(r'^www\.', '', clean_url_text)
    
    # 2. Extract Domain from CLEAN text
    try:
        ext = tldextract.extract(clean_url_text)
        domain_part = ext.domain
    except:
        domain_part = ""
        
    # Initialize Dictionary
    row = {}
    
    # --- A. Structural Features (Use CLEAN text) ---
    row['url_length'] = len(clean_url_text)
    row['num_digits'] = len(re.findall(r'\d', clean_url_text))
    row['num_periods'] = clean_url_text.count('.')
    row['num_slashes'] = clean_url_text.count('/')
    row['num_ats'] = clean_url_text.count('@')
    
    # --- B. Typosquatting ---
    min_dist = 100
    clean_domain = domain_part.lower()
    for brand in BRANDS:
        dist = Levenshtein.distance(clean_domain, brand)
        if dist < min_dist:
            min_dist = dist
    row['min_brand_dist'] = min_dist
    
    if min_dist == 0: row['is_typosquat'] = 0
    elif 0 < min_dist <= 2: row['is_typosquat'] = 1
    else: row['is_typosquat'] = 0
    
    # --- C. Ratios ---
    if row['url_length'] > 0:
        row['digit_len_ratio'] = row['num_digits'] / (row['url_length'] + 1)
    else:
        row['digit_len_ratio'] = 0
        
    # --- D. Suspicious Brand Usage (Use CLEAN text) ---
    row['is_suspicious_brand_usage'] = 0
    url_lower = clean_url_text.lower()
    domain_lower = domain_part.lower()
    for brand in BRANDS:
        if brand in url_lower and brand not in domain_lower:
            row['is_suspicious_brand_usage'] = 1
            
    # --- E. Entropy (Use CLEAN text) ---
    row['entropy'] = calc_entropy(clean_url_text)
    
    # --- F. Boolean Flags (The Logic Split) ---
    
    # CRITICAL: Use 'raw_url' for protocol checks (HTTPS/HTTP)
    # Note: We removed 'has_https'/'has_http' from training because they were broken,
    # so we DO NOT calculate them here to avoid column mismatch errors.
    
    # Use 'raw_url' for IP check and HTML check
    row['has_html'] = int(bool(re.search(r'\.html?', raw_url, re.IGNORECASE)))
    row['has_query_param'] = int(bool(re.search(r'\?query=', raw_url, re.IGNORECASE)))
    row['has_ip_address'] = int(bool(re.search(r'://(?:\d{1,3}\.){3}\d{1,3}', raw_url, re.IGNORECASE)))
    row['has_non_ascii_chars'] = int(bool(re.search(r'[^\x00-\x7F]', raw_url)))
    
    # Use 'clean_url_text' for keyword matching
    kw_pattern = '(' + '|'.join(SUSPICIOUS_WORDS) + ')'
    row['has_suspicious_kw'] = int(bool(re.search(kw_pattern, clean_url_text, re.IGNORECASE)))
    brand_pattern = '|'.join(BRANDS)
    row['has_brand_kw'] = int(bool(re.search(brand_pattern, clean_url_text, re.IGNORECASE)))
    
    # Add keyword count
    #row['sus_keyword_count'] = sum(1 for word in SUSPICIOUS_WORDS if word in clean_url_text.lower())

    # --- 3. DATAFRAME CREATION ---
    df = pd.DataFrame([row])
    
    # Neural Network Columns (Exact Order from Training)
    # Note: 'has_https' and 'has_http' are REMOVED based on your last training update
    nn_cols = [
        'url_length', 'num_digits', 'num_periods', 'num_slashes', 'num_ats',
        'min_brand_dist', 'is_typosquat', 'digit_len_ratio', 
        'is_suspicious_brand_usage', 'entropy', 
        'has_html', 'has_query_param', 
        'has_ip_address', 'has_suspicious_kw', 'has_brand_kw', 'has_non_ascii_chars'
    ]
    
    X_nn = df.reindex(columns=nn_cols, fill_value=0)
    
    # XGBoost Columns (Assuming you retrained it on the same features)
    #X_xgb = X_nn.copy()
    
    return X_nn, clean_url_text

# --- 5. PREDICTION LOGIC ---
def make_prediction(raw_url):
    # A. Extract Features & Clean Text
    X_nn_raw, clean_url_text = extract_features(raw_url)
    
    # B. XGBoost Prediction
    #xgb_pred = xgb_model.predict(X_xgb_raw)[0]
    #xgb_prob = xgb_model.predict_proba(X_xgb_raw)[0][1]
    
    # C. Neural Network Prediction
    # 1. Scale Numerical Data
    X_num_scaled = scaler.transform(X_nn_raw).astype('float32')
    
    # 2. Tokenize Text Data (Use the CLEAN text!)
    seq = tokenizer.texts_to_sequences([clean_url_text])
    X_text = pad_sequences(seq, maxlen=200, padding='post', truncating='post')
    
    # 3. Predict
    nn_probs = nn_model.predict([X_text, X_num_scaled])
    nn_class_idx = np.argmax(nn_probs)
    nn_label = encoder.inverse_transform([nn_class_idx])[0]
    nn_conf = np.max(nn_probs)
    
    return nn_label, nn_conf

# --- 6. UI LAYOUT ---
url_input = st.text_input("Enter URL:", placeholder="http://example.com")

if st.button("Scan URL"):
    if not url_input:
        st.warning("Please enter a URL.")
    
    # 1. Whitelist Check
    elif is_whitelisted(url_input):
        st.success(f"✅ **Result: BENIGN (Verified Safe Domain)**")
        st.info("This site is on our trusted whitelist (e.g. Google, GitHub). No AI scan needed.")
        
    # 2. AI Scan
    else:
        with st.spinner("Analyzing..."):
            try:
                nn_res, nn_conf = make_prediction(url_input)
                
                # --- DYNAMIC STYLING ---
                st.divider()
                st.subheader("Analysis Results")
                
                if nn_res == 'benign':
                    # GREEN BOX for Safe
                    st.success(f"✅ **Result: BENIGN**")
                    st.metric("Confidence Score", f"{nn_conf:.2%}")
                    st.caption("This URL appears safe based on our analysis.")
                else:
                    # RED BOX for Threat
                    st.error(f"🚨 **Result: {nn_res.upper()} DETECTED**")
                    st.metric("Confidence Score", f"{nn_conf:.2%}")
                    st.warning("⚠️ Proceed with extreme caution. This site exhibits malicious patterns.")
                
                st.divider()
                st.caption("Disclaimer: AI models can make mistakes. Always verify the source manually.")
                    
            except Exception as e:
                st.error(f"Prediction Error: {e}")