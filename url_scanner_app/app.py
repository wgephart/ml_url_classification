import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tldextract
import Levenshtein
import re
import math
from collections import Counter

# --- 1. SETUP & CONFIGURATION ---
st.set_page_config(page_title="Malicious URL Detector", page_icon="🛡️")

# Define the list of brands for feature extraction (Same as your notebook)
BRANDS = ['google', 'paypal', 'microsoft', 'apple', 'amazon', 'netflix', 'facebook']

# --- 2. LOAD ARTIFACTS (Cached for speed) ---
@st.cache_resource
def load_artifacts():
    # Load Helper Tools
    with open('artifacts.pkl', 'rb') as f:
        artifacts = pickle.load(f)
    
    # Load XGBoost
    with open('xgboost_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
        
    # Load Neural Network
    nn_model = tf.keras.models.load_model('neural_network.keras')
    
    return artifacts, xgb_model, nn_model

try:
    artifacts, xgb_model, nn_model = load_artifacts()
    scaler = artifacts['scaler']
    tokenizer = artifacts['tokenizer']
    encoder = artifacts['encoder']
    top_tlds = artifacts['top_tlds']
    max_len = artifacts['max_len']
except FileNotFoundError:
    st.error("Error: Model files not found. Make sure .pkl and .keras files are in the same directory.")
    st.stop()

# --- 3. FEATURE ENGINEERING FUNCTIONS ---
def calc_entropy(text):
    if not text: return 0
    entropy = 0
    for x, n in Counter(str(text)).items():
        p = n / len(text)
        entropy -= p * math.log2(p)
    return entropy

def extract_url_features(url):
    url = str(url)
    features = {}
    
    # --- Structural Features ---
    features['url_length'] = len(url)
    features['num_digits'] = sum(c.isdigit() for c in url)
    features['num_periods'] = url.count('.')
    features['num_slashes'] = url.count('/')
    features['num_ats'] = url.count('@')
    features['digit_len_ratio'] = sum(c.isdigit() for c in url) / len(url) if len(url) > 0 else 0
    features['entropy'] = calc_entropy(url)
    
    # --- Boolean Flags ---
    patterns = {
        'has_html': r'\.html?',
        'has_query_param': r'\?query=',
        'has_https': r'^https://',
        'has_http': r'^http://',
        'has_ip_address': r'://(?:\d{1,3}\.){3}\d{1,3}',
        'has_non_ascii_chars': r'[^\x00-\x7F]'
    }
    for name, pattern in patterns.items():
        features[name] = int(bool(re.search(pattern, url, re.IGNORECASE)))
        
    # --- Keywords & Brands ---
    suspicious_kw = ['login', 'secure', 'payment', 'verify']
    features['has_suspicious_kw'] = int(any(kw in url.lower() for kw in suspicious_kw))
    features['has_brand_kw'] = int(any(brand in url.lower() for brand in BRANDS))
    
    # --- Domain Logic ---
    ext = tldextract.extract(url)
    domain = ext.domain
    suffix = ext.suffix
    
    min_dist = 100
    for brand in BRANDS:
        dist = Levenshtein.distance(domain.lower(), brand)
        if dist < min_dist:
            min_dist = dist
    features['min_brand_dist'] = min_dist
    
    if min_dist == 0:
        features['is_typosquat'] = 0
    elif 0 < min_dist <= 2:
        features['is_typosquat'] = 1
    else:
        features['is_typosquat'] = 0
        
    return features, suffix

# --- 4. PREDICTION PIPELINE ---
def make_prediction(url):
    # A. Extract Base Features
    feats, tld_suffix = extract_url_features(url)
    
    # B. Handle TLD One-Hot Encoding (Manually match training columns)
    # We must create exactly 1 column for every TLD in 'top_tlds'
    tld_data = {}
    for t in top_tlds:
        tld_data[f'tld_{t}'] = 1 if tld_suffix == t else 0
    tld_data['tld_other'] = 1 if tld_suffix not in top_tlds else 0
    
    # Combine everything
    full_features = {**feats, **tld_data}
    
    # Convert to DataFrame (ensure correct column order if possible, 
    # but usually sklearn/keras handle this if keys match)
    # Ideally, we should enforce order, but for now we rely on the scaler
    df_single = pd.DataFrame([full_features])
    
    # C. Prepare Numerical Input
    # Important: Reorder columns to match scaler's expectation if needed
    # (The scaler expects the exact same feature order as training)
    # For robust code, we usually save the column names list in artifacts.
    # For now, we assume the dict insertion order matches generally.
    X_num = scaler.transform(df_single)
    
    # D. Prepare Text Input
    seq = tokenizer.texts_to_sequences([url])
    X_text = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    
    # E. Get Predictions
    xgb_prob = xgb_model.predict_proba(X_num)[0][1] # Probability of class 1
    xgb_class = "Malicious" if xgb_prob > 0.5 else "Benign"
    
    nn_probs = nn_model.predict([X_text, X_num])
    nn_class_idx = np.argmax(nn_probs, axis=1)[0]
    nn_label = encoder.inverse_transform([nn_class_idx])[0]
    nn_confidence = float(np.max(nn_probs))
    
    return xgb_class, xgb_prob, nn_label, nn_confidence

# --- 5. STREAMLIT UI ---
st.title("🛡️ Malicious URL Detector")
st.markdown("Enter a URL below to check if it's safe.")

url_input = st.text_input("Enter URL:", placeholder="http://example.com")

if st.button("Analyze URL"):
    if not url_input:
        st.warning("Please enter a URL first.")
    else:
        with st.spinner("Scanning..."):
            try:
                xgb_res, xgb_prob, nn_res, nn_conf = make_prediction(url_input)
                
                # Create two columns for results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Level 1: XGBoost")
                    if xgb_res == "Malicious":
                        st.error(f"🚨 {xgb_res}")
                    else:
                        st.success(f"✅ {xgb_res}")
                    st.metric("Probability of Malice", f"{xgb_prob:.2%}")
                
                with col2:
                    st.subheader("Level 2: Neural Network")
                    st.info(f"📂 Classification: **{nn_res}**")
                    st.metric("Confidence", f"{nn_conf:.2%}")
                
                # Specific advice based on findings
                st.divider()
                if nn_res == 'phishing' or nn_res == 'defacement':
                    st.warning("⚠️ This site appears to be structurally similar to known phishing/defacement attacks.")
                elif nn_res == 'benign' and xgb_res == 'Benign':
                    st.success("👍 Both models agree this site looks safe.")
                
            except Exception as e:
                st.error(f"An error occurred: {e}")