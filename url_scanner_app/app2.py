import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import tensorflow as tf
try:
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    pass
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tldextract
import Levenshtein
import re
import math
from collections import Counter

# --- 1. SETUP & CONFIGURATION ---
st.set_page_config(page_title="Malicious URL Detector", page_icon="🛡️")

BRANDS = ['google', 'paypal', 'microsoft', 'apple', 'amazon', 'netflix', 'facebook', 'bank', 'adobe']
SUSPICIOUS_WORDS = ['login', 'signin', 'verify', 'update', 'account', 'secure', 'banking', 'confirm']

# --- 2. LOAD ARTIFACTS ---
@st.cache_resource
def load_models():
    # 1. Load Tokenizer (Saved with Pickle)
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
        
    # 2. Load Scaler & Encoder (Saved with Joblib)
    scaler = joblib.load('scaler.pkl')        # <--- CHANGED to joblib
    encoder = joblib.load('label_encoder.pkl') # <--- CHANGED to joblib
    
    # 3. Load XGBoost (Saved with Joblib)
    xgb_model = joblib.load('xgb_model.pkl') # <--- CHANGED to joblib
    
    # 4. Load Neural Network (Saved with Keras)
    nn_model = tf.keras.models.load_model('hybrid_model.h5')
    
    return tokenizer, scaler, encoder, xgb_model, nn_model

try:
    tokenizer, scaler, encoder, xgb_model, nn_model = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()
# --- 3. FEATURE EXTRACTION ---
def calc_entropy(text):
    if not text: return 0
    entropy = 0
    for x, n in Counter(str(text)).items():
        p = n / len(text)
        entropy -= p * math.log2(p)
    return entropy

def check_brand_usage(row):
    url_str = str(row['url']).lower()
    domain_str = str(row['domain']).lower()
    for brand in BRANDS:
        if brand in url_str and brand not in domain_str:
            return 1
    return 0

# --- UPDATED EXTRACTOR (Matches your Notebook exactly) ---
def extract_features(url):
    # Base Dataframe
    try:
        domain = tldextract.extract(url).domain
    except:
        domain = ""
        
    df = pd.DataFrame({'url': [url], 'domain': [domain]})
    
    # 1. Structural Counts
    df['url_length'] = df['url'].str.len()
    df['num_digits'] = df['url'].str.count(r'\d')
    df['num_periods'] = df['url'].str.count(r'\.')
    df['num_slashes'] = df['url'].str.count(r'/')
    df['num_ats'] = df['url'].str.count(r'@')
    
    # 2. Typosquatting
    min_dist = 100 
    for brand in BRANDS:
        dist = Levenshtein.distance(domain.lower(), brand)
        if dist < min_dist: min_dist = dist
            
    df['min_brand_dist'] = min_dist
    
    if min_dist == 0: df['is_typosquat'] = 0
    elif 0 < min_dist <= 2: df['is_typosquat'] = 1
    else: df['is_typosquat'] = 0
    
    # 3. Ratios & Advanced Stats
    # Note: Your notebook called it 'digit_len_ratio', so we use that name
    df['digit_len_ratio'] = df['num_digits'] / (df['url_length'] + 1)
    
    df['entropy'] = df['url'].apply(calc_entropy)
    df['sus_keyword_count'] = df['url'].apply(lambda x: sum(1 for word in SUSPICIOUS_WORDS if word in x.lower()))
    df['is_suspicious_brand_usage'] = df.apply(check_brand_usage, axis=1)
    
    # 4. BOOLEAN FLAGS (The Missing Features!)
    # These match the regex logic from your notebook
    df['has_html'] = df['url'].str.contains(r'\.html?', case=False, regex=True).astype(int)
    df['has_query_param'] = df['url'].str.contains(r'\?query=', case=False, regex=True).astype(int)
    df['has_https'] = df['url'].str.contains(r'^https://', case=False, regex=True).astype(int)
    df['has_http'] = df['url'].str.contains(r'^http://', case=False, regex=True).astype(int)
    df['has_ip_address'] = df['url'].str.contains(r'://(?:\d{1,3}\.){3}\d{1,3}', case=False, regex=True).astype(int)
    
    suspicious_kw_str = '|'.join(['login', 'secure', 'payment', 'verify'])
    df['has_suspicious_kw'] = df['url'].str.contains(suspicious_kw_str, case=False, regex=True).astype(int)
    
    # Check for brands in URL
    brand_pattern = '|'.join(BRANDS)
    df['has_brand_kw'] = df['url'].str.contains(brand_pattern, case=False, regex=True).astype(int)
    
    df['has_non_ascii_chars'] = df['url'].str.contains(r'[^\x00-\x7F]', regex=True).astype(int)

    # --- 5. CREATE FEATURE SETS ---
    
    # FULL LIST: This must match the columns in your X_num_train exactly
    # I've added the missing ones to this list
    nn_cols = ['url_length', 'num_digits', 'num_periods', 'num_slashes', 'num_ats', 'digit_len_ratio', 
               'is_suspicious_brand_usage', 'has_html', 'has_query_param', 'has_https', 'has_http', 
               'has_ip_address', 'has_suspicious_kw', 'has_brand_kw', 'has_non_ascii_chars', 'entropy']
    
    # Reorder columns to match training
    # We use .reindex to ignore any extra columns and fill missing ones with 0 just in case
    X_nn = df.reindex(columns=nn_cols, fill_value=0)
    
    # Set B: XGBoost Features
    # If XGBoost was trained on the same data, it uses the same columns (minus typosquat if you dropped it)
    #xgb_cols = [c for c in nn_cols if c not in ['is_typosquat', 'min_brand_dist']]
    xgb_cols = ['url_length', 'num_digits', 'num_periods', 'num_slashes', 'num_ats',
       'min_brand_dist', 'is_typosquat', 'digit_len_ratio',
       'is_suspicious_brand_usage', 'has_html', 'has_query_param', 'has_https',
       'has_http', 'has_ip_address', 'has_suspicious_kw', 'has_brand_kw',
       'has_non_ascii_chars', 'entropy']
    X_xgb = df.reindex(columns=xgb_cols, fill_value=0)
    
    return X_nn, X_xgb

url_input = st.text_input("Enter URL:", placeholder="http://example.com")

# --- 4. UI & LOGIC ---
if st.button("Scan URL"):
    if not url_input:
        st.warning("Please enter a URL.")
    else:
        with st.spinner("Analyzing..."):
            try:
                # Get raw DataFrames
                X_nn_raw, X_xgb_raw = extract_features(url_input)
                
                # --- STAGE 1: XGBoost (The Gatekeeper) ---
                # Note: XGBoost usually handles raw unscaled data fine, 
                # but if you trained on SCALED data, you must scale it.
                # Assuming you trained XGBoost on the SCALED subset:
                # We need a separate scaler for XGBoost if the columns differ. 
                # If you don't have one, try running on raw data (often works for trees).
                xgb_pred = xgb_model.predict(X_xgb_raw)[0]
                xgb_prob = xgb_model.predict_proba(X_xgb_raw)[0][1]
                
                # --- STAGE 2: Neural Network (The Specialist) ---
                # 1. Scale Numerical Data
                X_num_scaled = scaler.transform(X_nn_raw).astype('float32')
                
                # 2. Tokenize Text Data
                seq = tokenizer.texts_to_sequences([url_input])
                X_text = pad_sequences(seq, maxlen=200, padding='post', truncating='post')
                
                # 3. Predict
                nn_probs = nn_model.predict([X_text, X_num_scaled])
                nn_class_idx = np.argmax(nn_probs)
                nn_label = encoder.inverse_transform([nn_class_idx])[0]
                nn_conf = np.max(nn_probs)
                
                # --- RESULTS ---
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("XGBoost Model")
                    if xgb_pred == 1:
                        st.error("🚨 Flagged Malicious")
                    else:
                        st.success("✅ Flagged Safe")
                    st.metric("Threat Probability", f"{xgb_prob:.2%}")
                    
                with col2:
                    st.subheader("Neural Net Specialist")
                    st.info(f"Classified as: **{nn_label.upper()}**")
                    st.metric("Confidence", f"{nn_conf:.2%}")
                    
                st.divider()
                st.write("### Model Agreement Analysis")
                if xgb_pred == 0 and nn_label == 'benign':
                    st.success("Both models agree this URL is Safe.")
                elif xgb_pred == 1 and nn_label != 'benign':
                    st.error(f"Critical Alert: Both models detected a threat ({nn_label}). Do not visit.")
                else:
                    st.warning("Models Disagree. Proceed with caution.")
                    
            except Exception as e:
                st.error(f"Prediction Error: {e}")
                st.write("Debug info - NN Columns expected:", list(X_nn_raw.columns))