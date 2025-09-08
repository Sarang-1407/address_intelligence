import streamlit as st
import spacy
from spacy.lang.en import Language
from spacy.tokens import Span
from spacy.util import filter_spans
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import string
import re
import ast
import json
import time
import glob
from datetime import datetime
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Address Intelligence System",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1e88e5;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #1e88e5;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .score-card {
        text-align: center;
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        font-size: 1.2rem;
        font-weight: bold;
    }
    .high-match {
        background: linear-gradient(135deg, #4caf50, #45a049);
        color: white;
    }
    .medium-match {
        background: linear-gradient(135deg, #ff9800, #f57c00);
        color: white;
    }
    .low-match {
        background: linear-gradient(135deg, #f44336, #d32f2f);
        color: white;
    }
    .component-tag {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        color: #0d47a1;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        display: inline-block;
        font-size: 0.9rem;
        font-weight: 500;
        border: 1px solid #2196f3;
    }
    .analysis-section {
        background-color: #fafafa;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
    }
    .completeness-card {
        background: linear-gradient(135deg, #e8f5e8 0%, #d4edda 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #28a745;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# =========================
# Global Variables
# =========================
cities = []
states = []
pincodes = []
area_names = []
state_abbv = []

# =========================
# Address Completeness Logic
# =========================
def catch_json(js):
    js = str(js)
    try:
        return ast.literal_eval(js)
    except:
        try:
            return json.loads(js)
        except:
            return None

def clean_address(address):
    """Clean and preprocess address text"""
    address = str(address) if address else ""
    
    #Rule1: remove urls
    address = re.sub(r'(https?://\S+)', "", address)
    #Rule2: replace "escape chars" by "space"
    address = re.sub("[\n\t\r]", " ", address)
    #Rule3: replace 'apostrophe s' by 's'
    address = re.sub("[\'\"\`]s", "s", address)
    #Rule4: remove single/double quotes having space on either side
    address = re.sub("[\'\"] ", " ", address)
    address = re.sub(" [\'\"]", " ", address)
    #Rule5: replace "single/double quotes surrounded by multiple alphabets on both sides" by "space"
    address = re.sub("[a-zA-Z]{2,}[\'\"][a-zA-Z]{2,}", " ", address)
    #Rule6: replace 'equal to', 'colon', 'tilde' by  'hyphen'
    address = re.sub("[\=\:\~]", "-", address)
    #Rule7: replace 'square and curly brackets' by  'round brackets'
    address = re.sub("[\[\{]", "(", address)
    address = re.sub("[\]\}]", ")", address)
    #Rule8: replace 'pipe and backslash' by  'forward slash'
    address = re.sub("[\|\\\]", "/", address)
    #Rule9: replace 'semicolon and question mark' by  'comma'
    address = re.sub("[;\?]", ",", address)
    #Rule10: replace ' ! $ @ * % < > _ ^' by  'space'
    address = re.sub("[\!\$@\*%\<\>_\^]", " ", address)
    #Rule10: replace repeated special chars by single char
    address = re.sub(",+", ",", address)
    address = re.sub("\.+", ".", address)
    address = re.sub("\++", "+", address)
    address = re.sub("\-+", "-", address)
    address = re.sub("\(+", "(", address)
    address = re.sub("\)+", ")", address)
    address = re.sub("\&+", "&", address)
    address = re.sub("\#+", "#", address)
    address = re.sub("\/+", "/", address)
    address = re.sub("\"+", '"', address)
    address = re.sub("\'+", "'", address)
    address = re.sub(" +", " ", address)
    #Rule11: remove special chars from start and end of string
    address = address.strip()
    address = re.sub("^[\.\,\-\+\/\)]", "", address)
    address = re.sub("[\.\,\-\+\/\(]$", "", address)
    address = address.strip()
    #Rule12: replace special_character by space + special_character from end of individual tokens
    address_ = []
    for add_string in address.split():    
        match_ = re.search("[\\\\.,;:\\-_]+$", add_string)
        if match_:
            add_string = re.sub("[\\\\.,;:\\-_]+$", " " + match_.group(0), add_string)
        address_.append(add_string)
    address = ' '.join(address_)
    address = address.lower()
    return address

@Language.component("expand_entities")
def expand_entities(doc):
    """Custom component to expand and refine entity labels"""
    def new_entities(doc, ent, prev_ent, prev_mod=False):
        street_suffix_keywords = ['road', 'street', 'lane', 'rd', 'marg', 'gali', 'cross']
        area_suffix_keywords = ['village', 'chowk', 'bazar', 'market', 'nagar', 'mohalla',
                                'puram', 'vihar', 'sarai']
        
        # add word or entity before suffix to single entity
        if ent.text in street_suffix_keywords and ent.start != 0:
            if prev_ent and not prev_mod:
                new_ent = Span(doc, prev_ent.start, ent.end, label='street_name')
            else:
                new_ent = Span(doc, ent.start - 1, ent.end, label='street_name')
            return new_ent
        elif ent.text in area_suffix_keywords and ent.start != 0:
            new_ent = Span(doc, ent.start - 1, ent.end, label='area_name')
            return new_ent
        elif re.search("^[0-9]{6}$", ent.text):
            ent.label_ = 'area_pincode'
            return ent
        elif len(ent.text) != 6 and not re.search("[^0-9]", ent.text) and ent.label_ != 'unit':
            ent.label_ = 'unassigned'
            return ent
        elif ent.text in cities:
            ent.label_ = 'city_name'
            return ent
        elif ent.text in states:
            ent.label_ = 'state_name'
            return ent
        else:
            return ent
    
    old_ents = doc.ents
    new_ents = []
    prev_ent = None
    mod = False
    for ent in doc.ents:
        ent_new = new_entities(doc, ent, prev_ent, mod)
        new_ents.append(ent_new)
        if ent.text != ent_new.text:
            mod = True
        else:
            mod = False
        prev_ent = ent
    
    doc.ents = filter_spans(new_ents + list(old_ents))
    return doc

def snake2camel(string):
    temp = string.split('_')
    res = temp[0] + ''.join(ele.title() for ele in temp[1:])
    return res

def has_digit(doc):
    try:
        for token in doc:
            if token.ent_type_ == 'unit':
                if any(char.isdigit() for char in token.text):
                    return 1
        return 0
    except Exception as e:
        print(f'Error checking for digits: {e}')
        return 0

MANDATORY = ["city_name", "state_name", "area_pincode"]

def unit_location_factor(doc):
    try:
        positions = [ent.start / max(len(doc), 1) for ent in doc.ents if ent.label_ == "unit"]
        if not positions:
            return -1
        first_pos = min(positions)
        if first_pos < 0.3:
            return 1
        if first_pos < 0.5:
            return 0
        return -1
    except Exception as e:
        print(f"Error calculating unit location factor: {e}")
        return 0

def _presence(ents, label):
    return 1 if ents.get(label) else 0

def _combos(ents):
    return {
        "combo.society_area": 1 if _presence(ents,"society_name") and _presence(ents,"area_name") else 0,
        "combo.street_landmark": 1 if _presence(ents,"street_name") and _presence(ents,"landmark") else 0,
    }

def _penalties(doc, ents):
    # Unassigned ratio
    total_tokens = max(len(doc), 1)
    unassigned_texts = ents.get("unassigned", [])
    unassigned_len = sum(len(s.split()) for s in unassigned_texts)
    unassigned_ratio = unassigned_len / total_tokens
    penalties = {
        "penalty.unassigned_high": 1 if unassigned_ratio > 0.4 else 0,
        "penalty.missing_mandatory": 1 if any(not _presence(ents, m) for m in MANDATORY) else 0,
    }
    return penalties, unassigned_ratio

# Weights
WEIGHTS = {
    # presence
    "unit.pres": 6, "street.pres": 6, "area.pres": 5, "society.pres": 4,
    "city.pres": 6, "state.pres": 4, "pincode.pres": 8,
    # unit quality
    "unit.has_digit": 2, "unit.early": 2,
    # combos
    "combo.society_area": 2, "combo.street_landmark": 2,
    # penalties
    "penalty.unassigned_high": -4, "penalty.missing_mandatory": -8,
}

def completeness_score(shippingAddress1, shippingAddress2, nlp, verbose=False):
    """Calculate address completeness score"""
    try:
        # 1) Normalize & parse
        address = (shippingAddress1 or "") + " " + (shippingAddress2 or "")
        address = " ".join(address.split())
        address = clean_address(address)
        doc = nlp(address)
        
        if verbose:
            print("Parsing address entities")
            for ent in doc.ents:
                print(f"Entity: {ent.text} - {ent.label_}")
        
        # 2) Collect ents into dict-of-lists
        labels = [
            "unit","street_name","society_name","area_name","city_name",
            "area_pincode","landmark","state_name","unassigned",
        ]
        ents = {k: [] for k in labels}
        for ent in doc.ents:
            if ent.label_ in ents:
                ents[ent.label_].append(ent.text)
            else:
                ents["unassigned"].append(ent.text)
        
        # 3) Feature extraction
        feats = {
            "unit.pres": _presence(ents, "unit"),
            "street.pres": _presence(ents, "street_name"),
            "area.pres": _presence(ents, "area_name"),
            "society.pres": _presence(ents, "society_name"),
            "city.pres": _presence(ents, "city_name"),
            "state.pres": _presence(ents, "state_name"),
            "pincode.pres": _presence(ents, "area_pincode"),
        }
        
        # Unit quality (only if unit present)
        if feats["unit.pres"]:
            units_text = " ".join(ents.get("unit", []))
            has_digit_flag = 1 if any(ch.isdigit() for ch in units_text) else 0
            unit_loc = unit_location_factor(doc)
            early_flag = 1 if unit_loc == 1 else 0
        else:
            has_digit_flag = 0
            early_flag = 0
        
        feats["unit.has_digit"] = has_digit_flag
        feats["unit.early"] = early_flag
        
        # Combos
        feats.update(_combos(ents))
        
        # Penalties
        penalties, unassigned_ratio = _penalties(doc, ents)
        feats.update(penalties)
        
        # 4) Score computation
        raw = sum(WEIGHTS[k] * v for k, v in feats.items())
        max_pos = sum(w for k, w in WEIGHTS.items() if not k.startswith("penalty"))
        scaled_score = max(0.0, min(100.0, 100.0 * max(0.0, raw) / max(1.0, max_pos)))
        
        # 5) Insights
        insights = []
        if not feats["unit.pres"]:
            insights.append("Missing flat/house/unit number (e.g., 'Flat 12B').")
        if not feats["street.pres"]:
            insights.append("Missing street/road (e.g., 'MG Road').")
        if not feats["area.pres"] and not feats["society.pres"]:
            insights.append("Missing locality/area or society name.")
        if not feats["city.pres"]:
            insights.append("Missing city (e.g., 'Bengaluru').")
        if not feats["state.pres"]:
            insights.append("Missing state (e.g., 'Karnataka').")
        if not feats["pincode.pres"]:
            insights.append("Missing 6-digit pincode.")
        if feats["unit.pres"] and not feats["unit.has_digit"]:
            insights.append("Unit found but no number detected—include house/flat number if available.")
        if unassigned_ratio > 0.4:
            insights.append("Address has too much unrecognized text—simplify or remove extra descriptors.")
        
        unit_pos_hint = unit_location_factor(doc)
        if feats["unit.pres"] and unit_pos_hint == -1:
            insights.append("Place unit/house number earlier in the address (before street/area).")
        
        # 6) Build response
        response = {
            "clean_address": address,
            "address_completeness_score": scaled_score,
            "address_insights": "\n".join(insights),
        }
        response.update(ents)
        response = {snake2camel(k): v for k, v in response.items()}
        
        # Ensure camelCase keys for UI compatibility
        response['addressCompletenessScore'] = response.get('addressCompletenessScore', scaled_score)
        response['cleanAddress'] = response.get('cleanAddress', address)
        
        return response
    except Exception as e:
        print(f"Error calculating address completeness score: {e}")
        return {
            'cleanAddress': clean_address((shippingAddress1 or "") + " " + (shippingAddress2 or "")),
            'addressCompletenessScore': 0,
            'addressInsights': 'Error processing address'
        }

# =========================
# Address Similarity Logic
# =========================
def combine_consecutive_single_characters(name):
    name_parts = name.split()
    result = []
    i = 0
    while i < len(name_parts):
        consecutive_singles = []
        while i < len(name_parts) and len(name_parts[i]) == 1:
            consecutive_singles.append(name_parts[i])
            i += 1
        if consecutive_singles:
            result.append(''.join(consecutive_singles))
        if i < len(name_parts):
            result.append(name_parts[i])
            i += 1
    return " ".join(result)

ABBR_MAP = {
    r"\bapt\.?\b": "apartment",
    r"\bfl(?:at)?\b": "flat",
    r"\bno\.?\b": "number",
    r"\bh\.?no\.?\b": "number",
    r"\bblk\b": "block",
    r"\bsec\b": "sector",
    r"\bph\.?\b": "phase",
    r"\brd\.?\b": "road",
    r"\bste\b": "suite",
    r"\bopp\.?\b": "opposite",
    r"\bnr\.?\b": "near",
    r"\bpo\b": "postoffice",
    r"\bps\b": "policestation",
}

PIN_RE = re.compile(r"\b[1-9]\d{5}\b")

def normalize_address_text(s):
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    for pat, repl in ABBR_MAP.items():
        s = re.sub(pat, repl, s)
    s = " ".join(s.split())
    try:
        s = combine_consecutive_single_characters(s)
    except:
        pass
    return s

def extract_pincode(text):
    m = PIN_RE.search(text or "")
    return m.group(0) if m else None

def bigrams_of_words(s):
    toks = s.split()
    out = []
    for t in toks:
        out.extend([t[i:i+2] for i in range(len(t)-1)])
    return out

def dice_coeff_fast(bigrams_a, bigrams_b):
    if not bigrams_a and not bigrams_b:
        return 0.0
    ca, cb = Counter(bigrams_a), Counter(bigrams_b)
    inter = sum(min(ca[k], cb.get(k, 0)) for k in ca)
    union = len(bigrams_a) + len(bigrams_b)
    return (200.0 * inter) / union if union else 0.0

def diceSimilarity(base_string, target_string):
    a = normalize_address_text(base_string)
    b = normalize_address_text(target_string)
    return dice_coeff_fast(bigrams_of_words(a), bigrams_of_words(b))

def norm_group_texts(vs):
    return [normalize_address_text(v) for v in (vs or [])]

def compare_component_groups(group1, group2):
    """Return best Dice similarity (0..100). If either group empty -> -1.0"""
    g1 = norm_group_texts(group1)
    g2 = norm_group_texts(group2)
    if not g1 or not g2:
        return -1.0
    best = 0.0
    for a in g1:
        for b in g2:
            best = max(best, diceSimilarity(a, b))
    return best

def pin_from_ents(ents):
    if not ents:
        return None
    text = " ".join(ents.get("areaPincode", []) or [])
    pin = extract_pincode(text)
    return pin

def pincode_similarity(pin1, pin2):
    """Soft scoring to tolerate neighbors/typos. Returns 0..100."""
    if not pin1 and not pin2:
        return 50.0
    if not pin1 or not pin2:
        return 50.0
    if pin1 == pin2:
        return 100.0
    
    if len(pin1) == len(pin2) == 6:
        mismatches = sum(c1 != c2 for c1, c2 in zip(pin1, pin2))
        if mismatches == 1:
            return 92.0
        if pin1[:5] == pin2[:5]:
            return 96.0
        if pin1[:4] == pin2[:4]:
            return 92.0
        if pin1[:3] == pin2[:3]:
            return 85.0
    return 40.0

def address_similarity_from_ents(ents1, ents2, threshold=70.0, return_components=False):
    """Calculate address similarity from parsed entities"""
    # component sims
    sim = {}
    sim["unit"] = max(0.0, compare_component_groups(ents1.get("unit", []), ents2.get("unit", [])))
    sim["street_name"] = max(0.0, compare_component_groups(ents1.get("streetName", []), ents2.get("streetName", [])))
    
    # locality = max(area_name, society_name)
    area_sim = max(
        max(0.0, compare_component_groups(ents1.get("areaName", []), ents2.get("areaName", []))),
        max(0.0, compare_component_groups(ents1.get("societyName", []), ents2.get("societyName", []))),
    )
    sim["locality"] = area_sim
    sim["city_name"] = max(0.0, compare_component_groups(ents1.get("cityName", []), ents2.get("cityName", [])))
    sim["state_name"] = max(0.0, compare_component_groups(ents1.get("stateName", []), ents2.get("stateName", [])))
    sim["landmark"] = max(0.0, compare_component_groups(ents1.get("landmark", []), ents2.get("landmark", [])))
    
    # soft pincode
    pin1, pin2 = pin_from_ents(ents1), pin_from_ents(ents2)
    sim["pincode"] = pincode_similarity(pin1, pin2)
    
    # weights
    W = {
        "pincode": 0.20,
        "street_name": 0.20,
        "locality": 0.20,
        "unit": 0.15,
        "city_name": 0.15,
        "state_name": 0.05,
        "landmark": 0.05,
    }
    
    score = 0.0
    for k, w in W.items():
        score += w * sim.get(k, 0.0)
    
    is_match = score >= threshold
    details = {"address_similarity_score": score, "component_match_scores": sim}
    
    if return_components:
        details["base_components"] = ents1
        details["target_components"] = ents2
    
    return details

def comprehensive_address_matching(base_address, target_address, nlp_model):
    """Main function to combine completeness and similarity analysis"""
    # Parse addresses for completeness and component extraction
    base_parsed = completeness_score(base_address, '', nlp_model, verbose=False)
    target_parsed = completeness_score(target_address, '', nlp_model, verbose=False)
    
    # Calculate similarity
    similarity_result = address_similarity_from_ents(
        base_parsed, target_parsed, threshold=70.0, return_components=True
    )
    
    return {
        'similarity_score': similarity_result['address_similarity_score'],
        'base_analysis': base_parsed,
        'target_analysis': target_parsed,
        'component_scores': similarity_result['component_match_scores'],
        'is_match': similarity_result['address_similarity_score'] >= 70.0,
        'processed_addresses': {
            'base': normalize_address_text(base_address),
            'target': normalize_address_text(target_address)
        }
    }

# =========================
# Model Loading
# =========================
@st.cache_resource
def load_nlp_model():
    try:
        # Try to load custom model first
        model_patterns = ["entity_rules_ner_*", "address_ner_model_*", "*address*model*"]
        found_models = []
        for pattern in model_patterns:
            found_models.extend(glob.glob(pattern))

        if found_models:
            latest_model = sorted(found_models)[-1]
            try:
                nlp = spacy.load(latest_model)
                # Check if expand_entities component already exists
                if "expand_entities" not in nlp.pipe_names:
                    if "ner" in nlp.pipe_names:
                        nlp.add_pipe("expand_entities", after="ner")
                    else:
                        nlp.add_pipe("expand_entities", last=True)
                st.success(f"Custom model loaded: {latest_model}")
                return nlp, latest_model
            except Exception as e:
                st.warning(f"Could not load custom model {latest_model}: {e}")
                st.info("Falling back to basic spaCy model...")

        # Fallback models
        for model_name in ["en_core_web_sm", "en_core_web_md"]:
            try:
                nlp = spacy.load(model_name)
                # Check if expand_entities component already exists
                if "expand_entities" not in nlp.pipe_names:
                    if "ner" in nlp.pipe_names:
                        nlp.add_pipe("expand_entities", after="ner")
                    else:
                        nlp.add_pipe("expand_entities", last=True)
                st.info(f"Using basic spaCy model: {model_name}")
                return nlp, model_name
            except Exception:
                continue

        st.error("No spaCy model found. Please install a basic model:")
        st.code("python -m spacy download en_core_web_sm")
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# =========================
# UI Functions
# =========================
def display_address_analysis(analysis, title):
    """Display address completeness analysis"""
    with st.container():
        st.markdown(f'<div class="analysis-section">', unsafe_allow_html=True)
        st.markdown(f"**{title}**")
        
        completeness = analysis.get('addressCompletenessScore', 0)
        
        # Color-coded completeness score
        if completeness >= 80:
            color = "#28a745"  # Green
            status = "Excellent"
        elif completeness >= 60:
            color = "#ffc107"  # Yellow
            status = "Good"
        elif completeness >= 40:
            color = "#fd7e14"  # Orange
            status = "Fair"
        else:
            color = "#dc3545"  # Red
            status = "Poor"
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Completeness Score", f"{completeness:.1f}%", help=f"Status: {status}")
        
        with col2:
            # Progress bar
            st.markdown(f"""
            <div style="background-color: #f0f0f0; border-radius: 10px; height: 20px; margin: 10px 0;">
                <div style="background-color: {color}; width: {completeness}%; height: 100%; border-radius: 10px; 
                           display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">
                    {completeness:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Components found
        st.markdown("**Components Found:**")
        component_count = 0
        for component, values in analysis.items():
            if component not in ['addressCompletenessScore', 'cleanAddress', 'addressInsights'] and isinstance(values, list) and values:
                for value in values:
                    st.markdown(f'<span class="component-tag">{component}: {value}</span>', unsafe_allow_html=True)
                    component_count += 1
        
        if component_count == 0:
            st.info("No components automatically detected")
        
        # Insights
        insights = analysis.get('addressInsights', '')
        if insights:
            with st.expander("💡 Improvement Suggestions"):
                for insight in insights.split('\n'):
                    if insight.strip():
                        st.write(f"• {insight}")
        
        st.markdown('</div>', unsafe_allow_html=True)

def create_component_comparison_chart(component_scores):
    """Create a radar chart for component comparison"""
    components = list(component_scores.keys())
    scores = [max(0, score) for score in component_scores.values()]  # Ensure non-negative
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=components,
        fill='toself',
        name='Component Similarity Scores',
        line_color='#1e88e5'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=True,
        title="Component-wise Similarity Analysis"
    )
    
    return fig

def create_similarity_bar_chart(component_scores):
    """Create a bar chart for component scores"""
    components = list(component_scores.keys())
    scores = [max(0, score) for score in component_scores.values()]
    
    # Color code based on score
    colors = ['#28a745' if s >= 70 else '#ffc107' if s >= 40 else '#dc3545' for s in scores]
    
    fig = go.Figure(data=[
        go.Bar(
            x=components,
            y=scores,
            marker_color=colors,
            text=[f'{s:.1f}%' for s in scores],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Component Similarity Scores",
        xaxis_title="Address Components",
        yaxis_title="Similarity Score (%)",
        yaxis=dict(range=[0, 100])
    )
    
    return fig

# =========================
# Main Application
# =========================
def main():
    st.markdown('<h1 class="main-header">Address Intelligence System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Advanced NLP-powered Address Analysis and Matching</p>', unsafe_allow_html=True)

    with st.spinner("Loading AI model..."):
        nlp_model, model_name = load_nlp_model()

    if nlp_model is None:
        st.error("Failed to load model. Please check your setup.")
        st.stop()

    # Sidebar Configuration
    with st.sidebar:
        st.header("Configuration")
        with st.expander("Model Information"):
            st.success(f"Model: {model_name}")
            st.info("Model loaded successfully")

        st.subheader("Analysis Settings")
        threshold = st.slider("Match Threshold (%)", 0, 100, 70, 5)
        show_detailed = st.checkbox("Show Detailed Breakdown", True)
        show_preprocessing = st.checkbox("Show Preprocessing Steps", False)
        show_completeness = st.checkbox("Show Completeness Analysis", True)

        st.subheader("Sample Test Cases")
        sample_addresses = {
            "High Match Example": {
                "base": "HIG/B-24, Indra Puram, shamshabad road, Near water tank, agra - 282002",
                "target": "H I G / B - 24, Indra Puram, shamshabad road, Near water tank, agra - 282002"
            },
            "Medium Match Example": {
                "base": "Flat 12B, MG Road, Koramangala, Bangalore - 560034",
                "target": "MG Road, Koramangala, Bangalore - 560034"
            },
            "Low Match Example": {
                "base": "123 Main Street, Mumbai - 400001",
                "target": "456 Oak Avenue, Chennai - 600001"
            },
            "Pincode Mismatch": {
                "base": "HIG/B-24, Indra Puram, shamshabad road, Near water tank, agra - 282002",
                "target": "HIG/B-24, Indra Puram, shamshabad road, Near water tank, agartala - 282011"
            }
        }
        
        selected_sample = st.selectbox("Load Sample Addresses:", ["Custom Input"] + list(sample_addresses.keys()))
        if st.button("Load Selected Sample"):
            if selected_sample != "Custom Input":
                st.session_state.sample_loaded = selected_sample

    # Address Input Section
    st.header("Address Input")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Base Address")
        default_base = sample_addresses[st.session_state.sample_loaded]["base"] if 'sample_loaded' in st.session_state and st.session_state.sample_loaded != "Custom Input" else ""
        base_address = st.text_area("Enter the first address:", value=default_base, height=120, key="base_addr")

    with col2:
        st.subheader("Target Address")
        default_target = sample_addresses[st.session_state.sample_loaded]["target"] if 'sample_loaded' in st.session_state and st.session_state.sample_loaded != "Custom Input" else ""
        target_address = st.text_area("Enter the second address:", value=default_target, height=120, key="target_addr")

    # Analysis Button and Results
    if st.button("Analyze Address Similarity", type="primary", use_container_width=True):
        if not base_address.strip() or not target_address.strip():
            st.error("Please enter both addresses to proceed with analysis.")
            return

        # Progress indicator
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Step 1/4: Preprocessing addresses...")
        progress_bar.progress(25)
        time.sleep(0.3)

        status_text.text("Step 2/4: Analyzing address components...")
        progress_bar.progress(50)
        time.sleep(0.3)

        status_text.text("Step 3/4: Calculating similarity...")
        progress_bar.progress(75)

        # Main analysis
        result = comprehensive_address_matching(base_address, target_address, nlp_model)

        status_text.text("Step 4/4: Generating results...")
        progress_bar.progress(100)
        time.sleep(0.2)
        progress_bar.empty()
        status_text.empty()

        # Results Display
        st.header("Analysis Results")

        # Main similarity score card
        similarity_score = result['similarity_score']
        if similarity_score >= threshold:
            score_class = "high-match"
            decision = "MATCH FOUND"
            decision_icon = "✅"
        elif similarity_score >= 40:
            score_class = "medium-match"
            decision = "PARTIAL MATCH"
            decision_icon = "⚠️"
        else:
            score_class = "low-match"
            decision = "NO MATCH"
            decision_icon = "❌"

        st.markdown(f"""
        <div class="score-card {score_class}">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">{decision_icon}</div>
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{decision}</div>
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">{similarity_score:.1f}%</div>
            <div style="font-size: 1rem; opacity: 0.9;">Similarity Score</div>
        </div>
        """, unsafe_allow_html=True)

        # Preprocessing Steps (Optional)
        if show_preprocessing:
            st.subheader("Preprocessing Steps")
            c1, c2 = st.columns(2)
            with c1:
                st.text("Original Base Address:")
                st.code(base_address)
                st.text("Processed Base Address:")
                st.code(result['processed_addresses']['base'])
            with c2:
                st.text("Original Target Address:")
                st.code(target_address)
                st.text("Processed Target Address:")
                st.code(result['processed_addresses']['target'])

        # Address Completeness Analysis (Optional)
        if show_completeness:
            st.subheader("Address Completeness Analysis")
            c1, c2 = st.columns(2)
            
            with c1:
                display_address_analysis(result['base_analysis'], "Base Address Analysis")
            
            with c2:
                display_address_analysis(result['target_analysis'], "Target Address Analysis")

        # Component Similarity Analysis
        st.subheader("Component Similarity Analysis")
        
        # Filter out negative scores for display
        valid_components = {k: v for k, v in result['component_scores'].items() if v >= 0}
        
        if valid_components:
            # Create two columns for charts
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                fig_radar = create_component_comparison_chart(valid_components)
                st.plotly_chart(fig_radar, use_container_width=True)
            
            with chart_col2:
                fig_bar = create_similarity_bar_chart(valid_components)
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # Component scores table
            if show_detailed:
                st.subheader("Detailed Component Scores")
                
                component_data = []
                for component, score in result['component_scores'].items():
                    if score >= 0:  # Only show components that were compared
                        status = "High Match" if score >= 70 else "Medium Match" if score >= 40 else "Low Match"
                        component_data.append({
                            'Component': component.replace('_', ' ').title(),
                            'Similarity Score': f"{score:.1f}%",
                            'Status': status
                        })
                
                if component_data:
                    df_components = pd.DataFrame(component_data)
                    st.dataframe(df_components, use_container_width=True)
                else:
                    st.info("No component matches found for detailed breakdown")
        else:
            st.info("No valid component comparisons could be made between the addresses")

        # Decision Summary
        st.subheader("Matching Decision Summary")
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.metric("Final Score", f"{similarity_score:.1f}%")
        with c2:
            st.metric("Threshold", f"{threshold}%")
        with c3:
            if similarity_score >= threshold:
                st.success("MATCH")
            else:
                st.error("NO MATCH")

        # Export functionality
        if st.button("Export Results as JSON"):
            results_json = json.dumps(result, indent=2, default=str)
            st.download_button(
                label="Download Results",
                data=results_json,
                file_name=f"address_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )


if __name__ == "__main__":
    main()