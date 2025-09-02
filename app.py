import streamlit as st
import spacy
from spacy.lang.en import Language
from spacy.tokens import Span
from spacy.util import filter_spans
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import string
import re
import ast
import json
import time
import glob
from datetime import datetime

# =========================
# Streamlit Page Config/CSS
# =========================
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
</style>
""", unsafe_allow_html=True)

# =========================
# Globals for optional rules
# =========================
cities = []
states = []
pincodes = []
area_names = []
state_abbv = []

# =========================
# Helpers / Text Cleaning
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

def clean_address(address: str) -> str:
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
    #Rule10: replace '` ! $ @ * % < > _ ^' by  'space'
    address = re.sub("[`\!\$@\*%\<\>_\^]", " ", address)
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
    def new_entities(doc, ent, prev_ent, prev_mod=False):
        street_suffix_keywords = ['road', 'street', 'lane', 'rd', 'marg', 'gali', 'cross']
        area_suffix_keywords = ['village', 'chowk', 'bazar', 'market', 'nagar', 'mohalla', 'puram', 'vihar', 'sarai']

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
        mod = (ent.text != ent_new.text)
        prev_ent = ent

    doc.ents = filter_spans(new_ents + list(old_ents))
    return doc

def removePunctuation(s: str) -> str:
    for ch in string.punctuation:
        s = s.replace(ch, '')
    return s

def getBigrams(s): return [s[i:i+2] for i in range(len(s) - 1)]

def tokenize(string_):
    string_ = map(lambda word: word.lower(), string_.split())
    bigrams = []
    for s in string_:
        bigrams.extend(getBigrams(s))
    return bigrams

def diceCoeff(s, t):
    union = len(s) + len(t)
    hit = 0
    t = list(t)  # avoid mutating caller
    for a in s:
        for idx, b in enumerate(t):
            if a == b:
                hit += 1
                del t[idx]
                break
    return (200.0 * hit) / union if union != 0 else 0

def combine_consecutive_single_characters(name: str) -> str:
    for ch in string.punctuation:
        name = name.replace(ch, '')
        
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

def diceSimilarity(base_string, target_string):
    base_string = combine_consecutive_single_characters(removePunctuation(base_string)).lower()
    tokenizeBase = tokenize(base_string)
    target_string = combine_consecutive_single_characters(removePunctuation(target_string)).lower()
    tokenizeTarget = tokenize(target_string)
    score = diceCoeff(tokenizeBase, tokenizeTarget)
    return score

# =========================
# ORIGINAL (Code A) Score
# =========================
def snake2camel(string_):
    temp = string_.split('_')
    return temp[0] + ''.join(ele.title() for ele in temp[1:])

def has_digit(doc):
    try:
        for token in doc:
            if token.ent_type_ == 'unit':
                if any(char.isdigit() for char in token.text):
                    return 1
        return 0
    except Exception:
        return 0

def ner_confidence(doc):
    try:
        ent_dic = {}
        for token in doc:
            ent_dic[token] = token.ent_type_

        import string as _string
        def ispunct(ch): return ch in _string.punctuation

        ner_confidence_value = sum([(1 if len(i) > 0 else 0) for i in ent_dic.values()]) / max(
            1, sum([(1 if not ispunct(i.text) else 0) for i, _ in ent_dic.items()])
        )

        def bucket(v):
            if v >= 0.7: return 1
            if v >= 0.5: return 0.7
            if v >= 0.3: return 0.5
            return 0.3

        return bucket(ner_confidence_value)
    except Exception:
        return 0.3

def unit_location_factor(doc):
    try:
        unit_location_list = [token.i / len(doc) for token in doc if token.ent_type_ == 'unit']
        min_ = np.min(unit_location_list) if unit_location_list else 0.4
        if min_ < 0.3: return 1
        if min_ < 0.5: return 0
        return -1
    except Exception:
        return 0

def completeness_score(address1, address2, nlp, verbose=False):
    try:
        address = (address1 or "") + ' ' + (address2 or "")
        address = ' '.join(address.split())
        address = clean_address(address)

        doc = nlp(address)
        if verbose:
            print('Parsing address entities')
            for ent in doc.ents:
                print(f'Entity: {ent} - {ent.label_}')

        tags = [ent.label_ for ent in doc.ents]
        _ = ner_confidence(doc)  # available if you later want to surface it

        labels_ = [
            'unit',
            'street_name',
            'society_name',
            'area_name',
            'city_name',
            'area_pincode',
            'landmark',
            'state_name',
            'unassigned',
        ]
        labels_ = {k: [] for k in labels_}
        for ent in doc.ents:
            labels_[ent.label_].append(ent.text)

        weights = {
            'unit': 8,
            'landmark': 10,
            'street_name': 8,
            'society_name': 6,
            'area_name': 5,
        }

        unit_score = 0
        unit_has_digit = has_digit(doc) * 2
        unit_loc_factor = unit_location_factor(doc) * 2

        unit_found = 1 if 'unit' in tags else 0
        if unit_found:
            unit_score = weights['unit'] + unit_has_digit  # + unit_loc_factor (kept disabled to mirror script)

        landmark_found = 1 if 'landmark' in tags else 0
        landmark_score = landmark_found * weights['landmark']

        street_found = 1 if 'street_name' in tags else 0
        area_found = 1 if 'area_name' in tags else 0

        insights = []
        if not unit_found:
            insights.append('Unit/Rooftop information ambiguous or not found')
        if not area_found:
            insights.append('Area or society name ambiguous or not found')
        if not street_found:
            insights.append('Street name ambiguous or not found')
        if not landmark_found:
            insights.append('Landmark name ambiguous or not found')

        # core score
        score = max(unit_score, landmark_score)
        for label in ['street_name', 'society_name', 'area_name']:
            tag_found = 1 if label in tags else 0
            this_score = tag_found * weights[label]
            score += this_score

        scaled_score = (score / 29) * 100

        # Build response in camelCase (to match the rest of the app)
        response = {
            'clean_address': address,
            'address_completeness_score': scaled_score,
            'address_insights': '\n'.join(insights),
        }
        response.update(labels_)
        response = {snake2camel(k): v for k, v in response.items()}

        # For UI compatibility keys
        response['addressCompletenessScore'] = response.pop('addressCompletenessScore', scaled_score)
        response['cleanAddress'] = response.pop('cleanAddress', address)

        return response
    except Exception as e:
        print(f'Error calculating address completeness score: {e}')
        return {
            'cleanAddress': clean_address((address1 or "") + " " + (address2 or "")),
            'addressCompletenessScore': 0
        }


def enhanced_compare_component_groups(group1, group2, similarity_func=diceSimilarity):
    if not group1 or not group2:
        return 0.0, None
    max_similarity = 0.0
    best_match = None
    for comp1 in group1:
        for comp2 in group2:
            similarity = similarity_func(comp1, comp2)
            if similarity > 1.0:  # safeguard
                similarity = similarity / 100.0
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = (comp1, comp2)
    return max_similarity, best_match

def hierarchical_address_comparison(base_parsed, target_parsed, similarity_func=diceSimilarity):
    COMPONENT_GROUPS = {
        'last_mile_markers': {'weight': 0.65, 'components': {
            'unit': 0.40, 'societyName': 0.25, 'streetName': 0.20, 'landmark': 0.15}},
        'area_markers': {'weight': 0.20, 'components': {
            'areaName': 0.70, 'areaPincode': 0.30}},
        'broad_markers': {'weight': 0.15, 'components': {
            'cityName': 0.75, 'stateName': 0.25}}
    }

    total_score = 0
    total_possible_weight = 0
    detailed_breakdown = {}

    for group_name, group_info in COMPONENT_GROUPS.items():
        group_weight = group_info['weight']
        group_components = group_info['components']
        group_score = 0
        group_possible_weight = 0
        group_details = {}

        for component, component_weight in group_components.items():
            base_vals = base_parsed.get(component, [])
            target_vals = target_parsed.get(component, [])
            if base_vals and target_vals:
                similarity, best_match = enhanced_compare_component_groups(base_vals, target_vals, similarity_func)
                contribution = similarity * component_weight
                group_score += contribution
                group_possible_weight += component_weight
                group_details[component] = {
                    'similarity': similarity,
                    'weight': component_weight,
                    'contribution': contribution,
                    'best_match': best_match,
                    'base_values': base_vals,
                    'target_values': target_vals
                }
            else:
                group_details[component] = {
                    'similarity': 0,
                    'weight': component_weight,
                    'contribution': 0,
                    'status': 'missing',
                    'base_values': base_vals,
                    'target_values': target_vals
                }

        normalized_group_score = (group_score / group_possible_weight * group_weight) if group_possible_weight else 0
        total_score += normalized_group_score
        total_possible_weight += group_weight
        detailed_breakdown[group_name] = {
            'group_weight': group_weight,
            'group_score': group_score,
            'group_possible_weight': group_possible_weight,
            'normalized_score': normalized_group_score,
            'components': group_details
        }

    final_similarity = (total_score / total_possible_weight * 100) if total_possible_weight else 0
    return final_similarity, detailed_breakdown

def comprehensive_address_matching(base_address, target_address, nlp_model):
    # Parse addresses for completeness and component extraction
    base_parsed = completeness_score(base_address, '', nlp_model, verbose=False)
    target_parsed = completeness_score(target_address, '', nlp_model, verbose=False)

    # Preprocessing for display + similarity
    base_processed = combine_consecutive_single_characters(removePunctuation(base_address))
    target_processed = combine_consecutive_single_characters(removePunctuation(target_address))

    # Compute hierarchical similarity
    similarity_score, detailed_breakdown = hierarchical_address_comparison(base_parsed, target_parsed, diceSimilarity)

    return {
        'similarity_score': similarity_score,
        'base_analysis': base_parsed,
        'target_analysis': target_parsed,
        'detailed_breakdown': detailed_breakdown,
        'processed_addresses': {'base': base_processed, 'target': target_processed}
    }

# =========================
# Model loading
# =========================
@st.cache_resource
def load_nlp_model():
    try:
        # Discover a custom model if present
        model_patterns = ["entity_rules_ner_*", "address_ner_model_*", "*address*model*"]
        found_models = []
        for pattern in model_patterns:
            found_models.extend(glob.glob(pattern))

        if found_models:
            latest_model = sorted(found_models)[-1]
            try:
                nlp = spacy.load(latest_model)
                # add expand_entities if there's an NER to attach after; else append at end
                try:
                    if "ner" in nlp.pipe_names:
                        nlp.add_pipe("expand_entities", after="ner")
                    else:
                        nlp.add_pipe("expand_entities", last=True)
                except Exception:
                    pass
                return nlp, latest_model
            except Exception as e:
                st.warning(f"Could not load custom model {latest_model}: {e}")
                st.info("Falling back to basic spaCy model...")

        # Fallback models
        for model_name in ["en_core_web_sm", "en_core_web_md"]:
            try:
                nlp = spacy.load(model_name)
                try:
                    if "ner" in nlp.pipe_names:
                        nlp.add_pipe("expand_entities", after="ner")
                    else:
                        nlp.add_pipe("expand_entities", last=True)
                except Exception:
                    pass
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
# UI
# =========================
def main():
    st.markdown('<h1 class="main-header">Address Intelligence System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Advanced NLP-powered Address Analysis and Matching</p>', unsafe_allow_html=True)

    with st.spinner("Loading AI model..."):
        nlp_model, model_name = load_nlp_model()

    if nlp_model is None:
        st.error("Failed to load model. Please check your setup.")
        st.stop()

    with st.sidebar:
        st.header("Configuration")
        with st.expander("Model Information"):
            st.success(f"Model: {model_name}")
            st.info("Model loaded successfully")

        st.subheader("Matching Settings")
        threshold = st.slider("Match Threshold (%)", 0, 100, 70, 5)
        show_detailed = st.checkbox("Show Detailed Breakdown", True)
        show_preprocessing = st.checkbox("Show Preprocessing Steps", False)

        st.subheader("Sample Test Cases")
        sample_addresses = {
            "High Match Example": {
                "base": "HIG/B-24, Indra Puram, shamshabad road, Near water tank, agra - 282002",
                "target": "HIG B-24, Indra Puram, shamshabad rd, Near water tank, agra - 282011"
            },
            "Medium Match Example": {
                "base": "123 Main Street, Sector 21, Noida, UP - 201301",
                "target": "123 Main St, Sec 21, Noida, Uttar Pradesh 201301"
            },
            "Low Match Example": {
                "base": "ABC Colony, Mumbai, Maharashtra - 400001",
                "target": "XYZ Nagar, Delhi, India - 110001"
            }
        }
        selected_sample = st.selectbox("Load Sample Addresses:", ["Custom Input"] + list(sample_addresses.keys()))
        if st.button("Load Selected Sample"):
            if selected_sample != "Custom Input":
                st.session_state.sample_loaded = selected_sample

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

    if st.button("Analyze Address Similarity", type="primary", use_container_width=True):
        if not base_address.strip() or not target_address.strip():
            st.error("Please enter both addresses to proceed with analysis.")
            return

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

        result = comprehensive_address_matching(base_address, target_address, nlp_model)

        status_text.text("Step 4/4: Generating results...")
        progress_bar.progress(100)
        time.sleep(0.2)
        progress_bar.empty()
        status_text.empty()

        st.header("Analysis Results")

        similarity_score = result['similarity_score']
        if similarity_score >= threshold:
            score_class = "high-match"; decision = "MATCH FOUND"; decision_icon = "✅"
        elif similarity_score >= 40:
            score_class = "medium-match"; decision = "PARTIAL MATCH"; decision_icon = "⚠️"
        else:
            score_class = "low-match"; decision = "NO MATCH"; decision_icon = "❌"

        st.markdown(f"""
        <div class="score-card {score_class}">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">{decision_icon}</div>
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{decision}</div>
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">{similarity_score:.1f}%</div>
            <div style="font-size: 1rem; opacity: 0.9;">Similarity Score</div>
        </div>
        """, unsafe_allow_html=True)

        if show_preprocessing:
            st.subheader("Preprocessing Steps")
            c1, c2 = st.columns(2)
            with c1:
                st.text("Original Base Address:"); st.code(base_address)
                st.text("Processed Base Address:"); st.code(result['processed_addresses']['base'])
            with c2:
                st.text("Original Target Address:"); st.code(target_address)
                st.text("Processed Target Address:"); st.code(result['processed_addresses']['target'])

        st.subheader("Address Component Analysis")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
            st.markdown("**Base Address Analysis**")
            base_analysis = result['base_analysis']
            completeness = base_analysis.get('addressCompletenessScore', 0)
            st.metric("Completeness Score", f"{completeness:.1f}%")
            st.markdown("**Components Found:**")
            component_count = 0
            for component, values in base_analysis.items():
                if component not in ['addressCompletenessScore', 'cleanAddress', 'addressInsights'] and isinstance(values, list) and values:
                    for value in values:
                        st.markdown(f'<span class="component-tag">{component}: {value}</span>', unsafe_allow_html=True)
                        component_count += 1
            if component_count == 0:
                st.info("No components automatically detected")
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
            st.markdown("**Target Address Analysis**")
            target_analysis = result['target_analysis']
            completeness = target_analysis.get('addressCompletenessScore', 0)
            st.metric("Completeness Score", f"{completeness:.1f}%")
            st.markdown("**Components Found:**")
            component_count = 0
            for component, values in target_analysis.items():
                if component not in ['addressCompletenessScore', 'cleanAddress', 'addressInsights'] and isinstance(values, list) and values:
                    for value in values:
                        st.markdown(f'<span class="component-tag">{component}: {value}</span>', unsafe_allow_html=True)
                        component_count += 1
            if component_count == 0:
                st.info("No components automatically detected")
            st.markdown('</div>', unsafe_allow_html=True)

        if show_detailed:
            st.subheader("Detailed Similarity Breakdown")
            breakdown = result['detailed_breakdown']
            breakdown_data = []
            for group_name, group_data in breakdown.items():
                group_display_name = group_name.replace('_', ' ').title()
                for component, comp_data in group_data['components'].items():
                    if comp_data.get('similarity', 0) > 0:
                        breakdown_data.append({
                            'Group': group_display_name,
                            'Component': component,
                            'Base Value': ', '.join(comp_data.get('base_values', [])),
                            'Target Value': ', '.join(comp_data.get('target_values', [])),
                            'Similarity': f"{comp_data['similarity']:.2f}",
                            'Weight': f"{comp_data['weight']:.2f}",
                            'Contribution': f"{comp_data['contribution']:.3f}"
                        })
            if breakdown_data:
                df_breakdown = pd.DataFrame(breakdown_data)
                st.dataframe(df_breakdown, use_container_width=True)
            else:
                st.info("No component matches found for detailed breakdown")

            st.subheader("Group-wise Contribution Analysis")
            group_names = []
            group_contributions = []
            group_weights = []
            for group_name, group_data in breakdown.items():
                group_names.append(group_name.replace('_', ' ').title())
                group_contributions.append(group_data['normalized_score'])
                group_weights.append(group_data['group_weight'])

            if any(contrib > 0 for contrib in group_contributions):
                fig = go.Figure()
                fig.add_trace(go.Bar(x=group_names, y=group_contributions, name='Actual Contribution'))
                fig.add_trace(go.Bar(x=group_names, y=group_weights, name='Maximum Possible', opacity=0.6))
                fig.update_layout(title="Component Group Contributions", xaxis_title="Component Groups", yaxis_title="Contribution Score", barmode='overlay')
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("Matching Decision Summary")
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Final Score", f"{similarity_score:.1f}%")
        with c2: st.metric("Threshold", f"{threshold}%")
        with c3:
            if similarity_score >= threshold:
                st.success("MATCH")
            else:
                st.error("NO MATCH")

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
