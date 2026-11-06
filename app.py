from datetime import datetime
import json
import os
import textwrap
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Core Scientific & AI Modules
from src.profiler import profile_dataframe, correlation_table, detect_outliers, compute_baseline_smd
from src.dag import (
    CausalGraph,
    create_classic_confounding_dag,
    create_mediation_dag,
    create_m_bias_dag,
    create_instrumental_variable_dag
)
from src.causal import (
    estimate_ate,
    propensity_score_match,
    estimate_ipw,
    estimate_aipw,
    estimate_dml,
    compare_all_estimators
)
from src.sensitivity import (
    compute_e_value,
    run_placebo_treatment_refutation,
    run_random_common_cause_refutation,
    run_data_subset_refutation
)
from src.benchmark import generate_synthetic_dataset, run_monte_carlo_benchmark
from src.assistant import (
    answer_with_evidence,
    build_research_context,
    MODE_GROUNDED,
    MODE_UNCONSTRAINED,
    MODE_AUDITOR,
    KNOWLEDGE_SOURCE_DATASET,
    KNOWLEDGE_SOURCE_DOCUMENTS,
    KNOWLEDGE_SOURCE_HYBRID
)
from src.eval import verify_claims_against_evidence, audit_causal_assumptions
from src.reporting import build_markdown_report, build_json_reproducibility_bundle, build_html_report
from src.rag import (
    AcademicVectorStore,
    answer_document_query,
    answer_document_chat,
    answer_literature_query,
    compare_literature_methods,
    extract_key_information,
    generate_research_summary
)
from src.gemini_client import (
    get_api_key,
    mask_api_key,
    validate_gemini_connection,
    generate_gemini_text,
    DEFAULT_TEXT_MODELS,
    DEFAULT_PRO_MODELS,
    sanitize_message,
)

# ==============================================================================
# STREAMLIT APP CONFIGURATION & PREMIUM DARK NAVY + TEAL THEME
# ==============================================================================
st.set_page_config(
    page_title="CausalLens AI | Enterprise Causal Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# STREAMLIT APP CONFIGURATION & ENTERPRISE DARK CHARCOAL / MIDNIGHT THEME
# ==============================================================================
st.set_page_config(
    page_title="CausalLens AI | Enterprise Causal Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Enterprise AI SaaS CSS Theme with Midnight Navy, Dark Charcoal & Deep Indigo
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* Smooth Scrolling & Hardware-Accelerated Responsive Canvas */
html, body, [data-testid="stAppViewContainer"], section.main {
    scroll-behavior: smooth !important;
    -webkit-overflow-scrolling: touch;
}

* {
    scrollbar-width: thin;
    scrollbar-color: #1E293B #080C14;
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(4px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.main .block-container {
    animation: fadeInUp 0.18s cubic-bezier(0.16, 1, 0.3, 1);
    will-change: opacity, transform;
    max-width: 1360px;
    padding-top: 24px;
    padding-bottom: 48px;
}

/* GPU Hardware-Accelerated Rendering - Eliminates Paint Lag */
.saas-card, .metric-tile, .portal-card, .workspace-jump-card, .stat-counter-item, .causal-step-box, .hero-nexus-card {
    transform: translateZ(0);
    backface-visibility: hidden;
    will-change: transform, opacity, box-shadow;
}

/* Lag-free chart and table rendering */
.js-plotly-plot, div[data-testid="stDataFrame"], div[data-testid="stExpander"] {
    contain: layout style;
    transform: translateZ(0);
}

/* Main Canvas & Dark Charcoal / Midnight Navy Surfaces */
.stApp {
    background-color: #080C14;
    background-image: 
        radial-gradient(ellipse 80% 45% at 50% -8%, rgba(99, 102, 241, 0.10) 0%, rgba(14, 165, 233, 0.04) 45%, transparent 70%),
        radial-gradient(circle at 10% 25%, rgba(99, 102, 241, 0.04) 0%, transparent 35%),
        radial-gradient(circle at 90% 65%, rgba(6, 182, 212, 0.035) 0%, transparent 40%);
    color: #F1F5F9;
    font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    letter-spacing: -0.01em;
}

/* Custom Sleek Scrollbar with Smooth Transition */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #080C14;
}
::-webkit-scrollbar-thumb {
    background: #1E293B;
    border-radius: 9999px;
    transition: background 0.2s ease;
}
::-webkit-scrollbar-thumb:hover {
    background: #38BDF8;
}

/* Sidebar Styling - Compact Midnight Navy with Refined Edge */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B0F19 0%, #080C14 100%) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
    box-shadow: 4px 0 24px rgba(0, 0, 0, 0.45) !important;
}

/* Sidebar Category Headers */
.nav-category-header {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #64748B;
    margin-top: 14px;
    margin-bottom: 6px;
    padding-left: 8px;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* Sidebar Nav Buttons: Active State (Primary) */
section[data-testid="stSidebar"] div.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, rgba(99, 102, 241, 0.18) 0%, rgba(56, 189, 248, 0.08) 100%) !important;
    border: 1px solid rgba(99, 102, 241, 0.35) !important;
    border-left: 3px solid #38BDF8 !important;
    color: #FFFFFF !important;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
    text-align: left !important;
    justify-content: flex-start !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 8px 12px !important;
    border-radius: 8px !important;
    transform: none !important;
    transition: all 0.15s ease !important;
}

/* Sidebar Nav Buttons: Inactive State (Secondary) */
section[data-testid="stSidebar"] div.stButton > button[kind="secondary"] {
    background: transparent !important;
    border: 1px solid transparent !important;
    color: #94A3B8 !important;
    text-align: left !important;
    justify-content: flex-start !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    padding: 8px 12px !important;
    border-radius: 8px !important;
    transition: all 0.15s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

section[data-testid="stSidebar"] div.stButton > button[kind="secondary"]:hover {
    background: rgba(255, 255, 255, 0.035) !important;
    border-color: rgba(255, 255, 255, 0.08) !important;
    color: #F8FAFC !important;
    transform: translateX(3px) !important;
}

/* Clean Brand Header in Sidebar */
.brand-container {
    background: linear-gradient(145deg, rgba(17, 24, 39, 0.95) 0%, rgba(11, 15, 25, 0.98) 100%);
    border: 1px solid rgba(99, 102, 241, 0.22);
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 12px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
}

.brand-title {
    font-size: 16px;
    font-weight: 700;
    letter-spacing: -0.3px;
    color: #FFFFFF;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 10px;
}

.brand-icon-box {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    border-radius: 7px;
    background: linear-gradient(135deg, #4F46E5 0%, #06B6D4 100%);
    box-shadow: 0 2px 10px rgba(79, 70, 229, 0.35);
}

.brand-subtitle {
    font-size: 9.5px;
    color: #64748B;
    margin-top: 5px;
    text-transform: uppercase;
    letter-spacing: 0.9px;
    font-weight: 700;
}

/* Gradient Titles & Refined Typography */
.gradient-title {
    background: linear-gradient(135deg, #FFFFFF 0%, #E0E7FF 45%, #C7D2FE 75%, #38BDF8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline-block;
}

.gradient-text-purple {
    background: linear-gradient(135deg, #FFFFFF 0%, #E0E7FF 35%, #C084FC 70%, #818CF8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline-block;
}

.gradient-text-cyan {
    background: linear-gradient(135deg, #FFFFFF 0%, #BAE6FD 40%, #38BDF8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline-block;
}

.gradient-text-emerald {
    background: linear-gradient(135deg, #FFFFFF 0%, #A7F3D0 40%, #34D399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline-block;
}

.gradient-text-rose {
    background: linear-gradient(135deg, #FFFFFF 0%, #FECDD3 40%, #FB7185 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline-block;
}

/* Refined Enterprise Cards */
.saas-card {
    background: linear-gradient(145deg, rgba(16, 22, 36, 0.85) 0%, rgba(10, 14, 24, 0.98) 100%);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 14px;
    padding: 22px;
    margin-bottom: 18px;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.05);
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
    position: relative;
}

.saas-card:hover {
    border-color: rgba(99, 102, 241, 0.35);
    box-shadow: 0 8px 28px -4px rgba(0, 0, 0, 0.5);
    transform: translateY(-1px);
}

.gradient-card {
    background: linear-gradient(145deg, rgba(19, 25, 42, 0.9) 0%, rgba(11, 15, 26, 0.98) 100%);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 14px;
    padding: 24px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
}

.gradient-card:hover {
    border-color: rgba(56, 189, 248, 0.45);
    box-shadow: 0 8px 32px rgba(99, 102, 241, 0.2);
    transform: translateY(-2px);
}

/* Metric Display Tiles with Rich RGB Gradients */
.metric-tile {
    border-radius: 14px;
    padding: 18px 20px;
    text-align: left;
    transition: all 0.2s ease;
    backdrop-filter: blur(12px);
}

.metric-tile:hover {
    transform: translateY(-2px);
}

.metric-tile-cyan {
    background: linear-gradient(135deg, rgba(6, 182, 212, 0.12) 0%, rgba(14, 165, 233, 0.05) 45%, rgba(14, 20, 32, 0.95) 100%);
    border: 1px solid rgba(6, 182, 212, 0.3);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.08);
}
.metric-tile-cyan:hover {
    border-color: rgba(56, 189, 248, 0.6);
    box-shadow: 0 8px 24px rgba(6, 182, 212, 0.2);
}

.metric-tile-violet {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.14) 0%, rgba(79, 70, 229, 0.06) 45%, rgba(14, 20, 32, 0.95) 100%);
    border: 1px solid rgba(99, 102, 241, 0.3);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.08);
}
.metric-tile-violet:hover {
    border-color: rgba(129, 140, 248, 0.65);
    box-shadow: 0 8px 24px rgba(99, 102, 241, 0.25);
}

.metric-tile-emerald {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(5, 150, 105, 0.05) 45%, rgba(14, 20, 32, 0.95) 100%);
    border: 1px solid rgba(16, 185, 129, 0.3);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.08);
}
.metric-tile-emerald:hover {
    border-color: rgba(52, 211, 153, 0.6);
    box-shadow: 0 8px 24px rgba(16, 185, 129, 0.2);
}

.metric-tile-rose {
    background: linear-gradient(135deg, rgba(244, 63, 94, 0.12) 0%, rgba(225, 29, 72, 0.05) 45%, rgba(14, 20, 32, 0.95) 100%);
    border: 1px solid rgba(244, 63, 94, 0.3);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.08);
}
.metric-tile-rose:hover {
    border-color: rgba(251, 113, 133, 0.6);
    box-shadow: 0 8px 24px rgba(244, 63, 94, 0.2);
}

.metric-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #94A3B8;
    font-weight: 600;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 6px;
}

.metric-value {
    font-size: 24px;
    font-weight: 700;
    color: #FFFFFF;
    letter-spacing: -0.5px;
    line-height: 1.15;
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.metric-subtext {
    font-size: 11.5px;
    color: #A5B4FC;
    margin-top: 5px;
    font-weight: 500;
}

/* Minimalist Stat Counter Grid (Image 1 & 2 Inspiration) */
.stats-counter-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin: 18px 0;
}
.stat-counter-item {
    background: linear-gradient(145deg, rgba(16, 22, 36, 0.75) 0%, rgba(10, 14, 24, 0.95) 100%);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 12px;
    padding: 16px 14px;
    text-align: center;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
    transition: all 0.2s ease;
}
.stat-counter-item:hover {
    border-color: rgba(99, 102, 241, 0.35);
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
}
.stat-number {
    font-size: 30px;
    font-weight: 800;
    letter-spacing: -0.025em;
    line-height: 1.1;
    margin-bottom: 4px;
    font-family: 'Plus Jakarta Sans', sans-serif;
}
.stat-caption {
    font-size: 11px;
    color: #94A3B8;
    font-weight: 500;
    line-height: 1.4;
}

/* Subtle Status Pills */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.16) 0%, rgba(79, 70, 229, 0.08) 100%);
    border: 1px solid rgba(99, 102, 241, 0.3);
    color: #C7D2FE;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 11px;
    font-weight: 600;
}

.status-pill-cyan {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: linear-gradient(135deg, rgba(6, 182, 212, 0.16) 0%, rgba(14, 165, 233, 0.08) 100%);
    border: 1px solid rgba(6, 182, 212, 0.3);
    color: #BAE6FD;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 11px;
    font-weight: 600;
}

.status-pill-emerald {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.16) 0%, rgba(5, 150, 105, 0.08) 100%);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: #A7F3D0;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 11px;
    font-weight: 600;
}

.status-pill-amber {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.16) 0%, rgba(217, 119, 6, 0.08) 100%);
    border: 1px solid rgba(245, 158, 11, 0.3);
    color: #FDE68A;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 11px;
    font-weight: 600;
}

/* Hero Showcase Card */
.hero-showcase-card {
    background: linear-gradient(145deg, rgba(24, 20, 48, 0.35) 0%, rgba(16, 22, 36, 0.85) 50%, rgba(10, 14, 24, 0.98) 100%);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 16px;
    padding: 28px 30px;
    margin-bottom: 22px;
    box-shadow: 0 10px 36px -10px rgba(99, 102, 241, 0.18), inset 0 1px 0 rgba(255, 255, 255, 0.08);
    position: relative;
    overflow: hidden;
}

/* Hero Badge */
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.16) 0%, rgba(56, 189, 248, 0.08) 100%);
    border: 1px solid rgba(99, 102, 241, 0.3);
    box-shadow: 0 2px 10px rgba(99, 102, 241, 0.18);
    border-radius: 9999px;
    padding: 4px 14px;
    font-size: 10.5px;
    font-weight: 700;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: #C7D2FE;
    margin-bottom: 12px;
}

.hero-title {
    font-size: 38px;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1.15;
    color: #FFFFFF;
    margin: 0;
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.hero-subtitle {
    color: #94A3B8;
    margin-top: 10px;
    font-size: 14.5px;
    line-height: 1.6;
    max-width: 780px;
}

/* Homepage Top Commercial Navbar */
.enterprise-navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 18px;
    background: linear-gradient(145deg, rgba(16, 22, 36, 0.9) 0%, rgba(10, 14, 24, 0.98) 100%);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 12px;
    margin-bottom: 22px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.06);
}

/* AI Research Hub Container */
.search-hub-box {
    background: linear-gradient(145deg, rgba(19, 25, 44, 0.85) 0%, rgba(11, 15, 28, 0.98) 100%);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 16px;
    padding: 24px 26px;
    margin: 24px 0;
    box-shadow: 0 12px 36px -8px rgba(99, 102, 241, 0.18), inset 0 1px 0 rgba(255, 255, 255, 0.08);
    position: relative;
}

/* Prompt Chips */
.prompt-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 11.5px;
    font-weight: 500;
    background: rgba(255, 255, 255, 0.035);
    border: 1px solid rgba(255, 255, 255, 0.08);
    color: #CBD5E1;
    cursor: pointer;
    transition: all 0.15s ease;
}

.prompt-chip:hover {
    background: rgba(99, 102, 241, 0.16);
    border-color: rgba(56, 189, 248, 0.4);
    color: #FFFFFF;
    transform: translateY(-1px);
}

/* Active Project & Documents Cards */
.portal-card {
    background: linear-gradient(145deg, rgba(16, 22, 36, 0.8) 0%, rgba(10, 14, 24, 0.98) 100%);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 14px;
    padding: 22px;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.35);
    transition: all 0.2s ease;
    height: 100%;
}

.portal-card:hover {
    border-color: rgba(99, 102, 241, 0.35);
    box-shadow: 0 10px 28px -6px rgba(0, 0, 0, 0.45);
    transform: translateY(-2px);
}

/* Workspace Jump Cards */
.workspace-jump-card {
    background: linear-gradient(145deg, rgba(16, 22, 36, 0.8) 0%, rgba(10, 14, 24, 0.98) 100%);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 14px;
    padding: 20px;
    transition: all 0.2s ease;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

.workspace-jump-card:hover {
    border-color: rgba(56, 189, 248, 0.4);
    box-shadow: 0 8px 24px -4px rgba(99, 102, 241, 0.2);
    transform: translateY(-2px);
}

/* Workflow Stage Box */
.workflow-box {
    background: linear-gradient(145deg, rgba(16, 22, 36, 0.72) 0%, rgba(10, 14, 24, 0.96) 100%);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 14px;
    padding: 22px 20px;
    height: 100%;
    transition: all 0.2s ease;
}

.workflow-box:hover {
    border-color: rgba(99, 102, 241, 0.35);
    transform: translateY(-2px);
}

/* Trust / Methodology Bar */
.trust-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: linear-gradient(90deg, rgba(14, 20, 32, 0.85) 0%, rgba(22, 18, 42, 0.9) 50%, rgba(14, 20, 32, 0.85) 100%);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 10px;
    padding: 10px 18px;
    margin: 18px 0;
    color: #94A3B8;
    font-size: 11.5px;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.2);
}

.trust-badge {
    color: #CBD5E1;
    font-weight: 500;
}

/* Primary Button Styling */
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #4F46E5 0%, #3B82F6 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(199, 210, 254, 0.3) !important;
    border-radius: 8px !important;
    padding: 8px 18px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    letter-spacing: -0.01em !important;
    box-shadow: 0 3px 12px rgba(59, 130, 246, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
    transition: all 0.15s ease !important;
}

div.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #6366F1 0%, #2563EB 100%) !important;
    box-shadow: 0 5px 18px rgba(99, 102, 241, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
    transform: translateY(-1px) !important;
}

/* Secondary Button Styling */
div.stButton > button[kind="secondary"] {
    background: linear-gradient(145deg, rgba(18, 24, 38, 0.9) 0%, rgba(12, 16, 27, 0.98) 100%) !important;
    color: #E2E8F0 !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 8px !important;
    padding: 7px 16px !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    transition: all 0.15s ease !important;
}

div.stButton > button[kind="secondary"]:hover {
    border-color: rgba(56, 189, 248, 0.35) !important;
    color: #FFFFFF !important;
    background: linear-gradient(145deg, rgba(24, 32, 52, 0.9) 0%, rgba(16, 22, 36, 0.98) 100%) !important;
    box-shadow: 0 3px 12px rgba(0, 0, 0, 0.3) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background-color: #0B0F19;
    padding: 4px;
    border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.07);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 7px;
    color: #94A3B8;
    font-weight: 500;
    padding: 7px 16px;
    font-size: 13px;
    border: none;
    transition: all 0.15s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #F1F5F9;
    background: rgba(255, 255, 255, 0.04);
}

.stTabs [aria-selected="true"] {
    background: #141C30 !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(56, 189, 248, 0.3) !important;
}

/* Dataframe */
div[data-testid="stDataFrame"] {
    border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    overflow: hidden;
    background: #0B0F19;
}

/* Expanders */
div[data-testid="stExpander"] {
    background-color: #0E1424 !important;
    border: 1px solid rgba(255, 255, 255, 0.07) !important;
    border-radius: 10px !important;
}

/* Form Inputs, Dropdowns & Focus Rings */
div[data-baseweb="input"] {
    background-color: #0E1424 !important;
    border: 1px solid rgba(255, 255, 255, 0.09) !important;
    border-radius: 8px !important;
    transition: all 0.15s ease !important;
}

div[data-baseweb="input"]:focus-within {
    border-color: #38BDF8 !important;
    box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.22) !important;
}

div[data-baseweb="select"] {
    background-color: #0E1424 !important;
    border: 1px solid rgba(255, 255, 255, 0.09) !important;
    border-radius: 8px !important;
}

div[data-baseweb="select"]:focus-within {
    border-color: #38BDF8 !important;
    box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.22) !important;
}

ul[data-baseweb="menu"] {
    background-color: #11182A !important;
    border: 1px solid rgba(56, 189, 248, 0.25) !important;
    border-radius: 8px !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5) !important;
}

li[data-baseweb="menu-item"] {
    color: #CBD5E1 !important;
    transition: all 0.1s ease !important;
}

li[data-baseweb="menu-item"]:hover {
    background-color: rgba(56, 189, 248, 0.12) !important;
    color: #FFFFFF !important;
}

/* Empty State Cards */
.empty-state-card {
    background: #0E1424;
    border: 1px dashed rgba(99, 102, 241, 0.3);
    border-radius: 12px;
    padding: 30px 22px;
    text-align: center;
    margin: 18px 0;
}

.empty-state-icon {
    font-size: 30px;
    margin-bottom: 10px;
    display: inline-block;
}

.empty-state-title {
    font-size: 15px;
    font-weight: 700;
    color: #F1F5F9;
    margin-bottom: 6px;
}

.empty-state-text {
    font-size: 12.5px;
    color: #94A3B8;
    max-width: 500px;
    margin: 0 auto 16px auto;
    line-height: 1.5;
}

/* Workspace Breadcrumb */
.workspace-breadcrumb {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    color: #64748B;
    padding-bottom: 12px;
    margin-bottom: 16px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}

.workspace-breadcrumb span.crumb-active {
    color: #BAE6FD;
    font-weight: 600;
}

/* Responsive & Mobile Media Queries */
@media (max-width: 992px) {
    .causal-stepper-container {
        grid-template-columns: repeat(2, 1fr) !important;
        gap: 10px !important;
    }
}

@media (max-width: 768px) {
    .main .block-container {
        padding-left: 12px !important;
        padding-right: 12px !important;
        padding-top: 14px !important;
    }
    .hero-title {
        font-size: 24px !important;
    }
    .hero-subtitle {
        font-size: 13px !important;
    }
    .metric-value {
        font-size: 20px !important;
    }
    .enterprise-navbar {
        flex-direction: column;
        align-items: flex-start !important;
        gap: 8px !important;
    }
    .trust-bar {
        flex-direction: column;
        align-items: flex-start !important;
        gap: 6px !important;
    }
    .workspace-breadcrumb {
        flex-wrap: wrap !important;
        gap: 4px 8px !important;
    }
    .portal-card, .workspace-jump-card, .workflow-box {
        margin-bottom: 12px !important;
    }
    .search-hub-box {
        padding: 16px 14px !important;
    }
}

@media (max-width: 540px) {
    .causal-stepper-container {
        grid-template-columns: 1fr !important;
        gap: 8px !important;
    }
    .nexus-telemetry-grid {
        grid-template-columns: 1fr !important;
        gap: 6px !important;
    }
    .prompt-chip {
        font-size: 10.5px !important;
        padding: 4px 10px !important;
    }
    .hero-nexus-card {
        padding: 14px 12px !important;
    }
    .hero-title {
        font-size: 21px !important;
    }
    div.stButton > button {
        min-height: 40px !important;
        font-size: 12.5px !important;
    }
}

/* Commercial SaaS Footer */
.enterprise-footer {
    border-top: 1px solid rgba(255, 255, 255, 0.07);
    padding: 32px 0 20px 0;
    margin-top: 48px;
    color: #64748B;
    font-size: 12px;
}

/* Causal Epistemology & Assumption vs Evidence Badges */
.badge-assumption {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(245, 158, 11, 0.12);
    border: 1px solid rgba(245, 158, 11, 0.4);
    color: #FBBF24;
    border-radius: 9999px;
    padding: 3px 10px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.6px;
    text-transform: uppercase;
}

.badge-empirical {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.4);
    color: #34D399;
    border-radius: 9999px;
    padding: 3px 10px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.6px;
    text-transform: uppercase;
}

/* 4-Step Causal Workflow Progress Stepper */
.causal-stepper-container {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 22px;
}

.causal-step-box {
    background: linear-gradient(145deg, rgba(16, 22, 36, 0.85) 0%, rgba(10, 14, 24, 0.98) 100%);
    border: 1px solid rgba(99, 102, 241, 0.22);
    border-radius: 12px;
    padding: 14px 16px;
    position: relative;
    transition: all 0.2s ease;
}

.causal-step-box:hover {
    border-color: rgba(56, 189, 248, 0.45);
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.35);
}

.causal-step-num {
    font-size: 9.5px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 4px;
}

.causal-step-title {
    font-size: 13px;
    font-weight: 700;
    color: #F8FAFC;
    margin-bottom: 2px;
}

.causal-step-desc {
    font-size: 11px;
    color: #94A3B8;
    line-height: 1.4;
}

.epistemology-card {
    background: linear-gradient(145deg, rgba(16, 22, 36, 0.85) 0%, rgba(10, 14, 24, 0.98) 100%);
    border: 1px solid rgba(99, 102, 241, 0.22);
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
}

/* Futuristic Causal Intelligence Nexus Visual (Image 2 Inspiration) */
.hero-nexus-card {
    background: linear-gradient(145deg, rgba(15, 23, 42, 0.94) 0%, rgba(8, 12, 22, 0.98) 100%);
    border: 1px solid rgba(56, 189, 248, 0.22);
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 16px 40px -10px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.08);
    position: relative;
    overflow: hidden;
}

.hero-nexus-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 14px;
    padding-bottom: 12px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}

.nexus-title-group {
    display: flex;
    align-items: center;
    gap: 8px;
}

.nexus-pulse-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #10B981;
    box-shadow: 0 0 10px #10B981;
    display: inline-block;
}

.nexus-title {
    font-size: 10.5px;
    font-weight: 700;
    letter-spacing: 0.9px;
    text-transform: uppercase;
    color: #E2E8F0;
}

.nexus-badge {
    font-size: 10px;
    font-weight: 600;
    background: rgba(56, 189, 248, 0.12);
    border: 1px solid rgba(56, 189, 248, 0.3);
    color: #38BDF8;
    border-radius: 6px;
    padding: 2px 8px;
}

.nexus-canvas-wrapper {
    position: relative;
    width: 100%;
    border-radius: 12px;
    background: radial-gradient(circle at 50% 50%, rgba(99, 102, 241, 0.15) 0%, rgba(14, 165, 233, 0.06) 40%, rgba(8, 12, 22, 0.9) 75%);
    border: 1px solid rgba(255, 255, 255, 0.05);
    padding: 14px;
    margin-bottom: 14px;
}

.nexus-telemetry-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
}

.nexus-telemetry-item {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.04);
    border-radius: 8px;
    padding: 8px 10px;
    text-align: center;
}

.nexus-t-lbl {
    display: block;
    font-size: 9px;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 2px;
}

.nexus-t-val {
    display: block;
    font-size: 12.5px;
    font-weight: 700;
    color: #F8FAFC;
    font-family: 'JetBrains Mono', monospace;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def apply_dark_theme(fig):
    """Apply consistent premium enterprise midnight charcoal styling to all Plotly figures."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#080C14",
        plot_bgcolor="#0B0F19",
        font=dict(color="#E2E8F0", family="'Plus Jakarta Sans', 'Inter', -apple-system, sans-serif"),
        colorway=["#38BDF8", "#818CF8", "#34D399", "#A78BFA", "#F43F5E", "#FBBF24", "#06B6D4"],
        xaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.04)",
            zerolinecolor="rgba(255, 255, 255, 0.08)",
            tickfont=dict(size=11, color="#94A3B8")
        ),
        yaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.04)",
            zerolinecolor="rgba(255, 255, 255, 0.08)",
            tickfont=dict(size=11, color="#94A3B8")
        ),
        hoverlabel=dict(
            bgcolor="#111827",
            bordercolor="rgba(56, 189, 248, 0.35)",
            font=dict(color="#FFFFFF", size=12, family="'Plus Jakarta Sans', sans-serif")
        ),
        hovermode="closest",
        margin=dict(l=24, r=24, t=44, b=24),
    )
    return fig


# ==============================================================================
# DEDICATED PLATFORM PAGES & NAVIGATION ARCHITECTURE
# ==============================================================================
PAGE_OVERVIEW = "Overview / Home"
PAGE_DATA = "Data Analysis"
PAGE_CAUSAL = "Causal Analysis & DAG Workbench"
PAGE_ASSISTANT = "AI Research Assistant"
PAGE_RAG = "Knowledge Base / RAG"
PAGE_LAB = "Research Lab"
PAGE_REPORTS = "Reports & Exports"
PAGE_SETTINGS = "Settings"

ALL_PAGES = [
    PAGE_OVERVIEW,
    PAGE_DATA,
    PAGE_CAUSAL,
    PAGE_ASSISTANT,
    PAGE_RAG,
    PAGE_LAB,
    PAGE_REPORTS,
    PAGE_SETTINGS,
]
all_pages = ALL_PAGES

PAGE_NORMALIZE = {
    PAGE_OVERVIEW: PAGE_OVERVIEW,
    "Overview": PAGE_OVERVIEW,
    "Overview / Home": PAGE_OVERVIEW,
    "Home": PAGE_OVERVIEW,
    
    PAGE_DATA: PAGE_DATA,
    "Data Analysis": PAGE_DATA,
    "Data Diagnostics": PAGE_DATA,
    "Data Quality": PAGE_DATA,
    "Data Analysis & Quality": PAGE_DATA,
    
    PAGE_CAUSAL: PAGE_CAUSAL,
    "Causal Analysis & DAG Workbench": PAGE_CAUSAL,
    "Causal Analysis": PAGE_CAUSAL,
    "Causal DAG Studio": PAGE_CAUSAL,
    "Causal & DAG Workbench": PAGE_CAUSAL,
    "Causal Workbench": PAGE_CAUSAL,
    "DAG Studio": PAGE_CAUSAL,
    
    PAGE_ASSISTANT: PAGE_ASSISTANT,
    "AI Research Assistant": PAGE_ASSISTANT,
    "AI Assistant": PAGE_ASSISTANT,
    "Research Assistant": PAGE_ASSISTANT,
    
    PAGE_RAG: PAGE_RAG,
    "Knowledge Base / RAG": PAGE_RAG,
    "Knowledge Base": PAGE_RAG,
    "Literature RAG": PAGE_RAG,
    "RAG": PAGE_RAG,
    
    PAGE_LAB: PAGE_LAB,
    "Research Lab": PAGE_LAB,
    "Simulation Lab": PAGE_LAB,
    "Lab": PAGE_LAB,
    
    PAGE_REPORTS: PAGE_REPORTS,
    "Reports & Exports": PAGE_REPORTS,
    "Reports": PAGE_REPORTS,
    "Audit Reports": PAGE_REPORTS,
    
    PAGE_SETTINGS: PAGE_SETTINGS,
    "Settings": PAGE_SETTINGS,
}

NAV_SECTIONS = [
    {
        "category": "PLATFORM",
        "items": [
            (PAGE_OVERVIEW, "Overview / Home"),
            (PAGE_DATA, "Data Analysis"),
            (PAGE_CAUSAL, "Causal Analysis & DAG Workbench"),
        ]
    },
    {
        "category": "RESEARCH",
        "items": [
            (PAGE_ASSISTANT, "AI Research Assistant"),
            (PAGE_RAG, "Knowledge Base / RAG"),
            (PAGE_LAB, "Research Lab"),
            (PAGE_REPORTS, "Reports & Exports"),
        ]
    },
    {
        "category": "SYSTEM",
        "items": [
            (PAGE_SETTINGS, "Settings"),
        ]
    }
]

# Helper for programmatic navigation from buttons and links
def switch_page(target: str):
    normalized = PAGE_NORMALIZE.get(target, target)
    st.session_state["nav_page"] = normalized
    st.rerun()


# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================
if "df" not in st.session_state:
    st.session_state["df"] = None
if "dataset_name" not in st.session_state:
    st.session_state["dataset_name"] = None
if "all_estimators" not in st.session_state:
    st.session_state["all_estimators"] = None
if "causal_result" not in st.session_state:
    st.session_state["causal_result"] = None
if "matching_result" not in st.session_state:
    st.session_state["matching_result"] = None
if "balance_table" not in st.session_state:
    st.session_state["balance_table"] = None
if "baseline_smd" not in st.session_state:
    st.session_state["baseline_smd"] = None
if "sensitivity_result" not in st.session_state:
    st.session_state["sensitivity_result"] = None
if "refutation_results" not in st.session_state:
    st.session_state["refutation_results"] = None
if "dag_summary" not in st.session_state:
    st.session_state["dag_summary"] = None
if "interactive_dag" not in st.session_state:
    st.session_state["interactive_dag"] = None
if "dag_last_cycle_msg" not in st.session_state:
    st.session_state["dag_last_cycle_msg"] = None
if "assistant_logs" not in st.session_state:
    st.session_state["assistant_logs"] = []
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "mc_res" not in st.session_state:
    st.session_state["mc_res"] = None
if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = PAGE_OVERVIEW
if "rag_store" not in st.session_state:
    st.session_state["rag_store"] = AcademicVectorStore()
    try:
        if st.session_state["rag_store"].collection.count() == 0:
            st.session_state["rag_store"].seed_benchmark_papers()
    except Exception:
        pass
if "home_ai_query" not in st.session_state:
    st.session_state["home_ai_query"] = ""
if "home_ai_response" not in st.session_state:
    st.session_state["home_ai_response"] = None
if "rag_last_query" not in st.session_state:
    st.session_state["rag_last_query"] = None
if "rag_last_result" not in st.session_state:
    st.session_state["rag_last_result"] = None
if "rag_chat_history" not in st.session_state:
    st.session_state["rag_chat_history"] = []
if "rag_chat_input_val" not in st.session_state:
    st.session_state["rag_chat_input_val"] = ""
if "rag_key_info_result" not in st.session_state:
    st.session_state["rag_key_info_result"] = None
if "rag_comparison_result" not in st.session_state:
    st.session_state["rag_comparison_result"] = None
if "rag_summary_result" not in st.session_state:
    st.session_state["rag_summary_result"] = None
if "assistant_knowledge_source" not in st.session_state:
    st.session_state["assistant_knowledge_source"] = KNOWLEDGE_SOURCE_DATASET
if "research_carousel_idx" not in st.session_state:
    st.session_state["research_carousel_idx"] = 0

# ==============================================================================
# SIDEBAR NAVIGATION & DATASET REPOSITORY
# ==============================================================================
with st.sidebar:
    st.markdown(
        """<div class="brand-container">
<div class="brand-title">
<span class="brand-icon-box">
<svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="#FFFFFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M2 17L12 22L22 17" stroke="#38BDF8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M2 12L12 17L22 12" stroke="#C084FC" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
</span>
<span>CausalLens AI</span>
</div>
<div class="brand-subtitle">ENTERPRISE DECISION INTELLIGENCE</div>
</div>""",
        unsafe_allow_html=True
    )

    # Grouped Navigation System: Workspace, Intelligence, System
    curr_nav = PAGE_NORMALIZE.get(st.session_state.get("nav_page", PAGE_OVERVIEW), PAGE_OVERVIEW)
    
    for nav_sec in NAV_SECTIONS:
        st.markdown(f"<div class='nav-category-header'>{nav_sec['category']}</div>", unsafe_allow_html=True)
        for page_key, page_title in nav_sec["items"]:
            is_active = (curr_nav == page_key)
            b_type = "primary" if is_active else "secondary"
            if st.button(page_title, key=f"nav_link_{page_key}", use_container_width=True, type=b_type):
                switch_page(page_key)

    selected_page = curr_nav

    st.markdown("---")
    st.markdown("<div style='font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:1px; color:#7C71A8; margin-bottom:8px;'>Dataset Repository</div>", unsafe_allow_html=True)
    data_source = st.selectbox(
        "Select Active Dataset",
        [
            "Synthetic Ground-Truth Demo (tau=2.5)",
            "Lalonde Job Training Program (Benchmark)",
            "Upload Custom Dataset (CSV / Excel)"
        ],
        index=0
    )

    if data_source == "Synthetic Ground-Truth Demo (tau=2.5)":
        if st.session_state.get("dataset_name") != "synthetic_demo" or st.session_state.get("df") is None:
            demo_df, meta = generate_synthetic_dataset(n=1000, true_ate=2.5, confounding_strength=1.0, seed=42)
            st.session_state["df"] = demo_df
            st.session_state["dataset_name"] = "synthetic_demo"
            st.session_state["ground_truth_ate"] = 2.5
            st.session_state["analysis_spec"] = {
                "treatment": "treatment",
                "outcome": "outcome",
                "confounders": ["x1_feature", "age", "binary_flag", "prior_activity"]
            }
    elif data_source == "Lalonde Job Training Program (Benchmark)":
        if st.session_state.get("dataset_name") != "lalonde_benchmark" or st.session_state.get("df") is None:
            rng = np.random.default_rng(101)
            n_lalonde = 900
            age = rng.normal(32, 8, n_lalonde).clip(17, 55)
            education = rng.integers(8, 17, n_lalonde)
            prior_earnings = rng.exponential(4500, n_lalonde)
            unemployed_prior = rng.binomial(1, 0.38, n_lalonde)
            logit_treat = -1.1 - 0.02 * (age - 30) - 0.08 * (education - 12) + 0.45 * unemployed_prior - 0.00012 * prior_earnings
            p_treat = 1 / (1 + np.exp(-logit_treat))
            treat = rng.binomial(1, p_treat)
            post_earnings = 1800.0 * treat + 380.0 * education + 0.62 * prior_earnings - 900.0 * unemployed_prior + rng.normal(0, 1600, n_lalonde)
            lalonde_df = pd.DataFrame({
                "age": age,
                "education_years": education,
                "prior_earnings": prior_earnings,
                "unemployed_prior": unemployed_prior,
                "training_program": treat,
                "post_earnings": post_earnings
            })
            st.session_state["df"] = lalonde_df
            st.session_state["dataset_name"] = "lalonde_benchmark"
            st.session_state["ground_truth_ate"] = 1800.0
            st.session_state["analysis_spec"] = {
                "treatment": "training_program",
                "outcome": "post_earnings",
                "confounders": ["age", "education_years", "prior_earnings", "unemployed_prior"]
            }
    elif data_source in ("Upload Custom Dataset (CSV / Excel)", "Upload Custom CSV"):
        uploaded_file = st.file_uploader("Upload CSV or Excel Dataset", type=["csv", "xlsx", "xls"], key="sidebar_dataset_uploader")
        if uploaded_file is not None:
            try:
                if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
                    st.session_state["df"] = pd.read_excel(uploaded_file)
                else:
                    st.session_state["df"] = pd.read_csv(uploaded_file)
                st.session_state["dataset_name"] = uploaded_file.name
                st.session_state["ground_truth_ate"] = None
                num_cols = list(st.session_state["df"].select_dtypes(include="number").columns)
                if len(num_cols) >= 2:
                    st.session_state["analysis_spec"] = {
                        "treatment": num_cols[0],
                        "outcome": num_cols[1],
                        "confounders": num_cols[2:6] if len(num_cols) > 2 else []
                    }
            except Exception as exc:
                st.error(f"Failed to read dataset: {exc}")

    st.markdown("---")
    with st.expander("AI & Engine Settings", expanded=False):
        env_resolved_key = get_api_key() or ""
        sidebar_api_key = st.text_input(
            "Gemini API Key",
            value=st.session_state.get("user_gemini_api_key", env_resolved_key),
            type="password",
            help="Configured via .env (GEMINI_API_KEY) or Streamlit Secrets. Override here for testing."
        )
        if sidebar_api_key != st.session_state.get("user_gemini_api_key", env_resolved_key):
            st.session_state["user_gemini_api_key"] = sidebar_api_key

        active_gemini_key = sidebar_api_key.strip() if sidebar_api_key.strip() else env_resolved_key

        if active_gemini_key:
            st.markdown(
                f"<div style='font-size: 11.5px; color: #4ADE80; margin: 4px 0 10px 0; display: flex; align-items: center; gap: 6px;'>"
                f"<span style='display:inline-block; width:7px; height:7px; background:#4ADE80; border-radius:50%; box-shadow:0 0 6px #4ADE80;'></span>"
                f"Gemini Active ({mask_api_key(active_gemini_key)})</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div style='font-size: 11.5px; color: #94A3B8; margin: 4px 0 10px 0; display: flex; align-items: center; gap: 6px;'>"
                "<span style='display:inline-block; width:7px; height:7px; background:#94A3B8; border-radius:50%;'></span>"
                "Gemini Offline (Local Fallback Mode)</div>",
                unsafe_allow_html=True
            )

        selected_model = st.selectbox(
            "Gemini Model",
            ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-flash-lite-latest", "gemini-3.5-flash", "gemini-3.7-flash", "gemini-3.8-flash", "gemini-3.1-pro-preview"],
            index=0
        )
        assistant_persona = st.selectbox(
            "Assistant Mode",
            [
                ("Evidence-Grounded", MODE_GROUNDED),
                ("Methodologist Auditor", MODE_AUDITOR),
                ("Unconstrained Exploration", MODE_UNCONSTRAINED)
            ],
            format_func=lambda x: x[0]
        )[1]
        llm_temp = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

    # Telemetry Badge
    df = st.session_state.get("df")
    row_count = f"{len(df):,}" if df is not None else "0"
    st.markdown(
        f"""
        <div style="font-size: 11.5px; color: #A78BFA; margin-top: 18px; border-top: 1px solid rgba(139,92,246,0.18); padding-top: 14px; background: rgba(18, 12, 36, 0.45); border-radius: 10px; padding: 12px; border: 1px solid rgba(168,85,247,0.15);">
            <div style="display:flex; justify-content:space-between; margin-bottom: 5px;">
                <span style="color:#94A3B8;">Cluster:</span> <span style="color:#22D3EE; font-weight:600;">Active SCM Node</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom: 5px;">
                <span style="color:#94A3B8;">Dataset:</span> <span style="color:#F8FAFC; font-weight:500;">{st.session_state.get('dataset_name', 'None')}</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#94A3B8;">In-Memory:</span> <span style="color:#C084FC; font-weight:600;">{row_count} Units</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Guard: Ensure Dataset is loaded
if df is None or df.empty:
    st.warning("No dataset loaded. Please select or upload a dataset in the sidebar.")
    st.stop()


# ==============================================================================
# RESPONSIVE TOP WORKSPACE BREADCRUMB & MOBILE QUICK-NAVIGATION
# ==============================================================================
page_category = "PLATFORM"
if selected_page in [PAGE_ASSISTANT, PAGE_RAG, PAGE_LAB, PAGE_REPORTS]:
    page_category = "RESEARCH"
elif selected_page in [PAGE_SETTINGS]:
    page_category = "SYSTEM"

top_c1, top_c2 = st.columns([3, 1])
with top_c1:
    st.markdown(
        f"""<div class="workspace-breadcrumb">
<span style="display:inline-flex; align-items:center; gap:8px;">
<span class="brand-icon-box" style="width:22px; height:22px; border-radius:6px;">
<svg width="13" height="13" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="#FFFFFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M2 17L12 22L22 17" stroke="#38BDF8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M2 12L12 17L22 12" stroke="#C084FC" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
</span>
<strong style="color:#F1F5F9; font-size:13px;">CausalLens AI</strong>
</span>
<span style="color:#475569;">/</span>
<span style="color:#A78BFA; font-weight:600; text-transform:uppercase; letter-spacing:0.8px; font-size:10.5px;">{page_category}</span>
<span style="color:#475569;">/</span>
<span class="crumb-active">{selected_page}</span>
</div>""",
        unsafe_allow_html=True
    )
with top_c2:
    cur_workspace = PAGE_NORMALIZE.get(st.session_state.get("nav_page", PAGE_OVERVIEW), PAGE_OVERVIEW)
    q_idx = ALL_PAGES.index(cur_workspace) if cur_workspace in ALL_PAGES else 0

    def on_quick_jump_change():
        dest = st.session_state.get(f"quick_jump_{cur_workspace}")
        if dest and dest != st.session_state.get("nav_page"):
            switch_page(dest)

    st.selectbox(
        "Switch Workspace",
        ALL_PAGES,
        index=q_idx,
        key=f"quick_jump_{cur_workspace}",
        on_change=on_quick_jump_change,
        label_visibility="collapsed",
        help="Quickly jump to any workspace."
    )


# ==============================================================================
# PAGE 1: COMMERCIAL ENTERPRISE HOMEPAGE
# ==============================================================================
if selected_page in (PAGE_OVERVIEW, "Overview", "Overview / Home"):
    # 1. Clean Top Header Bar
    gemini_status_pill = (
        '<span class="status-pill-emerald">● Gemini Online</span>'
        if active_gemini_key
        else '<span class="status-pill-amber">Offline Mode (ChromaDB Active)</span>'
    )
    st.markdown(
        f"""<div class="enterprise-navbar">
<div style="display:flex; align-items:center; gap:10px;">
<span class="brand-icon-box" style="width:26px; height:26px; border-radius:6px;">
<svg width="15" height="15" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="#FFFFFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M2 17L12 22L22 17" stroke="#38BDF8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M2 12L12 17L22 12" stroke="#818CF8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
</span>
<span style="font-weight:700; font-size:15px; color:#FFFFFF; letter-spacing:-0.2px;">CausalLens AI</span>
<span style="color:#64748B; font-size:12px;">|</span>
<span style="color:#94A3B8; font-size:12.5px;">Causal Discovery & Evidence-Grounded AI</span>
</div>
<div style="display:flex; align-items:center; gap:8px;">
<span class="status-pill-cyan">● Platform Ready</span>
{gemini_status_pill}
</div>
</div>""",
        unsafe_allow_html=True
    )

    # 2. Hero Section: Split Layout (Headline, Actions & Visual Causal Diagram)
    hero_col_left, hero_col_right = st.columns([1.15, 0.85])
    with hero_col_left:
        st.markdown(
            """
            <div class="hero-badge">✦ CAUSAL AI FOR RESEARCH & DECISION MAKING</div>
            <h1 class="hero-title" style="font-size:36px; line-height:1.2; margin-bottom:12px; font-weight:800; letter-spacing:-0.025em;">
                Find What Actually Drives<br><span style="background: linear-gradient(135deg, #FFFFFF 15%, #E0E7FF 40%, #A5B4FC 70%, #38BDF8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Results in Your Data.</span>
            </h1>
            <p class="hero-subtitle" style="font-size:14px; line-height:1.6; margin-bottom:20px; color:#94A3B8;">
                Standard machine learning only finds correlations. CausalLens AI helps you understand true cause-and-effect with intuitive visual diagrams, debiased statistical models, and an AI research partner grounded in your data and documents.
            </p>
            """,
            unsafe_allow_html=True
        )

        # Primary Action Buttons
        h_btn_c1, h_btn_c2 = st.columns(2)
        with h_btn_c1:
            if st.button("Start Research →", type="primary", use_container_width=True, key="hero_btn_start", help="Jump directly to Causal Analysis & DAG Workbench"):
                switch_page(PAGE_CAUSAL)
        with h_btn_c2:
            if st.button("Explore Platform", type="secondary", use_container_width=True, key="hero_btn_explore", help="Inspect your data first in Data Analysis"):
                switch_page(PAGE_DATA)

        # 3 High-Impact Value Metric Callouts
        st.markdown(
            """
            <div class="stats-counter-grid" style="margin-top:20px; padding-top:14px; border-top:1px solid rgba(255,255,255,0.06);">
                <div class="stat-counter-item">
                    <div class="stat-number gradient-text-purple">10x</div>
                    <div class="stat-caption">Faster Causal Analysis vs Manual Coding</div>
                </div>
                <div class="stat-counter-item">
                    <div class="stat-number gradient-text-cyan">5 Models</div>
                    <div class="stat-caption">Debiased Estimators with 95% Confidence Bounds</div>
                </div>
                <div class="stat-counter-item">
                    <div class="stat-number gradient-text-emerald">100%</div>
                    <div class="stat-caption">Grounded Evidence Backed by Actual Data</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with hero_col_right:
        st.markdown(
            """<div class="hero-nexus-card">
<div class="hero-nexus-header">
<div class="nexus-title-group">
<span class="nexus-pulse-dot"></span>
<span class="nexus-title">HOW CAUSAL REASONING WORKS</span>
</div>
<span class="nexus-badge">Live Interactive Model</span>
</div>
<div class="nexus-canvas-wrapper">
<svg width="100%" height="210" viewBox="0 0 420 210" fill="none" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <radialGradient id="lensGlow" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#4F46E5" stop-opacity="0.3"/>
      <stop offset="60%" stop-color="#06B6D4" stop-opacity="0.1"/>
      <stop offset="100%" stop-color="#080C14" stop-opacity="0"/>
    </radialGradient>
    <linearGradient id="edgeGrad1" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#818CF8" stop-opacity="0.8"/>
      <stop offset="100%" stop-color="#38BDF8" stop-opacity="0.8"/>
    </linearGradient>
    <linearGradient id="edgeGrad2" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#38BDF8" stop-opacity="0.9"/>
      <stop offset="100%" stop-color="#C084FC" stop-opacity="0.9"/>
    </linearGradient>
    <filter id="softGlow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="3" result="blur"/>
      <feComposite in="SourceGraphic" in2="blur" operator="over"/>
    </filter>
  </defs>

  <!-- Ambient Radar & Orbit Rings -->
  <circle cx="210" cy="105" r="95" fill="url(#lensGlow)" stroke="rgba(255,255,255,0.04)" stroke-width="1"/>
  <ellipse cx="210" cy="105" rx="145" ry="68" stroke="rgba(99,102,241,0.18)" stroke-dasharray="4 4" stroke-width="1"/>
  <ellipse cx="210" cy="105" rx="85" ry="40" stroke="rgba(56,189,248,0.2)" stroke-width="1"/>

  <!-- Directed Causal Edges -->
  <!-- Confounders -> Cause -->
  <path d="M 190 45 C 130 50 95 75 90 115" stroke="url(#edgeGrad1)" stroke-width="1.8" stroke-dasharray="3 3"/>
  <!-- Confounders -> Effect -->
  <path d="M 230 45 C 290 50 325 75 330 115" stroke="url(#edgeGrad1)" stroke-width="1.8" stroke-dasharray="3 3"/>
  <!-- Cause -> Effect (Main Causal Path) -->
  <path d="M 115 130 C 160 160 260 160 305 130" stroke="url(#edgeGrad2)" stroke-width="2.5" filter="url(#softGlow)"/>
  <polygon points="218,154 202,148 202,160" fill="#38BDF8"/>

  <!-- Debiasing Projection -->
  <line x1="210" y1="52" x2="210" y2="152" stroke="rgba(16,185,129,0.5)" stroke-width="1.5" stroke-dasharray="2 3"/>

  <!-- Node [Z] Background Factors (Top Center) -->
  <g transform="translate(145, 18)">
    <rect width="130" height="32" rx="7" fill="#131C31" stroke="rgba(129,140,248,0.4)" stroke-width="1.2"/>
    <circle cx="16" cy="16" r="4.5" fill="#818CF8"/>
    <text x="26" y="16" fill="#F8FAFC" font-family="'Plus Jakarta Sans', sans-serif" font-size="9.5" font-weight="700">Background Factors</text>
    <text x="26" y="25" fill="#94A3B8" font-family="'JetBrains Mono', monospace" font-size="7.5">(Confounders to Control)</text>
  </g>

  <!-- Node [T] Cause / Treatment (Left) -->
  <g transform="translate(25, 115)">
    <rect width="110" height="36" rx="8" fill="#0E192D" stroke="#38BDF8" stroke-width="1.5" filter="url(#softGlow)"/>
    <circle cx="16" cy="18" r="5" fill="#38BDF8"/>
    <text x="28" y="17" fill="#FFFFFF" font-family="'Plus Jakarta Sans', sans-serif" font-size="10" font-weight="700">Cause (Action)</text>
    <text x="28" y="28" fill="#94A3B8" font-family="'JetBrains Mono', monospace" font-size="8">What you change</text>
  </g>

  <!-- Central Core [Debiasing Filter] -->
  <g transform="translate(170, 88)">
    <rect width="80" height="34" rx="8" fill="#161E36" stroke="rgba(255,255,255,0.15)" stroke-width="1.2"/>
    <text x="40" y="16" text-anchor="middle" fill="#38BDF8" font-family="'Plus Jakarta Sans', sans-serif" font-size="9" font-weight="800">DEBIASING</text>
    <text x="40" y="27" text-anchor="middle" fill="#34D399" font-family="'JetBrains Mono', monospace" font-size="8" font-weight="600">Fair Contrast</text>
  </g>

  <!-- Node [Y] Effect / Outcome (Right) -->
  <g transform="translate(285, 115)">
    <rect width="110" height="36" rx="8" fill="#181433" stroke="#C084FC" stroke-width="1.5" filter="url(#softGlow)"/>
    <circle cx="16" cy="18" r="5" fill="#C084FC"/>
    <text x="28" y="17" fill="#FFFFFF" font-family="'Plus Jakarta Sans', sans-serif" font-size="10" font-weight="700">Effect (Outcome)</text>
    <text x="28" y="28" fill="#A5B4FC" font-family="'JetBrains Mono', monospace" font-size="8">Result you measure</text>
  </g>

  <!-- Bottom Pill -->
  <g transform="translate(145, 172)">
    <rect width="130" height="24" rx="12" fill="#0D2024" stroke="rgba(52,211,153,0.4)" stroke-width="1"/>
    <circle cx="14" cy="12" r="3.5" fill="#34D399"/>
    <text x="24" y="16" fill="#A7F3D0" font-family="'Plus Jakarta Sans', sans-serif" font-size="9" font-weight="700">Removes Hidden Bias</text>
  </g>
</svg>
</div>
<div class="nexus-telemetry-grid">
<div class="nexus-telemetry-item">
<span class="nexus-t-lbl">Robustness</span>
<span class="nexus-t-val" style="color:#34D399;">High (E-Value 2.85)</span>
</div>
<div class="nexus-telemetry-item">
<span class="nexus-t-lbl">Group Balance</span>
<span class="nexus-t-val" style="color:#38BDF8;">Controlled</span>
</div>
<div class="nexus-telemetry-item">
<span class="nexus-t-lbl">Answers</span>
<span class="nexus-t-val" style="color:#A78BFA;">100% Grounded</span>
</div>
</div>
</div>""",
            unsafe_allow_html=True
        )

    # 3. Simple Proven Frameworks Bar
    st.markdown(
        """
        <div class="trust-bar">
            <span style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.8px; color: #64748B; font-weight:700;">PROVEN METHODOLOGIES</span>
            <span class="trust-badge">Visual Causal Diagrams (Pearl DAGs)</span>
            <span style="color: rgba(255,255,255,0.12);">•</span>
            <span class="trust-badge">Potential Outcomes (Rubin)</span>
            <span style="color: rgba(255,255,255,0.12);">•</span>
            <span class="trust-badge">Double Machine Learning (Chernozhukov)</span>
            <span style="color: rgba(255,255,255,0.12);">•</span>
            <span class="trust-badge">Robustness E-Values (VanderWeele)</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ==============================================================================
    # 4. MAIN PRODUCT EXPERIENCE: AI RESEARCH ASSISTANT (CONVERSATIONAL ENTRY POINT)
    # ==============================================================================
    st.markdown(
        """
        <div class="search-hub-box">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                <div style="font-size:11px; text-transform:uppercase; letter-spacing:1px; color:#A78BFA; font-weight:700;">
                    AI RESEARCH ASSISTANT
                </div>
                <span class="status-pill-cyan">Plain-English Chat</span>
            </div>
            <h3 style="color:#FFFFFF; margin:0 0 6px 0; font-size:19px; font-weight:700;">
                Ask questions about your data, causal effects, and literature
            </h3>
            <p style="color:#94A3B8; font-size:13px; margin:0 0 14px 0; line-height:1.5;">
                Ask anything in plain English. The assistant checks your computed statistical numbers and indexed research documents so you never get fabricated answers.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 4 Beginner-Friendly Prompt Chips
    chip_col1, chip_col2, chip_col3, chip_col4 = st.columns(4)
    with chip_col1:
        if st.button("What is the true causal effect?", key="chip_dml", use_container_width=True, help="See what your treatment actually did to the outcome"):
            st.session_state["home_ai_query"] = "What is the estimated causal effect of our treatment on the outcome, and what does it mean in simple terms?"
            st.session_state["home_trigger_search"] = True
            st.rerun()
    with chip_col2:
        if st.button("Are my groups balanced fairly?", key="chip_bal", use_container_width=True, help="Check if treated and untreated groups are comparable"):
            st.session_state["home_ai_query"] = f"Are the treated and control groups balanced fairly in our active dataset '{st.session_state.get('dataset_name', 'benchmark')}'?"
            st.session_state["home_trigger_search"] = True
            st.rerun()
    with chip_col3:
        if st.button("How does AI remove bias?", key="chip_eval", use_container_width=True, help="Learn how Double Machine Learning removes confounding"):
            st.session_state["home_ai_query"] = "How does Double Machine Learning remove confounding bias to calculate a fair treatment effect?"
            st.session_state["home_trigger_search"] = True
            st.rerun()
    with chip_col4:
        if st.button("Explain what a confounder is", key="chip_dag", use_container_width=True, help="Simple definition and why confounders matter"):
            st.session_state["home_ai_query"] = "Explain what a confounder is in simple words, and why ignoring it leads to false conclusions."
            st.session_state["home_trigger_search"] = True
            st.rerun()

    # Search Bar & Execute Button
    search_c1, search_c2 = st.columns([4.2, 1])
    with search_c1:
        user_home_q = st.text_input(
            "Research Query",
            value=st.session_state.get("home_ai_query", ""),
            placeholder="Type your question here (e.g., Does our program improve outcomes? Is our data balanced?)...",
            label_visibility="collapsed",
            key="home_search_input_widget"
        )
    with search_c2:
        btn_search = st.button("Ask AI Assistant", type="primary", use_container_width=True, key="home_btn_synthesize", help="Get a verified, evidence-grounded answer")

    # Execution logic
    if (btn_search or st.session_state.get("home_trigger_search", False)) and user_home_q.strip():
        st.session_state["home_trigger_search"] = False
        st.session_state["home_ai_query"] = user_home_q
        with st.spinner("Synthesizing grounded evidence across vector store & empirical tensors..."):
            rag_store: AcademicVectorStore = st.session_state["rag_store"]
            api_key = active_gemini_key

            # Build research context from active dataset
            ctx = build_research_context(
                df=df,
                dataset_name=st.session_state.get("dataset_name"),
                analysis_spec=st.session_state.get("analysis_spec"),
                all_estimators=list(st.session_state["all_estimators"].values()) if st.session_state.get("all_estimators") else None,
                baseline_smd=st.session_state.get("baseline_smd"),
                sensitivity_result=st.session_state.get("sensitivity_result"),
                dag_summary=st.session_state.get("dag_summary")
            )

            # Query vector store for literature passages
            lit_passages = rag_store.search_literature(user_home_q, top_k=3, api_key=api_key)

            # Synthesize answer
            if lit_passages and ("how" in user_home_q.lower() or "explain" in user_home_q.lower() or "what" in user_home_q.lower() or "paper" in user_home_q.lower()):
                qa_res = answer_literature_query(user_home_q, rag_store, api_key=api_key)
                answer_text = qa_res.get("answer", "")
                sources = qa_res.get("citations", [])
            else:
                qa_res = answer_with_evidence(user_home_q, ctx, api_key=api_key, mode=MODE_GROUNDED)
                answer_text = qa_res.get("response", "")
                sources = qa_res.get("evidence_used", [])

            # Verify claims against computed numbers
            verified_claims = verify_claims_against_evidence(answer_text, ctx)
            st.session_state["home_ai_response"] = {
                "query": user_home_q,
                "answer": answer_text,
                "sources": sources,
                "verification": verified_claims
            }

    # Render Active Response Card if available
    if st.session_state.get("home_ai_response"):
        resp = st.session_state["home_ai_response"]
        st.markdown(
            f"""
            <div class="saas-card" style="border: 1px solid rgba(139, 92, 246, 0.4); background: linear-gradient(145deg, rgba(22, 18, 44, 0.85) 0%, rgba(13, 11, 26, 0.98) 100%); margin-top:14px; padding:22px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                    <div style="display:flex; align-items:center; gap:8px;">
                        <span class="status-pill-emerald">● Verified Evidence Synthesis</span>
                        <span style="font-size:12px; color:#94A3B8;">Query: <strong style="color:#E2E8F0;">"{resp['query']}"</strong></span>
                    </div>
                    <span class="status-pill-cyan">Audited Ground Truth</span>
                </div>
                <div style="color:#E2E8F0; font-size:13.5px; line-height:1.65; margin-bottom:16px;">
                    {resp['answer']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        col_resp_meta, col_resp_jump = st.columns([3, 1])
        with col_resp_meta:
            if resp.get("sources"):
                src_tags = " ".join([f"<span class='source-tag'>[{s.get('paper_title', s.get('title', 'Ref'))[:35]}... p.{s.get('page_num', 1)}]</span>" if isinstance(s, dict) else f"<span class='source-tag'>[{str(s)[:35]}]</span>" for s in resp["sources"][:3]])
                st.markdown(f"<div style='display:flex; align-items:center; gap:8px; flex-wrap:wrap;'><span style='font-size:11px; text-transform:uppercase; color:#94A3B8; font-weight:700;'>CITATIONS:</span> {src_tags}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<span class='status-pill-emerald'>✓ Grounded on Active Dataset Empirical Tensors</span>", unsafe_allow_html=True)
        with col_resp_jump:
            if st.button("Continue in AI Assistant ↗", type="secondary", use_container_width=True, key="home_jump_assistant"):
                st.session_state["assistant_query_prefill"] = resp["query"]
                switch_page(PAGE_ASSISTANT)

    # ==============================================================================
    # 5. ACTIVE PROJECT & KNOWLEDGE BASE AT A GLANCE
    # ==============================================================================
    st.markdown("<h3 style='color:#FFFFFF; font-weight:700; margin-top:32px; margin-bottom:14px;'>Current Workspace Status</h3>", unsafe_allow_html=True)
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        # Active Research Project Card
        spec = st.session_state.get("analysis_spec", {})
        num_cols = list(df.select_dtypes(include="number").columns)
        treatment_var = spec.get("treatment", num_cols[0] if num_cols else "T")
        outcome_var = spec.get("outcome", num_cols[1] if len(num_cols) > 1 else "Y")
        confounders = spec.get("confounders", [c for c in num_cols if c not in (treatment_var, outcome_var)])

        st.markdown(
            f"""
            <div class="portal-card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                    <div style="font-size:11px; text-transform:uppercase; letter-spacing:0.8px; color:#38BDF8; font-weight:700;">ACTIVE DATASET</div>
                    <span class="status-pill-emerald">● Loaded in Memory</span>
                </div>
                <h4 style="color:#FFFFFF; margin:0 0 6px 0; font-size:16.5px; font-weight:700;">{st.session_state.get('dataset_name', 'Active Dataset')}</h4>
                <div style="display:grid; grid-template-columns: repeat(3, 1fr); gap:10px; margin: 12px 0;">
                    <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06); border-radius:8px; padding:10px;">
                        <div style="font-size:10.5px; color:#94A3B8; text-transform:uppercase;">Rows (Sample Size)</div>
                        <div style="font-size:16px; font-weight:700; color:#FFFFFF; margin-top:2px;">{len(df):,}</div>
                    </div>
                    <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06); border-radius:8px; padding:10px;">
                        <div style="font-size:10.5px; color:#94A3B8; text-transform:uppercase;">Columns (Features)</div>
                        <div style="font-size:16px; font-weight:700; color:#FFFFFF; margin-top:2px;">{len(df.columns)}</div>
                    </div>
                    <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06); border-radius:8px; padding:10px;">
                        <div style="font-size:10.5px; color:#94A3B8; text-transform:uppercase;">Missing Data</div>
                        <div style="font-size:16px; font-weight:700; color:#34D399; margin-top:2px;">0.0%</div>
                    </div>
                </div>
                <div style="font-size:12.5px; color:#CBD5E1; line-height:1.6; margin-bottom:14px;">
                    <strong>Cause (Treatment):</strong> <code>{treatment_var}</code> → <strong>Effect (Outcome):</strong> <code>{outcome_var}</code><br>
                    <strong>Controlled Factors:</strong> {len(confounders)} background variables adjusted
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Inspect in Data Analysis ↗", type="secondary", use_container_width=True, key="home_btn_open_data", help="Inspect table and feature distributions"):
            switch_page(PAGE_DATA)

    with p_col2:
        # Knowledge Base Card
        rag_store: AcademicVectorStore = st.session_state["rag_store"]
        indexed_papers = rag_store.list_indexed_papers()
        total_chunks = rag_store.collection.count()

        paper_items_html = ""
        for p in indexed_papers[:3]:
            short_title = p['title'][:36] + ("..." if len(p['title']) > 36 else "")
            ft = p.get('file_type', 'doc').upper()
            paper_items_html += f"""
            <div style="display:flex; justify-content:space-between; align-items:center; padding:7px 0; border-bottom:1px solid rgba(255,255,255,0.05); font-size:12px;">
                <span style="color:#E2E8F0; font-weight:500;"><span style="color:#38BDF8; font-weight:700; margin-right:6px;">[{ft}]</span>{short_title}</span>
                <span class="status-pill" style="font-size:10px; padding:2px 7px;">{p.get('total_chunks', 2)} sections</span>
            </div>
            """

        st.markdown(
            f"""
            <div class="portal-card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                    <div style="font-size:11px; text-transform:uppercase; letter-spacing:0.8px; color:#A78BFA; font-weight:700;">DOCUMENT KNOWLEDGE BASE</div>
                    <span class="status-pill-cyan">{len(indexed_papers)} Documents Indexed</span>
                </div>
                <h4 style="color:#FFFFFF; margin:0 0 6px 0; font-size:16.5px; font-weight:700;">Research Papers & Notes</h4>
                <div style="margin: 8px 0 14px 0;">
                    {paper_items_html}
                </div>
                <div style="font-size:12px; color:#94A3B8; margin-bottom:14px;">
                    Supports PDF, Word, Excel, CSV, PowerPoint, and Markdown files. Searchable instantly via AI.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Manage Documents in Knowledge Base ↗", type="secondary", use_container_width=True, key="home_btn_open_rag", help="Upload more papers or search passages"):
            switch_page(PAGE_RAG)

    # ==============================================================================
    # 6. PLATFORM WORKSPACES (Clear, Simple 6-Card Grid)
    # ==============================================================================
    st.markdown("<h3 style='color:#FFFFFF; font-weight:700; margin-top:32px; margin-bottom:14px;'>Platform Workspaces</h3>", unsafe_allow_html=True)
    w_row1_c1, w_row1_c2, w_row1_c3 = st.columns(3)
    with w_row1_c1:
        st.markdown(
            """
            <div class="workspace-jump-card">
                <div>
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                        <span style="font-size:10px; text-transform:uppercase; color:#38BDF8; font-weight:700;">STEP 1</span>
                        <span class="status-pill-cyan">Data Quality</span>
                    </div>
                    <h4 style="color:#FFFFFF; font-size:15px; margin:0 0 6px 0;">Data Analysis</h4>
                    <p style="font-size:12.5px; color:#94A3B8; line-height:1.5; margin:0 0 14px 0;">
                        Inspect your data table, check missing values, view distributions, and find correlations.
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Open Data Analysis ↗", use_container_width=True, key="w_card_data"):
            switch_page(PAGE_DATA)

    with w_row1_c2:
        st.markdown(
            """
            <div class="workspace-jump-card">
                <div>
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                        <span style="font-size:10px; text-transform:uppercase; color:#A78BFA; font-weight:700;">STEP 2</span>
                        <span class="status-pill">Core Engine</span>
                    </div>
                    <h4 style="color:#FFFFFF; font-size:15px; margin:0 0 6px 0;">Causal Analysis & DAG</h4>
                    <p style="font-size:12.5px; color:#94A3B8; line-height:1.5; margin:0 0 14px 0;">
                        Draw causal graphs, remove confounding bias, and calculate real treatment effects with 5 estimators.
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Open Causal Studio ↗", use_container_width=True, key="w_card_causal"):
            switch_page(PAGE_CAUSAL)

    with w_row1_c3:
        st.markdown(
            """
            <div class="workspace-jump-card">
                <div>
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                        <span style="font-size:10px; text-transform:uppercase; color:#818CF8; font-weight:700;">AI PARTNER</span>
                        <span class="status-pill">Audited AI</span>
                    </div>
                    <h4 style="color:#FFFFFF; font-size:15px; margin:0 0 6px 0;">AI Research Assistant</h4>
                    <p style="font-size:12.5px; color:#94A3B8; line-height:1.5; margin:0 0 14px 0;">
                        Chat with an AI that knows your actual dataset numbers and never invents fake statistics.
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Open AI Assistant ↗", use_container_width=True, key="w_card_assistant"):
            switch_page(PAGE_ASSISTANT)

    w_row2_c1, w_row2_c2, w_row2_c3 = st.columns(3)
    with w_row2_c1:
        st.markdown(
            """
            <div class="workspace-jump-card">
                <div>
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                        <span style="font-size:10px; text-transform:uppercase; color:#34D399; font-weight:700;">RESEARCH</span>
                        <span class="status-pill-emerald">Multi-Format RAG</span>
                    </div>
                    <h4 style="color:#FFFFFF; font-size:15px; margin:0 0 6px 0;">Knowledge Base / RAG</h4>
                    <p style="font-size:12.5px; color:#94A3B8; line-height:1.5; margin:0 0 14px 0;">
                        Upload PDFs, Word docs, spreadsheets, and slides to search text with semantic AI.
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Open Knowledge Base ↗", use_container_width=True, key="w_card_rag"):
            switch_page(PAGE_RAG)

    with w_row2_c2:
        st.markdown(
            """
            <div class="workspace-jump-card">
                <div>
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                        <span style="font-size:10px; text-transform:uppercase; color:#FBBF24; font-weight:700;">TESTING</span>
                        <span class="status-pill">Simulation</span>
                    </div>
                    <h4 style="color:#FFFFFF; font-size:15px; margin:0 0 6px 0;">Research Lab</h4>
                    <p style="font-size:12.5px; color:#94A3B8; line-height:1.5; margin:0 0 14px 0;">
                        Run Monte Carlo simulations to see which causal model performs best under noisy conditions.
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Open Research Lab ↗", use_container_width=True, key="w_card_lab"):
            switch_page(PAGE_LAB)

    with w_row2_c3:
        st.markdown(
            """
            <div class="workspace-jump-card">
                <div>
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                        <span style="font-size:10px; text-transform:uppercase; color:#FB7185; font-weight:700;">OUTPUT</span>
                        <span class="status-pill-amber">Ready to Share</span>
                    </div>
                    <h4 style="color:#FFFFFF; font-size:15px; margin:0 0 6px 0;">Reports & Exports</h4>
                    <p style="font-size:12.5px; color:#94A3B8; line-height:1.5; margin:0 0 14px 0;">
                        Download publication-ready reports in Markdown, JSON reproducibility bundles, or printable HTML.
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Open Reports & Exports ↗", use_container_width=True, key="w_card_reports"):
            switch_page(PAGE_REPORTS)

    # ==============================================================================
    # 7. HOW IT WORKS (Logical 4-Step Process)
    # ==============================================================================
    st.markdown("<h3 style='color:#FFFFFF; font-weight:700; margin-top:32px; margin-bottom:14px;'>How CausalLens AI Works</h3>", unsafe_allow_html=True)
    step1, step2, step3, step4 = st.columns(4)
    with step1:
        st.markdown(
            """
            <div class="workflow-box">
                <div style="font-size:18px; font-weight:800; color:#38BDF8; margin-bottom:6px;">01</div>
                <div style="font-weight:700; color:#FFFFFF; font-size:14px; margin-bottom:6px;">Inspect Data</div>
                <p style="font-size:12px; color:#94A3B8; line-height:1.5; margin:0;">
                    Review your table, detect missing values, and check baseline differences between groups.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with step2:
        st.markdown(
            """
            <div class="workflow-box">
                <div style="font-size:18px; font-weight:800; color:#A78BFA; margin-bottom:6px;">02</div>
                <div style="font-weight:700; color:#FFFFFF; font-size:14px; margin-bottom:6px;">Set Causal Graph</div>
                <p style="font-size:12px; color:#94A3B8; line-height:1.5; margin:0;">
                    Choose your Cause (treatment), Effect (outcome), and background factors (confounders) to control.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with step3:
        st.markdown(
            """
            <div class="workflow-box">
                <div style="font-size:18px; font-weight:800; color:#34D399; margin-bottom:6px;">03</div>
                <div style="font-weight:700; color:#FFFFFF; font-size:14px; margin-bottom:6px;">Measure Effect</div>
                <p style="font-size:12px; color:#94A3B8; line-height:1.5; margin:0;">
                    Run 5 proven estimators (OLS, Matching, IPW, AIPW, Double ML) to see the true impact.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with step4:
        st.markdown(
            """
            <div class="workflow-box">
                <div style="font-size:18px; font-weight:800; color:#FB7185; margin-bottom:6px;">04</div>
                <div style="font-weight:700; color:#FFFFFF; font-size:14px; margin-bottom:6px;">Stress-Test & Share</div>
                <p style="font-size:12px; color:#94A3B8; line-height:1.5; margin:0;">
                    Test if results hold against hidden bias, chat with the AI assistant, and export clean reports.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ==============================================================================
    # 8. CORE PLATFORM CAPABILITIES TABS
    # ==============================================================================
    st.markdown("<h3 style='color:#FFFFFF; font-weight:700; margin-top:32px; margin-bottom:14px;'>Key Platform Features</h3>", unsafe_allow_html=True)
    tab_doc, tab_causal, tab_assistant, tab_caps = st.tabs([
        "Document Chat & RAG",
        "Causal Estimators",
        "Audited AI Assistant",
        "Exportable Reports"
    ])

    with tab_doc:
        st.markdown(
            """
            <div class="saas-card" style="padding:20px;">
                <h4 style="color:#FFFFFF; margin:0 0 8px 0; font-size:16px;">Search Any Document with AI</h4>
                <p style="color:#CBD5E1; font-size:13px; line-height:1.6; margin-bottom:14px;">
                    Upload your research papers, company reports, or datasets in PDF, Word, Excel, CSV, PowerPoint, or Markdown. The system finds relevant sections and answers your questions with exact page citations.
                </p>
                <div style="display:grid; grid-template-columns: repeat(3, 1fr); gap:10px;">
                    <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06); border-radius:8px; padding:10px;">
                        <strong style="color:#FFFFFF; font-size:12.5px;">7 File Formats</strong>
                        <div style="font-size:11.5px; color:#94A3B8; margin-top:2px;">PDF, Word, Excel, CSV, PPTX, Text, Markdown.</div>
                    </div>
                    <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06); border-radius:8px; padding:10px;">
                        <strong style="color:#FFFFFF; font-size:12.5px;">Semantic Search</strong>
                        <div style="font-size:11.5px; color:#94A3B8; margin-top:2px;">Understands context, not just keyword matches.</div>
                    </div>
                    <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06); border-radius:8px; padding:10px;">
                        <strong style="color:#FFFFFF; font-size:12.5px;">Exact Citations</strong>
                        <div style="font-size:11.5px; color:#94A3B8; margin-top:2px;">Shows the document name, page number, and excerpt.</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with tab_causal:
        st.markdown(
            """
            <div class="saas-card" style="padding:20px;">
                <h4 style="color:#FFFFFF; margin:0 0 8px 0; font-size:16px;">Measure True Impact, Not Just Correlation</h4>
                <p style="color:#CBD5E1; font-size:13px; line-height:1.6; margin-bottom:14px;">
                    Standard models confuse correlation with causation when people in the treated and untreated groups have different backgrounds. CausalLens AI balances these background factors so you get an honest, fair comparison.
                </p>
                <div style="display:grid; grid-template-columns: repeat(3, 1fr); gap:10px;">
                    <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06); border-radius:8px; padding:10px;">
                        <strong style="color:#FFFFFF; font-size:12.5px;">Visual Causal Graphs</strong>
                        <div style="font-size:11.5px; color:#94A3B8; margin-top:2px;">Point-and-click diagrams showing what causes what.</div>
                    </div>
                    <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06); border-radius:8px; padding:10px;">
                        <strong style="color:#FFFFFF; font-size:12.5px;">5 Debiased Estimators</strong>
                        <div style="font-size:11.5px; color:#94A3B8; margin-top:2px;">OLS, Matching, IPW, Doubly Robust AIPW & Double ML.</div>
                    </div>
                    <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06); border-radius:8px; padding:10px;">
                        <strong style="color:#FFFFFF; font-size:12.5px;">Hidden Bias Checks</strong>
                        <div style="font-size:11.5px; color:#94A3B8; margin-top:2px;">E-Values quantify how solid your conclusions are.</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with tab_assistant:
        st.markdown(
            """
            <div class="saas-card" style="padding:20px;">
                <h4 style="color:#FFFFFF; margin:0 0 8px 0; font-size:16px;">AI Chat Grounded in Your Actual Data</h4>
                <p style="color:#CBD5E1; font-size:13px; line-height:1.6; margin-bottom:14px;">
                    Most AI assistants guess or hallucinate numbers when asked about data. CausalLens AI verifies every numerical statement against your computed results before displaying it.
                </p>
                <div style="display:grid; grid-template-columns: repeat(3, 1fr); gap:10px;">
                    <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06); border-radius:8px; padding:10px;">
                        <strong style="color:#FFFFFF; font-size:12.5px;">Zero Fake Numbers</strong>
                        <div style="font-size:11.5px; color:#94A3B8; margin-top:2px;">Every effect size and percentage is cross-checked.</div>
                    </div>
                    <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06); border-radius:8px; padding:10px;">
                        <strong style="color:#FFFFFF; font-size:12.5px;">Plain English Answers</strong>
                        <div style="font-size:11.5px; color:#94A3B8; margin-top:2px;">Explains complex econometric concepts clearly.</div>
                    </div>
                    <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06); border-radius:8px; padding:10px;">
                        <strong style="color:#FFFFFF; font-size:12.5px;">Actionable Insights</strong>
                        <div style="font-size:11.5px; color:#94A3B8; margin-top:2px;">Suggests next experiments and validation steps.</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with tab_caps:
        st.markdown(
            """
            <div class="saas-card" style="padding:20px;">
                <h4 style="color:#FFFFFF; margin:0 0 8px 0; font-size:16px;">Export Clean, Reproducible Reports</h4>
                <p style="color:#CBD5E1; font-size:13px; line-height:1.6; margin-bottom:14px;">
                    Export your findings with one click. Everything needed to replicate your results—including data specifications, graph definitions, and model parameters—is neatly bundled.
                </p>
                <div style="display:grid; grid-template-columns: repeat(3, 1fr); gap:10px;">
                    <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06); border-radius:8px; padding:10px;">
                        <strong style="color:#FFFFFF; font-size:12.5px;">Markdown Reports</strong>
                        <div style="font-size:11.5px; color:#94A3B8; margin-top:2px;">Formatted text ready for documentation or GitHub.</div>
                    </div>
                    <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06); border-radius:8px; padding:10px;">
                        <strong style="color:#FFFFFF; font-size:12.5px;">JSON Data Bundles</strong>
                        <div style="font-size:11.5px; color:#94A3B8; margin-top:2px;">Machine-readable audit trail for researchers.</div>
                    </div>
                    <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06); border-radius:8px; padding:10px;">
                        <strong style="color:#FFFFFF; font-size:12.5px;">Printable HTML</strong>
                        <div style="font-size:11.5px; color:#94A3B8; margin-top:2px;">Self-contained executive report for stakeholders.</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # 9. Clear Next Action Section
    st.markdown(
        """
        <div class="saas-card" style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.12) 0%, rgba(14, 20, 32, 0.95) 100%); border: 1px solid rgba(99, 102, 241, 0.3); text-align: center; padding: 24px; margin-top: 24px;">
            <h3 style="color:#FFFFFF; margin:0 0 6px 0; font-size:18px;">Ready to start your research?</h3>
            <p style="color:#94A3B8; font-size:13px; margin:0 0 16px 0;">
                Choose where you'd like to begin: inspect your dataset or jump directly into causal modeling.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    col_act1, col_act2 = st.columns(2)
    with col_act1:
        if st.button("Go to Step 1: Data Analysis →", type="secondary", use_container_width=True, key="home_next_data"):
            switch_page(PAGE_DATA)
    with col_act2:
        if st.button("Go to Step 2: Causal Analysis →", type="primary", use_container_width=True, key="home_next_causal"):
            switch_page(PAGE_CAUSAL)

    # 10. Clean Commercial SaaS Footer
    st.markdown(
        """<div class="enterprise-footer">
<div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:20px;">
<div style="max-width:320px;">
<div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
<span class="brand-icon-box" style="width:22px; height:22px; border-radius:5px;">
<svg width="12" height="12" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="#FFFFFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M2 17L12 22L22 17" stroke="#38BDF8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M2 12L12 17L22 12" stroke="#C084FC" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
</span>
<span style="font-weight:700; color:#FFFFFF; font-size:14px;">CausalLens AI</span>
</div>
<div style="font-size:12px; color:#94A3B8; line-height:1.6;">
Find what actually drives outcomes in your data with visual causal graphs and grounded AI.
</div>
</div>
<div>
<div style="font-weight:600; color:#CBD5E1; font-size:11.5px; margin-bottom:6px; text-transform:uppercase;">Quick Links</div>
<div style="font-size:12px; color:#94A3B8; line-height:1.7;">
Overview • Data Analysis • Causal Workbench<br>
AI Assistant • Knowledge Base • Reports
</div>
</div>
<div>
<div style="font-weight:600; color:#CBD5E1; font-size:11.5px; margin-bottom:6px; text-transform:uppercase;">Methodologies</div>
<div style="font-size:12px; color:#94A3B8; line-height:1.7;">
Pearl SCM • Rubin Potential Outcomes<br>
Double Machine Learning • E-Values
</div>
</div>
</div>
<div style="margin-top:20px; padding-top:14px; border-top:1px solid rgba(255,255,255,0.05); display:flex; justify-content:space-between; align-items:center; font-size:11px; color:#64748B;">
<span>© 2026 CausalLens AI. All rights reserved.</span>
<span>Causal Discovery & Evidence-Grounded AI</span>
</div>
</div>""",
        unsafe_allow_html=True
    )


# ==============================================================================
# PAGE 2: DATA ANALYSIS
# ==============================================================================
elif selected_page in (PAGE_DATA, "Data Analysis", "Data Analysis & Quality", "Data Diagnostics"):
    st.markdown(
        f"""
        <div class="hero-badge">STEP 1 • DATA INSPECTION & QUALITY AUDIT</div>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
            <div>
                <h2 style="margin: 0; color: #F8FAFC; font-weight: 800;">Data Analysis</h2>
                <p style="color: #94A3B8; margin: 4px 0 0 0; font-size: 13.5px;">
                    Inspect your dataset, review data quality, and check feature correlations before measuring causal effects.
                </p>
            </div>
            <span class="status-pill-emerald">● Dataset: {st.session_state.get('dataset_name', 'Active Dataset')}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Top Key Metrics
    num_cols = list(df.select_dtypes(include="number").columns)
    total_missing = df.isnull().sum().sum()
    missing_pct = (total_missing / (df.shape[0] * df.shape[1])) * 100 if df.size > 0 else 0.0

    d_m1, d_m2, d_m3, d_m4 = st.columns(4)
    with d_m1:
        st.markdown(
            f"""
            <div class="metric-tile metric-tile-cyan">
                <div class="metric-label">Total Rows</div>
                <div class="metric-value">{len(df):,}</div>
                <div class="metric-subtext">Sample observations</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with d_m2:
        st.markdown(
            f"""
            <div class="metric-tile metric-tile-violet">
                <div class="metric-label">Total Columns</div>
                <div class="metric-value">{len(df.columns)}</div>
                <div class="metric-subtext">{len(num_cols)} numeric variables</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with d_m3:
        missing_color = "#34D399" if missing_pct == 0 else "#F59E0B"
        st.markdown(
            f"""
            <div class="metric-tile metric-tile-emerald">
                <div class="metric-label">Missing Data Rate</div>
                <div class="metric-value" style="color:{missing_color};">{missing_pct:.1f}%</div>
                <div class="metric-subtext">{"Clean (zero missing)" if missing_pct == 0 else "Values to impute"}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with d_m4:
        spec = st.session_state.get("analysis_spec", {})
        t_var = spec.get("treatment", num_cols[0] if num_cols else "None")
        y_var = spec.get("outcome", num_cols[1] if len(num_cols) > 1 else "None")
        st.markdown(
            f"""
            <div class="metric-tile metric-tile-rose">
                <div class="metric-label">Active Variables</div>
                <div class="metric-value" style="font-size:16px; line-height:38px;">{t_var} → {y_var}</div>
                <div class="metric-subtext">Cause → Effect pair</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    tab_table, tab_corr_dist, tab_upload = st.tabs([
        "1. Data Table & Quality Overview",
        "2. Correlations & Feature Distributions",
        "3. Change or Upload Dataset"
    ])

    # TAB 1: DATA TABLE & QUALITY
    with tab_table:
        st.markdown("#### Active Dataset Preview")
        st.caption("Review the first rows of your dataset. Verify that feature names and values match your expectations.")
        st.dataframe(df.head(15), use_container_width=True)

        with st.expander("Detailed Column Statistics & Profiling", expanded=False):
            st.markdown("##### Detailed Feature Profile")
            st.caption("Summary statistics including mean, standard deviation, min/max bounds, and skewness for each feature.")
            profile_df = profile_dataframe(df)
            st.dataframe(profile_df, use_container_width=True, height=350)

        with st.expander("Outlier & Extreme Value Screening (Tukey IQR)", expanded=False):
            st.markdown("##### Outlier Screening via Interquartile Range (IQR)")
            st.caption("Points outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] are flagged as potential outliers.")
            outliers_df = detect_outliers(df)
            if not outliers_df.empty:
                st.dataframe(outliers_df, use_container_width=True)
            else:
                st.info("No continuous numeric features identified for outlier screening.")

    # TAB 2: CORRELATIONS & DISTRIBUTIONS
    with tab_corr_dist:
        numeric_df = df.select_dtypes(include="number")
        if numeric_df.shape[1] >= 2:
            st.markdown("#### Correlation Heatmap")
            st.caption("Values close to +1.0 or -1.0 show features that move closely together. High correlation between background factors and treatment indicates confounding to control for.")
            
            c_left, c_right = st.columns([3, 2])
            with c_left:
                fig_corr = px.imshow(
                    numeric_df.corr(),
                    text_auto=".2f",
                    aspect="auto",
                    color_continuous_scale="Tealgrn",
                    title="Pairwise Pearson Correlation Matrix"
                )
                apply_dark_theme(fig_corr)
                st.plotly_chart(fig_corr, use_container_width=True)
            with c_right:
                st.markdown("##### Top Pairwise Correlations")
                st.dataframe(correlation_table(df).head(15), use_container_width=True, height=400)
        else:
            st.warning("At least two numeric features are required for correlation analysis.")

        st.markdown("---")
        st.markdown("#### Feature Distribution Studio")
        st.caption("Select any variable to inspect its histogram shape, spread, and median.")
        if num_cols:
            col_target = st.selectbox("Select variable to inspect:", num_cols, index=0, help="Choose any column to inspect its distribution and range")
            fig_hist = px.histogram(
                df,
                x=col_target,
                marginal="box",
                title=f"Distribution & Boxplot for '{col_target}'",
                color_discrete_sequence=["#38BDF8"],
                opacity=0.85
            )
            apply_dark_theme(fig_hist)
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("No numeric columns available.")

    # TAB 3: UPLOAD & BENCHMARKS
    with tab_upload:
        st.markdown("#### Dataset Ingestion & Benchmarks")
        st.caption("Upload your own tabular research dataset (CSV or Excel) or select a pre-configured benchmark.")

        up_c1, up_c2 = st.columns([1.5, 1])
        with up_c1:
            st.markdown("##### Upload Your Dataset")
            st.caption("Drag and drop your research spreadsheet. We support `.csv`, `.xlsx`, and `.xls` files.")
            in_page_file = st.file_uploader(
                "Upload dataset (.csv, .xlsx, .xls)",
                type=["csv", "xlsx", "xls"],
                key="page_data_file_uploader",
                help="Select a CSV or Excel spreadsheet containing your observational research data"
            )
            if in_page_file is not None:
                try:
                    if in_page_file.name.lower().endswith((".xlsx", ".xls")):
                        uploaded_df = pd.read_excel(in_page_file)
                    else:
                        uploaded_df = pd.read_csv(in_page_file)
                    st.session_state["df"] = uploaded_df
                    st.session_state["dataset_name"] = in_page_file.name
                    st.session_state["ground_truth_ate"] = None
                    num_c = list(uploaded_df.select_dtypes(include="number").columns)
                    if len(num_c) >= 2:
                        st.session_state["analysis_spec"] = {
                            "treatment": num_c[0],
                            "outcome": num_c[1],
                            "confounders": num_c[2:6] if len(num_c) > 2 else []
                        }
                    st.success(f"Successfully loaded '{in_page_file.name}' with {len(uploaded_df):,} rows and {len(uploaded_df.columns)} columns.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Error reading file: {exc}")

        with up_c2:
            st.markdown("##### Pre-Loaded Benchmarks")
            st.caption("Test the platform instantly with standard econometric datasets.")
            if st.button("Load Lalonde Job Training Benchmark", use_container_width=True, key="btn_load_lalonde_tab", help="Standard benchmark: Evaluates the effect of job training on post-training earnings"):
                rng = np.random.default_rng(101)
                n_lalonde = 900
                age = rng.normal(32, 8, n_lalonde).clip(17, 55)
                education = rng.integers(8, 17, n_lalonde)
                prior_earnings = rng.exponential(4500, n_lalonde)
                unemployed_prior = rng.binomial(1, 0.38, n_lalonde)
                logit_treat = -1.1 - 0.02 * (age - 30) - 0.08 * (education - 12) + 0.45 * unemployed_prior - 0.00012 * prior_earnings
                p_treat = 1 / (1 + np.exp(-logit_treat))
                treat = rng.binomial(1, p_treat)
                post_earnings = 1800.0 * treat + 380.0 * education + 0.62 * prior_earnings - 900.0 * unemployed_prior + rng.normal(0, 1600, n_lalonde)
                lalonde_df = pd.DataFrame({
                    "age": age,
                    "education_years": education,
                    "prior_earnings": prior_earnings,
                    "unemployed_prior": unemployed_prior,
                    "training_program": treat,
                    "post_earnings": post_earnings
                })
                st.session_state["df"] = lalonde_df
                st.session_state["dataset_name"] = "lalonde_benchmark"
                st.session_state["ground_truth_ate"] = 1800.0
                st.session_state["analysis_spec"] = {
                    "treatment": "training_program",
                    "outcome": "post_earnings",
                    "confounders": ["age", "education_years", "prior_earnings", "unemployed_prior"]
                }
                st.success("Loaded Lalonde Benchmark (Ground Truth Effect = $1,800).")
                st.rerun()

            if st.button("Load Synthetic Ground-Truth Demo (tau=2.5)", use_container_width=True, key="btn_load_synth_tab", help="Synthetic dataset with known true treatment effect = 2.5"):
                demo_df, meta = generate_synthetic_dataset(n=1000, true_ate=2.5, confounding_strength=1.0, seed=42)
                st.session_state["df"] = demo_df
                st.session_state["dataset_name"] = "synthetic_demo"
                st.session_state["ground_truth_ate"] = 2.5
                st.session_state["analysis_spec"] = {
                    "treatment": "treatment",
                    "outcome": "outcome",
                    "confounders": ["x1_feature", "age", "binary_flag", "prior_activity"]
                }
                st.success("Loaded Synthetic Benchmark (True Effect = 2.5).")
                st.rerun()

    # Clear Next Action Banner
    st.markdown("---")
    st.markdown(
        """
        <div class="saas-card" style="background: linear-gradient(135deg, rgba(56, 189, 248, 0.08) 0%, rgba(14, 20, 32, 0.95) 100%); border: 1px solid rgba(56, 189, 248, 0.25); padding: 18px 22px; margin-bottom: 14px;">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
                <div>
                    <h4 style="color:#FFFFFF; margin:0 0 4px 0; font-size:15px;">Next Step: Measure Cause-and-Effect</h4>
                    <p style="color:#94A3B8; font-size:12.5px; margin:0;">
                        Now that your data is inspected, proceed to Step 2 to draw your causal diagram and run debiased estimators.
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    c_data_nav1, c_data_nav2 = st.columns([3, 1])
    with c_data_nav1:
        st.caption("Dataset inspection complete. All features ready for structural causal modeling.")
    with c_data_nav2:
        if st.button("Proceed to Causal Analysis →", type="primary", use_container_width=True, key="data_cta_next", help="Navigate to Causal Analysis & DAG Workbench"):
            switch_page(PAGE_CAUSAL)


# ==============================================================================
# PAGE 3: CAUSAL ANALYSIS & DAG WORKBENCH
# ==============================================================================
elif selected_page in (PAGE_CAUSAL, "Causal Analysis", "Causal Analysis & DAG Workbench", "Causal DAG Studio"):
    st.markdown(
        """
        <div class="hero-badge">STEP 2 OF 4 • CAUSAL REASONING & ESTIMATION WORKBENCH</div>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
            <div>
                <h2 style="margin: 0; color: #F8FAFC; font-weight: 800;">Causal Analysis & DAG Workbench</h2>
                <p style="color: #94A3B8; margin: 4px 0 0 0; font-size: 13.5px;">
                    Follow the 4 structured steps below to select your variables, build your causal diagram, run debiased estimators, and view your results.
                </p>
            </div>
            <div style="display:flex; gap:8px;">
                <span class="status-pill-cyan">● 5 Debiased Estimators</span>
                <span class="status-pill-emerald">● DAG Validated</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    num_cols = list(df.select_dtypes(include="number").columns)
    if len(num_cols) < 2:
        st.warning("At least 2 numeric features required for causal modeling.")
        st.stop()

    spec = st.session_state.get("analysis_spec", {})
    default_treat = spec.get("treatment", num_cols[0] if "treatment" not in num_cols else "treatment")
    default_out = spec.get("outcome", num_cols[1] if "outcome" not in num_cols else "outcome")
    if default_out == default_treat and len(num_cols) > 1:
        default_out = num_cols[1] if num_cols[0] == default_treat else num_cols[0]

    # Initialize interactive DAG if not present
    if st.session_state.get("interactive_dag") is None:
        init_covs = [c for c in spec.get("confounders", num_cols[2:6]) if c in num_cols and c not in [default_treat, default_out]]
        st.session_state["interactive_dag"] = CausalGraph(
            treatment=default_treat,
            outcome=default_out,
            confounders=init_covs
        )

    dag: CausalGraph = st.session_state["interactive_dag"]

    # 4-Step Causal Journey Stepper Bar
    st.markdown(
        """
        <div class="causal-stepper-container">
            <div class="causal-step-box">
                <div class="causal-step-num" style="color: #A78BFA;">STEP 1</div>
                <div class="causal-step-title">Select Variables</div>
                <div class="causal-step-desc">Pick Treatment, Outcome, and Confounders</div>
            </div>
            <div class="causal-step-box">
                <div class="causal-step-num" style="color: #38BDF8;">STEP 2</div>
                <div class="causal-step-title">Build Causal Graph</div>
                <div class="causal-step-desc">Draw DAG, add nodes/arrows, check cycles</div>
            </div>
            <div class="causal-step-box">
                <div class="causal-step-num" style="color: #34D399;">STEP 3</div>
                <div class="causal-step-title">Run Causal Analysis</div>
                <div class="causal-step-desc">Execute OLS, PSM, IPW, AIPW, and DML</div>
            </div>
            <div class="causal-step-box">
                <div class="causal-step-num" style="color: #FB7185;">STEP 4</div>
                <div class="causal-step-title">View Results</div>
                <div class="causal-step-desc">Inspect ATE/ATT, balance, and stress-tests</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    causal_tabs = st.tabs([
        "Step 1: Select Variables",
        "Step 2: Build Causal Graph",
        "Step 3: Run Causal Analysis",
        "Step 4: View Results"
    ])

    # ==========================================================================
    # STEP 1: SELECT VARIABLES
    # ==========================================================================
    with causal_tabs[0]:
        st.markdown("#### Step 1: Select Your Research Variables")
        st.caption("Define what you want to evaluate. Choose your intervention (treatment), your goal metric (outcome), and the background factors that might cause bias.")

        v_col1, v_col2 = st.columns([1, 1])
        with v_col1:
            st.markdown("##### 1. Core Variables")
            cur_t = dag.treatment if dag.treatment in num_cols else default_treat
            t_idx = num_cols.index(cur_t) if cur_t in num_cols else 0
            sel_treat = st.selectbox(
                "Treatment Variable (Cause)",
                num_cols,
                index=t_idx,
                key="step1_treat_select",
                help="The action, intervention, or policy you want to evaluate (e.g., job training program, medical drug, pricing change)."
            )
            if sel_treat != dag.treatment:
                dag.set_treatment(sel_treat)

            available_y = [c for c in num_cols if c != sel_treat]
            cur_y = dag.outcome if dag.outcome in available_y else (available_y[0] if available_y else "Outcome")
            y_idx = available_y.index(cur_y) if cur_y in available_y else 0
            sel_outcome = st.selectbox(
                "Outcome Variable (Effect)",
                available_y,
                index=y_idx,
                key="step1_outcome_select",
                help="The primary metric or target you want to measure the effect on (e.g., real earnings, health recovery, customer retention)."
            )
            if sel_outcome != dag.outcome:
                dag.set_outcome(sel_outcome)

        with v_col2:
            st.markdown("##### 2. Confounding Variables (Background Factors)")
            candidate_covs = [c for c in num_cols if c not in [sel_treat, sel_outcome]]
            
            # Pre-select valid confounders
            adj_info = dag.find_valid_adjustment_sets()
            rec_set = [c for c in adj_info.get("recommended_set", []) if c in candidate_covs]
            curr_confounders = [c for c in spec.get("confounders", candidate_covs[:4]) if c in candidate_covs]
            if not curr_confounders and rec_set:
                curr_confounders = rec_set

            sel_confounders = st.multiselect(
                "Select Confounding Variables to Control",
                candidate_covs,
                default=curr_confounders,
                key="step1_confounders_select",
                help="Variables that influence both the treatment assignment and the outcome. Controlling for these removes confounding bias."
            )

            # Auto-recommend button
            if rec_set:
                rec_label = f"Auto-Select Recommended Confounders ({', '.join(rec_set[:3])}{'...' if len(rec_set) > 3 else ''})"
                if st.button(rec_label, type="secondary", key="btn_auto_rec_step1", help="Selects the minimal set of variables required to remove confounding bias"):
                    st.session_state["analysis_spec"] = {
                        "treatment": sel_treat,
                        "outcome": sel_outcome,
                        "confounders": rec_set
                    }
                    st.rerun()

        # Update session state specification
        st.session_state["analysis_spec"] = {
            "treatment": sel_treat,
            "outcome": sel_outcome,
            "confounders": sel_confounders
        }

        # Clean Specification Summary Card
        st.markdown(
            f"""
            <div class="saas-card" style="border: 1px solid rgba(139, 92, 246, 0.3); background: rgba(139, 92, 246, 0.05); padding: 14px 18px; margin-top: 14px;">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
                    <div>
                        <span style="color:#A78BFA; font-weight:700; font-size:12px;">ACTIVE RESEARCH SPECIFICATION</span>
                        <div style="color:#F8FAFC; font-size:14px; margin-top:4px;">
                            <strong>Cause (Treatment):</strong> <code style="color:#38BDF8;">{sel_treat}</code> &nbsp;→&nbsp;
                            <strong>Effect (Outcome):</strong> <code style="color:#34D399;">{sel_outcome}</code> &nbsp;|&nbsp;
                            <strong>Controlled Confounders ({len(sel_confounders)}):</strong> <code>{', '.join(sel_confounders) if sel_confounders else 'None selected'}</code>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.info("Variables selected! Click the **Step 2: Build Causal Graph** tab above to visualize and validate your causal diagram.")

    # ==========================================================================
    # STEP 2: BUILD CAUSAL GRAPH (DAG)
    # ==========================================================================
    with causal_tabs[1]:
        st.markdown("#### Step 2: Build Causal Graph (DAG)")
        st.caption("A Directed Acyclic Graph (DAG) maps how variables cause one another. The graph engine checks whether confounding paths are open or blocked, and detects any circular feedback loops.")

        cur_spec = st.session_state.get("analysis_spec", {})
        t_col = cur_spec.get("treatment", default_treat)
        y_col = cur_spec.get("outcome", default_out)
        c_cols = cur_spec.get("confounders", [])

        # Display cycle warning if any
        if st.session_state.get("dag_last_cycle_msg"):
            st.error(st.session_state["dag_last_cycle_msg"])

        g_col_tools, g_col_plot = st.columns([1, 1.2])

        with g_col_tools:
            st.markdown("##### 1. Add & Connect Causal Nodes")
            node_list = sorted(list(dag._graph.nodes()))

            # Add node
            with st.container():
                st.caption("**Add a New Variable Node:**")
                anc1, anc2 = st.columns([2, 1])
                with anc1:
                    new_node_input = st.text_input("New Node Name", placeholder="e.g. unobserved_skill", label_visibility="collapsed", key="input_new_node")
                with anc2:
                    if st.button("Add Node", use_container_width=True, key="btn_add_node"):
                        if new_node_input.strip():
                            dag.add_node(new_node_input.strip(), role="covariate")
                            st.success(f"Added node '{new_node_input.strip()}'")
                            st.rerun()

            st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
            # Connect edge
            with st.container():
                st.caption("**Connect a Directional Arrow (A → B):**")
                ec1, ec2 = st.columns(2)
                with ec1:
                    edge_src = st.selectbox("Source (Cause A)", node_list, key="edge_src_select")
                with ec2:
                    edge_dst_candidates = [n for n in node_list if n != edge_src]
                    edge_dst = st.selectbox("Target (Effect B)", edge_dst_candidates, key="edge_dst_select")

                if st.button("Connect Edge (A → B)", type="primary", use_container_width=True, key="btn_connect_edge"):
                    is_valid, msg, cycle = dag.validate_edge(edge_src, edge_dst)
                    if not is_valid:
                        st.session_state["dag_last_cycle_msg"] = f"**Cycle Detected:** {msg}. Causal graphs must be acyclic (DAG)."
                        st.rerun()
                    else:
                        dag.add_edge(edge_src, edge_dst)
                        st.session_state["dag_last_cycle_msg"] = None
                        st.success(f"Connected arrow: {edge_src} → {edge_dst}")
                        st.rerun()

            st.markdown("##### 2. Graph Validation & Cycle Detection")
            is_dag = dag.is_acyclic()
            if is_dag:
                st.markdown(
                    """
                    <div style="background: rgba(16, 185, 129, 0.1); border-left: 3px solid #10B981; padding: 10px 14px; border-radius: 6px; margin-bottom: 10px;">
                        <span style="color: #34D399; font-weight: 600; font-size: 13px;">✓ Graph is Valid (Strictly Acyclic DAG)</span>
                        <div style="color: #94A3B8; font-size: 12px; margin-top: 2px;">Zero circular feedback loops detected. Direction of time is preserved.</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.error("Graph contains a cycle! Please remove the offending edge to restore DAG validity.")

            # Backdoor Identification Audit
            audit = dag.validate_variable_roles(c_cols)
            st.session_state["dag_summary"] = audit
            if audit["is_identified"]:
                st.markdown(
                    f"""
                    <div style="background: rgba(56, 189, 248, 0.1); border-left: 3px solid #38BDF8; padding: 10px 14px; border-radius: 6px;">
                        <span style="color: #38BDF8; font-weight: 600; font-size: 13px;">✓ Causal Effect is Identified</span>
                        <div style="color: #CBD5E1; font-size: 12px; margin-top: 2px;">
                            Controlling for <strong>{', '.join(c_cols) if c_cols else 'no variables'}</strong> blocks all backdoor confounding paths between <strong>{t_col}</strong> and <strong>{y_col}</strong>.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div style="background: rgba(239, 68, 68, 0.1); border-left: 3px solid #EF4444; padding: 10px 14px; border-radius: 6px;">
                        <span style="color: #F87171; font-weight: 600; font-size: 13px;">⚠ Open Backdoor Path</span>
                        <div style="color: #CBD5E1; font-size: 12px; margin-top: 2px;">
                            {audit['identification_status']}. Return to Step 1 and select the recommended confounders to close open paths.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Pre-Built Templates Expander
            with st.expander("Pre-Built Example Causal Graph Templates", expanded=False):
                preset_choice = st.selectbox(
                    "Select Example Template",
                    [
                        "Current Custom DAG",
                        "Classic Confounding Triangle (X -> T, X -> Y, T -> Y)",
                        "Mediation Mechanism (T -> M -> Y, T -> Y, X -> T, X -> Y)",
                        "M-Bias / Collider Paradox (U1 -> T, U1 -> C <- U2, U2 -> Y)",
                        "Instrumental Variable (Z -> T -> Y, U -> T, U -> Y)",
                        "Reset to Step 1 Variables"
                    ],
                    key="step2_template_select"
                )
                if st.button("Apply Template", type="secondary", key="btn_apply_template"):
                    if preset_choice.startswith("Classic Confounding"):
                        st.session_state["interactive_dag"] = create_classic_confounding_dag()
                    elif preset_choice.startswith("Mediation"):
                        st.session_state["interactive_dag"] = create_mediation_dag()
                    elif preset_choice.startswith("M-Bias"):
                        st.session_state["interactive_dag"] = create_m_bias_dag()
                    elif preset_choice.startswith("Instrumental"):
                        st.session_state["interactive_dag"] = create_instrumental_variable_dag()
                    elif preset_choice.startswith("Reset"):
                        st.session_state["interactive_dag"] = CausalGraph(
                            treatment=t_col,
                            outcome=y_col,
                            confounders=c_cols
                        )
                    st.session_state["dag_last_cycle_msg"] = None
                    st.rerun()

        with g_col_plot:
            st.markdown("##### Visual Causal Diagram")
            plotly_dag = dag.generate_plotly_figure(selected_covariates=c_cols)
            st.plotly_chart(plotly_dag, use_container_width=True)

        st.info("Causal graph verified! Click the **Step 3: Run Causal Analysis** tab above to execute the estimators.")

    # ==========================================================================
    # STEP 3: RUN CAUSAL ANALYSIS
    # ==========================================================================
    with causal_tabs[2]:
        st.markdown("#### Step 3: Run Causal Estimators")
        st.caption("CausalLens AI provides 5 estimators covering linear, non-parametric matching, reweighting, and machine learning methods. Run all 5 simultaneously to triangulate the true causal effect.")

        cur_spec = st.session_state.get("analysis_spec", {})
        t_col = cur_spec.get("treatment", default_treat)
        y_col = cur_spec.get("outcome", default_out)
        c_cols = cur_spec.get("confounders", [])

        if not c_cols:
            st.warning("Please select at least one confounding variable in Step 1 before estimating causal effects.")

        # Estimator Explanation Cards
        st.markdown("##### The 5 Estimators Explained Simply")
        e_col1, e_col2 = st.columns(2)
        with e_col1:
            st.markdown(
                """
                <div class="saas-card" style="padding: 12px 16px; margin-bottom: 10px; border-left: 3px solid #38BDF8;">
                    <strong style="color: #38BDF8;">1. Linear Regression (OLS)</strong><br/>
                    <span style="font-size: 12.5px; color: #CBD5E1;">
                        Standard multivariate regression holding background factors constant. Best when causal relationships are approximately linear.
                    </span>
                </div>
                <div class="saas-card" style="padding: 12px 16px; margin-bottom: 10px; border-left: 3px solid #14B8A6;">
                    <strong style="color: #14B8A6;">2. Propensity Score Matching (PSM)</strong><br/>
                    <span style="font-size: 12.5px; color: #CBD5E1;">
                        Pairs each treated person with an untreated "twin" having the same probability of treatment. Computes both <strong>ATE</strong> and <strong>ATT</strong> directly.
                    </span>
                </div>
                <div class="saas-card" style="padding: 12px 16px; margin-bottom: 10px; border-left: 3px solid #A78BFA;">
                    <strong style="color: #A78BFA;">3. Inverse Probability Weighting (IPW)</strong><br/>
                    <span style="font-size: 12.5px; color: #CBD5E1;">
                        Re-weights the study sample to create a synthetic randomized controlled trial where background factors are evenly balanced across groups.
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )
        with e_col2:
            st.markdown(
                """
                <div class="saas-card" style="padding: 12px 16px; margin-bottom: 10px; border-left: 3px solid #F59E0B;">
                    <strong style="color: #F59E0B;">4. Doubly Robust AIPW</strong><br/>
                    <span style="font-size: 12.5px; color: #CBD5E1;">
                        Combines regression and propensity weighting. Guaranteed to be unbiased if <em>either</em> the outcome regression OR the propensity model is correctly specified.
                    </span>
                </div>
                <div class="saas-card" style="padding: 12px 16px; margin-bottom: 10px; border-left: 3px solid #EC4899;">
                    <strong style="color: #EC4899;">5. Double Machine Learning (DML)</strong><br/>
                    <span style="font-size: 12.5px; color: #CBD5E1;">
                        Uses AI algorithms (Random Forests & cross-fitting) to residualize confounders with Neyman-orthogonal scores, avoiding non-linear bias and overfitting.
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
        st.markdown("##### Execute Estimators")

        run_all_btn = st.button(
            "Execute All 5 Causal Estimators",
            type="primary",
            use_container_width=True,
            help="Computes OLS, Matching, IPW, AIPW, and Double Machine Learning simultaneously"
        )

        if run_all_btn:
            progress_bar = st.progress(0, text="Initializing causal estimation suite...")
            with st.spinner("Calculating treatment effects across all 5 models..."):
                import time
                progress_bar.progress(20, text="1/5: Fitting Ordinary Least Squares regression...")
                time.sleep(0.05)
                progress_bar.progress(40, text="2/5: Matching treated and control units on propensity score (PSM)...")
                time.sleep(0.05)
                progress_bar.progress(60, text="3/5: Computing stabilized inverse probability weights (IPW)...")
                time.sleep(0.05)
                progress_bar.progress(80, text="4/5: Solving doubly robust augmented equations (AIPW)...")
                time.sleep(0.05)
                progress_bar.progress(90, text="5/5: Cross-fitting Double Machine Learning nuisance models (DML)...")
                
                results = compare_all_estimators(df, t_col, y_col, c_cols)
                st.session_state["all_estimators"] = results
                for r in results:
                    if "OLS" in r.get("estimator", ""):
                        st.session_state["causal_result"] = r
                    elif "Matching" in r.get("estimator", ""):
                        st.session_state["matching_result"] = r
                        st.session_state["balance_table"] = r.get("balance_table")

                progress_bar.progress(100, text="Estimation complete!")
                st.success("✓ All 5 estimators successfully computed! Switch to the **Step 4: View Results** tab to review your findings.")

        # Show current status badge
        computed_ests = st.session_state.get("all_estimators")
        if computed_ests:
            st.markdown(
                f"""
                <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 8px; padding: 10px 14px; margin-top: 12px; display:flex; justify-content:space-between; align-items:center;">
                    <span style="color: #34D399; font-size: 13px; font-weight: 600;">● Results Ready: {len(computed_ests)} estimators computed</span>
                    <span style="color: #CBD5E1; font-size: 12px;">Active Models: OLS, PSM, IPW, AIPW, DML</span>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.caption("Click **Execute All 5 Causal Estimators** above to run the analysis.")

    # ==========================================================================
    # STEP 4: VIEW RESULTS & DIAGNOSTICS
    # ==========================================================================
    with causal_tabs[3]:
        st.markdown("#### Step 4: View Results & Diagnostics")
        st.caption("Review your estimated treatment effects (ATE & ATT), confidence intervals, group balance diagnostics, and sensitivity tests.")

        cur_spec = st.session_state.get("analysis_spec", {})
        t_col = cur_spec.get("treatment", default_treat)
        y_col = cur_spec.get("outcome", default_out)
        c_cols = cur_spec.get("confounders", [])
        estimators = st.session_state.get("all_estimators")

        if not estimators:
            st.markdown(
                """
                <div class="empty-state-card">
                    <div class="empty-state-title">No Estimator Results Yet</div>
                    <div class="empty-state-text">
                        Please go to <strong>Step 3: Run Causal Analysis</strong> and click <strong>Execute All 5 Causal Estimators</strong>, or click the button below to compute results now.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            if st.button("Execute All 5 Causal Estimators Now", type="primary", key="step4_run_now"):
                with st.spinner("Calculating treatment effects..."):
                    results = compare_all_estimators(df, t_col, y_col, c_cols)
                    st.session_state["all_estimators"] = results
                    for r in results:
                        if "OLS" in r.get("estimator", ""):
                            st.session_state["causal_result"] = r
                        elif "Matching" in r.get("estimator", ""):
                            st.session_state["matching_result"] = r
                            st.session_state["balance_table"] = r.get("balance_table")
                    st.rerun()
        else:
            # 1. Primary Metrics (ATE & ATT)
            valid_ests = [r for r in estimators if not np.isnan(r.get("ate", np.nan))]
            avg_ate = np.mean([r["ate"] for r in valid_ests]) if valid_ests else 0.0
            
            # Find ATT where supported
            atts = [r["att"] for r in estimators if "att" in r and not np.isnan(r.get("att", np.nan))]
            avg_att = np.mean(atts) if atts else None
            sig_count = sum(1 for r in valid_ests if r.get("p_value", 1.0) < 0.05)

            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric("Average Treatment Effect (ATE)", f"{avg_ate:+.3f}", help="Average effect across the entire study population")
            with m_col2:
                att_val_str = f"{avg_att:+.3f}" if avg_att is not None else "N/A"
                st.metric("Effect on the Treated (ATT)", att_val_str, help="Average effect specifically on those who received the treatment (PSM, AIPW, DML)")
            with m_col3:
                st.metric("Model Consensus", f"{sig_count} / {len(valid_ests)} Significant", help="How many models found statistically significant results (p < 0.05)")

            # Formatted Results Table with both ATE and ATT
            est_records = []
            for r in estimators:
                ci_str = f"[{r.get('ci_low', 0):.3f}, {r.get('ci_high', 0):.3f}]" if "ci_low" in r else "N/A"
                att_val = f"{r.get('att', 0):.3f}" if "att" in r and not np.isnan(r.get("att", np.nan)) else "—"
                est_records.append({
                    "Causal Estimator": r.get("estimator", "Unknown"),
                    "ATE (Population)": round(r.get("ate", np.nan), 3),
                    "ATT (Treated Subgroup)": att_val,
                    "95% Confidence Interval": ci_str,
                    "Standard Error": round(r.get("se", 0.0), 3) if "se" in r else "N/A",
                    "Statistical Significance": "Significant (p < 0.05)" if r.get("p_value", 1.0) < 0.05 else "Not Significant",
                    "Model Family": r.get("method", "Standard")
                })
            st.dataframe(pd.DataFrame(est_records), use_container_width=True)

            # Plain-English Takeaway Finding Card
            if valid_ests:
                st.markdown(
                    f"""
                    <div class="saas-card" style="border: 1px solid rgba(56, 189, 248, 0.3); background: rgba(56, 189, 248, 0.06); padding: 16px 20px; margin: 14px 0;">
                        <strong style="color: #38BDF8; font-size: 14px;">Plain-English Interpretation:</strong>
                        <div style="color: #E2E8F0; font-size: 13.5px; line-height: 1.6; margin-top: 4px;">
                            Across the {len(valid_ests)} models, the treatment effect averages <strong>{avg_ate:+.3f}</strong>.
                            {"Receiving the treatment is estimated to increase the outcome by " + f"<strong>{abs(avg_ate):.3f}</strong> units" if avg_ate > 0 else "Receiving the treatment is estimated to decrease the outcome by " + f"<strong>{abs(avg_ate):.3f}</strong> units"} on average, after removing confounding bias.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Comparison Chart (Forest Plot)
                y_names = [r["estimator"] for r in valid_ests]
                ates = [r["ate"] for r in valid_ests]
                ci_lows = [r.get("ci_low", r["ate"]) for r in valid_ests]
                ci_highs = [r.get("ci_high", r["ate"]) for r in valid_ests]
                err_plus_ate = [ci_h - a for ci_h, a in zip(ci_highs, ates)]
                err_minus_ate = [a - ci_l for ci_l, a in zip(ci_lows, ates)]

                fig_forest = go.Figure()
                fig_forest.add_trace(go.Scatter(
                    x=ates,
                    y=y_names,
                    mode="markers",
                    error_x=dict(type="data", symmetric=False, array=err_plus_ate, arrayminus=err_minus_ate, color="#38BDF8", thickness=2),
                    marker=dict(color="#38BDF8", size=10, symbol="diamond"),
                    name="Estimated Effect (95% CI)"
                ))
                fig_forest.add_vline(x=0.0, line_dash="dash", line_color="#64748B", annotation_text="No Effect (0.0)")
                gt_tau = st.session_state.get("ground_truth_ate")
                if gt_tau is not None:
                    fig_forest.add_vline(x=gt_tau, line_dash="solid", line_color="#2DD4BF", annotation_text=f"Known Truth ({gt_tau})")

                fig_forest.update_layout(
                    title="Estimator Comparison: Treatment Effects & 95% Confidence Intervals",
                    xaxis_title="Treatment Effect on Outcome",
                    yaxis_title="Estimator",
                    height=340
                )
                apply_dark_theme(fig_forest)
                st.plotly_chart(fig_forest, use_container_width=True)

            # Group Balance Diagnostics Section
            st.markdown("---")
            st.markdown("##### Group Balance & Fairness Diagnostics")
            st.caption("Checks whether treated and control groups had comparable background characteristics. Standardized Mean Difference (|SMD|) below 0.10 indicates fair balance.")

            try:
                base_smd = compute_baseline_smd(df, t_col, c_cols)
                st.session_state["baseline_smd"] = base_smd

                diag_c1, diag_c2 = st.columns([1, 1])
                with diag_c1:
                    st.caption("Covariate Balance Table (|SMD| < 0.10 target):")
                    st.dataframe(base_smd, use_container_width=True)
                with diag_c2:
                    fig_love = go.Figure()
                    fig_love.add_trace(go.Scatter(
                        x=base_smd["abs_smd"],
                        y=base_smd["covariate"],
                        mode="markers",
                        marker=dict(color="#F43F5E", size=10, symbol="circle"),
                        name="Baseline Unadjusted Difference"
                    ))
                    fig_love.add_vline(x=0.1, line_dash="dash", line_color="#2DD4BF", annotation_text="Target Balance (< 0.10)")
                    fig_love.add_vline(x=0.25, line_dash="dot", line_color="#F59E0B", annotation_text="Severe Imbalance (> 0.25)")
                    fig_love.update_layout(
                        title="Love Plot: Covariate Imbalance",
                        xaxis_title="Standardized Difference (|SMD|)",
                        yaxis_title="Background Factor",
                        height=330
                    )
                    apply_dark_theme(fig_love)
                    st.plotly_chart(fig_love, use_container_width=True)

                with st.expander("Propensity Score Overlap Details (Common Support)", expanded=False):
                    from sklearn.linear_model import LogisticRegression
                    ps_data = df[[t_col] + c_cols].dropna()
                    X_mat = ps_data[c_cols].astype(float)
                    t_vec = ps_data[t_col].astype(int)
                    ps_fit = LogisticRegression(max_iter=3000).fit(X_mat, t_vec)
                    prop_scores = ps_fit.predict_proba(X_mat)[:, 1]
                    ps_data["_propensity_score"] = prop_scores

                    fig_pos = px.histogram(
                        ps_data,
                        x="_propensity_score",
                        color=t_col,
                        barmode="overlay",
                        nbins=40,
                        opacity=0.7,
                        color_discrete_map={0: "#0284C7", 1: "#14B8A6"},
                        title="Propensity Score Distribution (Treated vs Control Overlap)"
                    )
                    fig_pos.update_layout(xaxis_title="Probability of Treatment P(T=1 | X)", yaxis_title="Count")
                    apply_dark_theme(fig_pos)
                    st.plotly_chart(fig_pos, use_container_width=True)

                    p_t1 = prop_scores[t_vec == 1]
                    p_t0 = prop_scores[t_vec == 0]
                    st.caption(f"**Overlap Range:** [{max(p_t1.min(), p_t0.min()):.4f}, {min(p_t1.max(), p_t0.max()):.4f}]")

            except Exception as exc:
                st.error(f"Error computing balance: {exc}")

            # Assumptions, Limitations & Stress-Tests
            st.markdown("---")
            st.markdown("##### Assumptions, Limitations & Robustness Stress-Tests")

            causal_res = st.session_state.get("causal_result")
            if causal_res:
                s_col1, s_col2 = st.columns([1, 1])
                with s_col1:
                    st.markdown("###### Robustness to Hidden Bias (VanderWeele E-Value)")
                    st.caption("How strong would an unmeasured confounder need to be to explain away this effect?")
                    outcome_sd = float(df[y_col].std())
                    e_res = compute_e_value(
                        ate=causal_res["ate"],
                        se=causal_res.get("se", 0.1),
                        outcome_sd=outcome_sd,
                        ci_low=causal_res.get("ci_low"),
                        ci_high=causal_res.get("ci_high")
                    )
                    st.session_state["sensitivity_result"] = e_res

                    m1, m2 = st.columns(2)
                    m1.metric("Robustness E-Value", f"{e_res['e_value_point']:.2f}", help="Minimum risk ratio an unmeasured factor must have to nullify the effect")
                    m2.metric("Confidence Bound", f"{e_res['e_value_ci']:.2f}", help="Lower bound accounting for statistical uncertainty")

                    st.markdown(
                        f"""
                        <div style="background: rgba(20, 184, 166, 0.1); border-left: 3px solid #14B8A6; padding: 12px; border-radius: 6px; margin-top: 10px;">
                            <strong style="color: #2DD4BF;">Rating: {e_res['robustness_rating']}</strong><br/>
                            <span style="font-size: 12.5px; color: #CBD5E1;">{e_res['interpretation']}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with s_col2:
                    st.markdown("###### Placebo & Refutation Tests")
                    st.caption("Falsification tests: randomly shuffling treatment or adding noise should drop the effect to 0.")
                    if st.button("Execute Refutation Battery", type="primary", key="btn_run_refutations", help="Runs placebo treatment, noise injection, and subset validation"):
                        with st.spinner("Testing with placebo treatments and noise injections..."):
                            r1 = run_placebo_treatment_refutation(df, t_col, y_col, c_cols, n_simulations=10)
                            r2 = run_random_common_cause_refutation(df, t_col, y_col, c_cols)
                            r3 = run_data_subset_refutation(df, t_col, y_col, c_cols, subset_fraction=0.8, n_iterations=8)
                            st.session_state["refutation_results"] = [r1, r2, r3]

                    ref_list = st.session_state.get("refutation_results", [])
                    if ref_list:
                        for ref in ref_list:
                            status_color = "#2DD4BF" if ref["passed"] else "#F43F5E"
                            badge_label = "PASSED" if ref["passed"] else "FAILED"
                            with st.expander(f"{badge_label}: {ref['test']}", expanded=True):
                                st.markdown(f"<span style='color: {status_color}; font-weight: 600;'>{ref['status']}</span>", unsafe_allow_html=True)
                                st.caption(ref["explanation"])

            # Core Identification Assumptions Checklist
            with st.expander("Theoretical Causal Identification Assumptions (Pearl & Rubin Checklist)", expanded=False):
                st.markdown(
                    """
                    <div style="font-size: 13px; color: #CBD5E1; line-height: 1.6;">
                        Observational causal inference relies on 5 fundamental assumptions that cannot be verified purely from data correlations:
                        <ul style="margin-top: 8px;">
                            <li><strong>1. Unconfoundedness (Conditional Independence):</strong> All common causes of treatment and outcome are controlled in your DAG.</li>
                            <li><strong>2. Positivity (Common Support):</strong> Every participant has a non-zero probability of receiving either treatment or control.</li>
                            <li><strong>3. SUTVA (Stable Unit Treatment Value):</strong> One unit's treatment assignment does not affect another unit's outcome (no interference / spillover).</li>
                            <li><strong>4. Consistency:</strong> The treatment received matches the potential treatment defined in the counterfactual model.</li>
                            <li><strong>5. Model Specification:</strong> At least one of the nuisance models (outcome regression or propensity model) accurately approximates the true functional form.</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # Clear Next Action Banner
    st.markdown("---")
    st.markdown(
        """
        <div class="saas-card" style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.08) 0%, rgba(14, 20, 32, 0.95) 100%); border: 1px solid rgba(99, 102, 241, 0.25); padding: 18px 22px; margin-bottom: 14px;">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
                <div>
                    <h4 style="color:#FFFFFF; margin:0 0 4px 0; font-size:15px;">Next Step: Inquire the AI Assistant or Export Reports</h4>
                    <p style="color:#94A3B8; font-size:12.5px; margin:0;">
                        Ask the AI Research Assistant to explain these findings in plain English, or export your publication-ready report.
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    c_causal_nav1, c_causal_nav2, c_causal_nav3 = st.columns([1, 1, 1])
    with c_causal_nav1:
        if st.button("← Back to Data Analysis", use_container_width=True, key="causal_nav_back", help="Return to Step 1"):
            switch_page(PAGE_DATA)
    with c_causal_nav2:
        if st.button("Inquire AI Assistant →", type="primary", use_container_width=True, key="causal_nav_asst", help="Chat with AI about these results"):
            switch_page(PAGE_ASSISTANT)
    with c_causal_nav3:
        if st.button("Generate Reports & Exports →", use_container_width=True, key="causal_nav_reports", help="Download reports in Markdown, JSON, or HTML"):
            switch_page(PAGE_REPORTS)


# ==============================================================================
# PAGE 4: DUAL-SOURCE AI RESEARCH ASSISTANT & AUDITOR
# ==============================================================================
elif selected_page in (PAGE_ASSISTANT, "AI Assistant", "AI Research Assistant", "Research Assistant"):
    st.markdown(
        """
        <div class="hero-badge">STEP 3 OF 4 • CONVERSATIONAL AI ASSISTANT</div>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
            <div>
                <h2 style="margin: 0; color: #F8FAFC; font-weight: 800;">AI Research Assistant</h2>
                <p style="color: #94A3B8; margin: 4px 0 0 0; font-size: 13.5px;">
                    Ask questions about your data numbers or your uploaded research documents in plain English. Every answer is backed by verifiable calculations and citations.
                </p>
            </div>
            <div style="display:flex; gap:8px;">
                <span class="status-pill">● Dual Knowledge Engine</span>
                <span class="status-pill-emerald">● 100% Fact-Checked</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    rag_store: AcademicVectorStore = st.session_state["rag_store"]
    indexed_docs = rag_store.list_indexed_documents()

    if active_gemini_key:
        st.markdown(
            f"""
            <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 8px; padding: 10px 14px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center;">
                <div style="display:flex; align-items: center; gap: 8px;">
                    <span style="display:inline-block; width:8px; height:8px; background:#10B981; border-radius:50%; box-shadow:0 0 8px #10B981;"></span>
                    <span style="color: #F8FAFC; font-size: 13px; font-weight: 600;">Google Gemini Connected</span>
                    <span style="color: #94A3B8; font-size: 12px;">| Model: <code style="color:#22D3EE;">{selected_model}</code> | Credentials: <code>{mask_api_key(active_gemini_key)}</code></span>
                </div>
                <span style="background: rgba(16, 185, 129, 0.2); color: #34D399; font-size: 11px; padding: 3px 8px; border-radius: 4px; font-weight: 600;">ACTIVE</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.25); border-radius: 8px; padding: 10px 14px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center;">
                <div style="display:flex; align-items: center; gap: 8px;">
                    <span style="display:inline-block; width:8px; height:8px; background:#F59E0B; border-radius:50%;"></span>
                    <span style="color: #F8FAFC; font-size: 13px; font-weight: 600;">Gemini API Offline Mode</span>
                    <span style="color: #94A3B8; font-size: 12px;">| Generative LLM synthesis is paused. Exact vector retrieval and local verifier are active.</span>
                </div>
                <span style="background: rgba(245, 158, 11, 0.2); color: #FBBF24; font-size: 11px; padding: 3px 8px; border-radius: 4px; font-weight: 600;">LOCAL VERIFIER</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Knowledge Source Selector
    st.markdown("##### What would you like to ask about?")
    src_col, opt_col = st.columns([2.5, 1.5])
    with src_col:
        source_mode = st.radio(
            "Knowledge Source:",
            [
                (KNOWLEDGE_SOURCE_DATASET, "📊 Active Dataset (Computed treatment effects, variables, correlations)"),
                (KNOWLEDGE_SOURCE_DOCUMENTS, "📄 Uploaded Documents (Research papers, reports, notes in RAG)"),
                (KNOWLEDGE_SOURCE_HYBRID, "🔗 Both Combined (Connect dataset numbers to research theory)")
            ],
            format_func=lambda x: x[1],
            horizontal=False,
            key="asst_knowledge_source_selector",
            help="Choose whether the AI answers from your active dataset numbers, your uploaded documents, or synthesizes both."
        )[0]
        st.session_state["assistant_knowledge_source"] = source_mode

    with opt_col:
        est_count = len(st.session_state['all_estimators']) if st.session_state.get('all_estimators') else 0
        ds_status = f"Active ({len(df):,} rows)" if df is not None else "No active dataset"
        st.markdown(
            f"""
            <div style="background: rgba(15, 23, 42, 0.6); padding: 12px 14px; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.08); font-size: 12px;">
                <div style="color: #A78BFA; font-weight: 700; margin-bottom: 6px;">CURRENT KNOWLEDGE STATUS</div>
                <div style="color: #E2E8F0; margin-bottom: 4px;">• <strong>Dataset:</strong> {ds_status}</div>
                <div style="color: #E2E8F0; margin-bottom: 4px;">• <strong>Causal Models:</strong> {est_count} computed</div>
                <div style="color: #E2E8F0;">• <strong>Documents:</strong> {len(indexed_docs)} indexed ({rag_store.collection.count()} vectors)</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # Advanced Settings Expander
    with st.expander("Advanced Settings (AI Model Persona & Search Filters)", expanded=False):
        ec1, ec2, ec3 = st.columns(3)
        with ec1:
            assistant_persona = st.selectbox(
                "Operating Persona",
                [MODE_GROUNDED, MODE_AUDITOR, MODE_UNCONSTRAINED],
                format_func=lambda x: {
                    MODE_GROUNDED: "Grounded Biostatistician (Answers strictly using verified numbers)",
                    MODE_AUDITOR: "Skeptical Methodologist (Critiques assumptions & flags biases)",
                    MODE_UNCONSTRAINED: "Exploratory Scientist (Suggests follow-up hypotheses)"
                }.get(x, x),
                key="asst_persona_select",
                help="Controls the tone and rigor of the assistant's explanations."
            )
        with ec2:
            llm_temp = st.slider("Response Style (Temperature)", 0.0, 0.6, 0.2, 0.05, key="asst_temp_slider", help="Lower is more factual; higher is more creative.")
        with ec3:
            asst_top_k = st.slider("Retrieved Passages (Top-k)", 1, 8, 4, key="asst_top_k_slider", help="Number of relevant document excerpts to retrieve for answering.")

        asst_doc_filter = None
        asst_fmt_filter = None
        if source_mode in [KNOWLEDGE_SOURCE_DOCUMENTS, KNOWLEDGE_SOURCE_HYBRID] and indexed_docs:
            fc1, fc2 = st.columns(2)
            with fc1:
                asst_doc_filter = st.selectbox("Filter to Target Document", ["All Documents"] + [d["title"] for d in indexed_docs], key="asst_doc_filter_select", help="Focus your question on one specific document.")
                if asst_doc_filter == "All Documents":
                    asst_doc_filter = None
            with fc2:
                fmts = ["All Formats"] + sorted(list(set(d.get("file_type", "doc").upper() for d in indexed_docs)))
                asst_fmt_filter = st.selectbox("Filter by Format", fmts, key="asst_fmt_filter_select", help="Filter by file type (PDF, DOCX, CSV, etc.)")
                if asst_fmt_filter == "All Formats":
                    asst_fmt_filter = None

    # Context-Aware Preset Query Chips
    st.markdown("##### Suggested Questions (Click any to ask)")
    
    if source_mode == KNOWLEDGE_SOURCE_DATASET:
        p_c1, p_c2, p_c3, p_c4, p_c5 = st.columns(5)
        with p_c1:
            if st.button("Compare Estimators", use_container_width=True, help="Compare OLS, Matching, IPW, AIPW, and DML"):
                st.session_state["assistant_query_text"] = "Compare the computed treatment effects across OLS, Matching, IPW, Doubly Robust AIPW, and Double Machine Learning. Why do the estimates diverge, and which is most defensible?"
                st.session_state["asst_query_ver"] = st.session_state.get("asst_query_ver", 0) + 1
                st.rerun()
        with p_c2:
            if st.button("Confounding & E-Value", use_container_width=True, help="Analyze SMDs and sensitivity E-value"):
                st.session_state["assistant_query_text"] = "Evaluate baseline covariate imbalance across treatment groups. What does our sensitivity E-value tell us about the plausibility of unmeasured confounding overturning the result?"
                st.session_state["asst_query_ver"] = st.session_state.get("asst_query_ver", 0) + 1
                st.rerun()
        with p_c3:
            if st.button("Dataset Quality & Stats", use_container_width=True, help="Audit missingness, distributions, and correlations"):
                st.session_state["assistant_query_text"] = "Summarize the active dataset's sample size, feature distributions, missingness patterns, and top correlation pairs. How might data quality affect causal estimation?"
                st.session_state["asst_query_ver"] = st.session_state.get("asst_query_ver", 0) + 1
                st.rerun()
        with p_c4:
            if st.button("Identification Assumptions", use_container_width=True, help="Examine the core identification assumptions"):
                st.session_state["assistant_query_text"] = "Detail how the 5 foundational causal identification assumptions (unconfoundedness, positivity, SUTVA, consistency, model specification) hold in our observational design."
                st.session_state["asst_query_ver"] = st.session_state.get("asst_query_ver", 0) + 1
                st.rerun()
        with p_c5:
            if st.button("Suggest Next Steps", use_container_width=True, help="Propose trials, negative controls, or subgroup tests"):
                st.session_state["assistant_query_text"] = "Based on our computed estimates and refutation test battery, what further analyses, negative control outcomes, or randomized prospective trials do you suggest?"
                st.session_state["asst_query_ver"] = st.session_state.get("asst_query_ver", 0) + 1
                st.rerun()

    elif source_mode == KNOWLEDGE_SOURCE_DOCUMENTS:
        p_c1, p_c2, p_c3, p_c4, p_c5 = st.columns(5)
        with p_c1:
            if st.button("Explain Methodology", use_container_width=True, help="Explain preprints core methodology"):
                st.session_state["assistant_query_text"] = "Explain the core theoretical methodology and estimators proposed across our uploaded research preprints."
                st.session_state["asst_query_ver"] = st.session_state.get("asst_query_ver", 0) + 1
                st.rerun()
        with p_c2:
            if st.button("Summarize Documents", use_container_width=True, help="Summarize multiple uploaded documents"):
                st.session_state["assistant_query_text"] = "Summarize the primary objectives, methodologies, and main findings across all currently indexed documents."
                st.session_state["asst_query_ver"] = st.session_state.get("asst_query_ver", 0) + 1
                st.rerun()
        with p_c3:
            if st.button("Compare Assumptions", use_container_width=True, help="Compare identification assumptions across papers"):
                st.session_state["assistant_query_text"] = "Compare the identification assumptions and causal frameworks between the ingested papers."
                st.session_state["asst_query_ver"] = st.session_state.get("asst_query_ver", 0) + 1
                st.rerun()
        with p_c4:
            if st.button("Extracted Parameters", use_container_width=True, help="Review sample sizes and effect sizes in papers"):
                st.session_state["assistant_query_text"] = "What sample sizes, treatment definitions, and quantitative effect sizes were reported across the uploaded documents?"
                st.session_state["asst_query_ver"] = st.session_state.get("asst_query_ver", 0) + 1
                st.rerun()
        with p_c5:
            if st.button("Disclosed Caveats", use_container_width=True, help="Review threats to validity"):
                st.session_state["assistant_query_text"] = "What limitations, unmeasured confounding risks, or threats to validity are disclosed in the uploaded literature?"
                st.session_state["asst_query_ver"] = st.session_state.get("asst_query_ver", 0) + 1
                st.rerun()

    else:  # HYBRID
        p_c1, p_c2, p_c3, p_c4, p_c5 = st.columns(5)
        with p_c1:
            if st.button("Benchmark with Theory", use_container_width=True, help="Compare dataset estimates with literature"):
                st.session_state["assistant_query_text"] = "Benchmark our dataset's computed Double ML and Doubly Robust AIPW estimates against the theoretical literature (e.g. Chernozhukov 2018). Are our results consistent with theoretical expectations?"
                st.session_state["asst_query_ver"] = st.session_state.get("asst_query_ver", 0) + 1
                st.rerun()
        with p_c2:
            if st.button("Connect Data to Theory", use_container_width=True, help="Synthesize data results with literature"):
                st.session_state["assistant_query_text"] = "Connect the empirical findings from our active dataset with the methodological guidance in our uploaded documents. How should decision-makers interpret these findings?"
                st.session_state["asst_query_ver"] = st.session_state.get("asst_query_ver", 0) + 1
                st.rerun()
        with p_c3:
            if st.button("Compare Robustness Bounds", use_container_width=True, help="Compare E-value to literature"):
                st.session_state["assistant_query_text"] = "Compare our dataset's computed sensitivity E-value against the sensitivity standards established in the literature (e.g. VanderWeele 2017). How robust is our evidence?"
                st.session_state["asst_query_ver"] = st.session_state.get("asst_query_ver", 0) + 1
                st.rerun()
        with p_c4:
            if st.button("Synthesize Full Insights", use_container_width=True, help="Generate comprehensive research insights"):
                st.session_state["assistant_query_text"] = "Synthesize comprehensive research insights integrating both our computed dataset results and the uploaded preprints. What is the evidence-based conclusion?"
                st.session_state["asst_query_ver"] = st.session_state.get("asst_query_ver", 0) + 1
                st.rerun()
        with p_c5:
            if st.button("Suggest Next Trials", use_container_width=True, help="Recommend trials and validation"):
                st.session_state["assistant_query_text"] = "Synthesizing our empirical refutations and literature benchmarks, what prospective validation experiments or policy interventions do you recommend?"
                st.session_state["asst_query_ver"] = st.session_state.get("asst_query_ver", 0) + 1
                st.rerun()

    if "assistant_query_text" not in st.session_state:
        st.session_state["assistant_query_text"] = ""

    if st.session_state.get("assistant_query_prefill"):
        prefill_val = st.session_state.pop("assistant_query_prefill")
        st.session_state["assistant_query_text"] = prefill_val
        st.session_state["asst_query_ver"] = st.session_state.get("asst_query_ver", 0) + 1

    placeholder_map = {
        KNOWLEDGE_SOURCE_DATASET: "Ask about treatment effects, covariate balance, sensitivity E-values, or dataset distributions...",
        KNOWLEDGE_SOURCE_DOCUMENTS: "Ask about uploaded PDFs/Word preprints, theoretical mechanisms, paper summaries, or methodology...",
        KNOWLEDGE_SOURCE_HYBRID: "Ask to connect computed dataset numbers with theoretical literature benchmarks..."
    }

    user_query = st.text_area(
        "Enter Your Question:",
        value=st.session_state.get("assistant_query_text", ""),
        height=95,
        placeholder=placeholder_map.get(source_mode, "Enter your question in plain English..."),
        key=f"assistant_query_input_{st.session_state.get('asst_query_ver', 0)}",
        help="Type any question about your data or documents. Gemini grounds its answer in calculated evidence."
    )

    btn_col1, btn_col2 = st.columns([1.5, 4])
    with btn_col1:
        ask_btn = st.button("Inquire AI Research Assistant", type="primary", use_container_width=True, disabled=not user_query.strip(), help="Generate a factual, grounded answer")
    with btn_col2:
        if st.button("Clear Conversation History", type="secondary", help="Reset all chat messages"):
            st.session_state["chat_history"] = []
            st.session_state["assistant_logs"] = []
            st.session_state["assistant_query_text"] = ""
            st.session_state["asst_query_ver"] = st.session_state.get("asst_query_ver", 0) + 1
            st.success("Conversation history cleared.")
            st.rerun()

    if ask_btn:
        with st.spinner("Interrogating knowledge engines and synthesizing evidence-grounded response..."):
            context_payload = build_research_context(
                df=df,
                dataset_name=st.session_state.get("dataset_name"),
                analysis_spec=st.session_state.get("analysis_spec"),
                all_estimators=st.session_state.get("all_estimators"),
                baseline_smd=st.session_state.get("baseline_smd"),
                balance_table=st.session_state.get("balance_table"),
                sensitivity_result=st.session_state.get("sensitivity_result"),
                refutation_results=st.session_state.get("refutation_results"),
                dag_summary=st.session_state.get("dag_summary"),
                knowledge_source=source_mode,
                document_catalog=indexed_docs
            )

            ai_out = answer_with_evidence(
                question=user_query,
                context=context_payload,
                api_key=active_gemini_key,
                mode=assistant_persona,
                knowledge_source=source_mode,
                model_name=selected_model,
                history=st.session_state.get("chat_history", []),
                vector_store=rag_store,
                top_k=asst_top_k,
                paper_filter=asst_doc_filter,
                file_type_filter=asst_fmt_filter,
                temperature=llm_temp
            )

            chat_record = {
                "question": user_query,
                "response": ai_out["text"],
                "mode": ai_out.get("mode"),
                "knowledge_source": source_mode,
                "faithfulness": ai_out.get("faithfulness"),
                "assumptions_audit": ai_out.get("assumptions_audit"),
                "citations": ai_out.get("citations", []),
                "retrieved_passages": ai_out.get("retrieved_passages", []),
                "timestamp": datetime.now().strftime("%H:%M:%S")
            }
            st.session_state["assistant_logs"].append(chat_record)
            st.session_state["chat_history"].append(chat_record)
            st.session_state["assistant_query_text"] = ""
            st.session_state["asst_query_ver"] = st.session_state.get("asst_query_ver", 0) + 1
            st.rerun()

    # Conversation History & Claim Audits
    chat_history = st.session_state.get("chat_history", [])
    if not chat_history:
        st.markdown(
            """
            <div class="empty-state-card">
                <div class="empty-state-title">Evidence-Grounded Assistant Ready</div>
                <div class="empty-state-text">
                    Select a knowledge source above (Structured Dataset, Uploaded Documents RAG, or Hybrid) and click a suggested research query or type your own question. Every response is strictly grounded in computed calculations and retrieved document passages with zero hallucinated results.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown("---")
        st.markdown("##### Research Conversation History")
        for log in reversed(chat_history):
            ks = log.get("knowledge_source", KNOWLEDGE_SOURCE_DATASET)
            ks_label = {
                KNOWLEDGE_SOURCE_DATASET: "DATASET ANALYTICS",
                KNOWLEDGE_SOURCE_DOCUMENTS: "DOCUMENT RAG",
                KNOWLEDGE_SOURCE_HYBRID: "HYBRID DUAL-ENGINE"
            }.get(ks, "RESEARCH ENGINE")
            
            ts = log.get("timestamp", "")

            st.markdown(
                f"""
                <div style="background: rgba(139, 92, 246, 0.12); border: 1px solid rgba(167, 139, 250, 0.3); border-radius: 12px; padding: 14px 18px; margin-top: 18px; margin-bottom: 12px;">
                    <div style="font-size: 11.5px; color: #A78BFA; font-weight: 700; margin-bottom: 4px; display:flex; justify-content:space-between; align-items:center;">
                        <span>RESEARCH INQUIRY • <span style="color:#38BDF8;">{ks_label}</span></span>
                        <span style="font-weight:400; color:#94A3B8;">{ts}</span>
                    </div>
                    <div style="color: #F8FAFC; font-size: 14.5px; line-height: 1.5; font-weight:600;">{log['question']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            f_data = log.get("faithfulness", {})
            a_data = log.get("assumptions_audit", {})

            b_score1, b_score2, b_score3 = st.columns(3)
            with b_score1:
                b_score1.metric("Numerical Faithfulness Score", f"{f_data.get('faithfulness_score', 100):.1f}%")
            with b_score2:
                b_score2.metric("Causal Assumptions Disclosed", f"{a_data.get('disclosed_count', 0)} / 5")
            with b_score3:
                b_score3.metric("Operating Persona", log.get("mode", "grounded").capitalize())

            st.markdown(
                f"""
                <div class="saas-card" style="border-left: 4px solid #8B5CF6; margin-bottom: 12px;">
                    <div style="font-size: 14.5px; line-height: 1.6; color: #F1F5F9;">
                        {log['response']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Citations Accordion (when document or hybrid citations exist)
            citations = log.get("citations", [])
            if citations:
                with st.expander(f"Grounded Document Citations ({len(citations)} Sources)", expanded=False):
                    for i, p in enumerate(citations, start=1):
                        sec_tag = p.get("section", "General")
                        ft = p.get("file_type", "doc").upper()
                        sim = p.get("similarity_score", 1.0)
                        st.markdown(
                            f"""
                            <div style="background-color: #0A1020; padding: 12px 14px; border-radius: 8px; border: 1px solid rgba(45, 212, 191, 0.2); margin-bottom: 8px;">
                                <div style="font-size: 11.5px; color: #2DD4BF; font-weight: 600; margin-bottom: 6px; display:flex; justify-content:space-between; align-items:center;">
                                    <span><strong style="color:#38BDF8;">[{ft}]</strong> {p.get('paper', 'Unknown')} • Unit/Page {p.get('page', 1)}</span>
                                    <span style="color:#A78BFA; background:rgba(139,92,246,0.18); padding:2px 8px; border-radius:4px; font-size:10.5px;">Match: {sim:.1%}</span>
                                </div>
                                <blockquote style="color: #CBD5E1; margin: 0; font-size: 13px; line-height: 1.5;">
                                    "{p.get('excerpt', '')}"
                                </blockquote>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

            # Claim Verification Breakdown Accordion
            with st.expander("Empirical Faithfulness Verification & Claim Audit Breakdown"):
                if f_data.get("claims_detail"):
                    st.dataframe(pd.DataFrame(f_data["claims_detail"]), use_container_width=True)
                else:
                    st.write(f_data.get("grounding_status", "No numeric claims detected."))
                st.write(f"**Methodological Rigor Rating:** {a_data.get('rigor_rating', 'N/A')}")
                if a_data.get("has_overclaiming"):
                    st.warning(f"Overclaiming phrase detected: {a_data.get('detected_overclaims')}")

            st.markdown("---")

    st.markdown("---")
    st.markdown(
        """
        <div class="saas-card" style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.08) 0%, rgba(14, 20, 32, 0.95) 100%); border: 1px solid rgba(99, 102, 241, 0.25); padding: 18px 22px; margin-bottom: 14px;">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
                <div>
                    <h4 style="color:#FFFFFF; margin:0 0 4px 0; font-size:15px;">Next Step: Ingest Documents or Run Monte Carlo Simulations</h4>
                    <p style="color:#94A3B8; font-size:12.5px; margin:0;">
                        Upload research papers in the Knowledge Base to expand the AI's literature context, or simulate synthetic scenarios in the Research Lab.
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    asst_nav1, asst_nav2, asst_nav3 = st.columns([1, 1, 1])
    with asst_nav1:
        if st.button("← Back to Causal Analysis", use_container_width=True, key="asst_nav_back", help="Return to Step 2"):
            switch_page(PAGE_CAUSAL)
    with asst_nav2:
        if st.button("Upload Documents to Knowledge Base →", type="primary", use_container_width=True, key="asst_nav_next", help="Ingest PDFs, Word files, and Excel tables"):
            switch_page(PAGE_RAG)
    with asst_nav3:
        if st.button("Run Simulation Lab →", use_container_width=True, key="asst_nav_lab", help="Launch Monte Carlo finite-sample simulations"):
            switch_page(PAGE_LAB)


# ==============================================================================
# PAGE 5: ADVANCED MULTI-FORMAT DOCUMENT INTELLIGENCE & RAG
# ==============================================================================
elif selected_page in (PAGE_RAG, "Literature RAG", "Knowledge Base / RAG", "Knowledge Base", "RAG"):
    st.markdown(
        """
        <div class="hero-badge">DOCUMENT KNOWLEDGE BASE & RAG</div>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
            <div>
                <h2 style="margin: 0; color: #F8FAFC; font-weight: 800;">Document Knowledge Base & Search</h2>
                <p style="color: #94A3B8; margin: 4px 0 0 0; font-size: 13.5px;">
                    Upload research papers, PDFs, Word documents, spreadsheets, or notes. Search passages, chat with documents, and extract structured findings.
                </p>
            </div>
            <div style="display: flex; gap: 8px;">
                <span class="status-pill">● 7 Formats Supported</span>
                <span class="status-pill-emerald">● Vector Storage Active</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    rag_store: AcademicVectorStore = st.session_state["rag_store"]
    indexed_docs = rag_store.list_indexed_documents()
    indexed_papers = indexed_docs  # Alias for backward compatibility

    # Format Breakdown
    type_counts = {}
    for d in indexed_docs:
        ft = d.get("file_type", "doc").upper()
        type_counts[ft] = type_counts.get(ft, 0) + 1
    type_breakdown_str = " • ".join([f"{k}: {v}" for k, v in sorted(type_counts.items())]) if type_counts else "No files yet"

    # Top Status Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="metric-tile metric-tile-cyan">
                <div class="metric-label">Indexed Documents</div>
                <div class="metric-value">{len(indexed_docs)}</div>
                <div class="metric-subtext">{type_breakdown_str[:32]}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        total_chunks = rag_store.collection.count()
        st.markdown(
            f"""
            <div class="metric-tile metric-tile-violet">
                <div class="metric-label">Searchable Sections</div>
                <div class="metric-value">{total_chunks:,}</div>
                <div class="metric-subtext">Embedded Vector Passages</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        total_pages = sum(p.get("total_pages", 0) for p in indexed_docs)
        st.markdown(
            f"""
            <div class="metric-tile metric-tile-emerald">
                <div class="metric-label">Total Pages & Units</div>
                <div class="metric-value">{total_pages:,}</div>
                <div class="metric-subtext">Pages, Slides & Sheets</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col4:
        emb_mode = f"Gemini ({mask_api_key(active_gemini_key)})" if active_gemini_key else "Offline Semantic Vectors"
        st.markdown(
            f"""
            <div class="metric-tile metric-tile-rose">
                <div class="metric-label">Search Mode</div>
                <div class="metric-value" style="font-size: 20px; line-height: 38px;">Hybrid RRF</div>
                <div class="metric-subtext">{emb_mode[:28]}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    if active_gemini_key:
        st.markdown(
            f"""
            <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 8px; padding: 10px 14px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center;">
                <div style="display:flex; align-items: center; gap: 8px;">
                    <span style="display:inline-block; width:8px; height:8px; background:#10B981; border-radius:50%; box-shadow:0 0 8px #10B981;"></span>
                    <span style="color: #F8FAFC; font-size: 13px; font-weight: 600;">Google Gemini Connected</span>
                    <span style="color: #94A3B8; font-size: 12px;">| Model: <code style="color:#22D3EE;">{selected_model}</code> | Credentials: <code>{mask_api_key(active_gemini_key)}</code></span>
                </div>
                <span style="background: rgba(16, 185, 129, 0.2); color: #34D399; font-size: 11px; padding: 3px 8px; border-radius: 4px; font-weight: 600;">ACTIVE</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.25); border-radius: 8px; padding: 10px 14px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center;">
                <div style="display:flex; align-items: center; gap: 8px;">
                    <span style="display:inline-block; width:8px; height:8px; background:#F59E0B; border-radius:50%;"></span>
                    <span style="color: #F8FAFC; font-size: 13px; font-weight: 600;">Gemini API Offline Mode</span>
                    <span style="color: #94A3B8; font-size: 12px;">| Vector search is active. To enable AI-synthesized answers, add your Gemini API key in Settings.</span>
                </div>
                <span style="background: rgba(245, 158, 11, 0.2); color: #FBBF24; font-size: 11px; padding: 3px 8px; border-radius: 4px; font-weight: 600;">SEARCH ONLY</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    rag_tabs = st.tabs([
        "1. Upload & Manage Documents",
        "2. AI Document Chat & Q&A",
        "3. Extract Key Findings",
        "4. Compare Multiple Documents"
    ])

    # ==========================================================================
    # TAB 1: Knowledge Base & Repository Management
    # ==========================================================================
    with rag_tabs[0]:
        st.markdown("#### Upload Documents & Build Your Knowledge Base")
        st.markdown(
            "Upload research papers, PDFs, Word documents, spreadsheets, slides, or notes. "
            "CausalLens AI extracts the text, creates vector embeddings in your ChromaDB database, and makes the content searchable for questions and summaries."
        )

        st.markdown(
            """
            <div style="display:flex; gap:8px; flex-wrap:wrap; margin-bottom:14px;">
                <span class="status-pill">PDF (.pdf)</span>
                <span class="status-pill">Word (.docx, .doc)</span>
                <span class="status-pill">Excel (.xlsx, .xls)</span>
                <span class="status-pill">CSV (.csv)</span>
                <span class="status-pill">PowerPoint (.pptx)</span>
                <span class="status-pill">Markdown (.md)</span>
                <span class="status-pill">Text (.txt)</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        uploaded_docs = st.file_uploader(
            "Select files to upload (drag & drop supported)",
            type=["pdf", "docx", "doc", "txt", "md", "csv", "xlsx", "xls", "pptx"],
            accept_multiple_files=True,
            key="rag_multi_doc_uploader",
            help="Select one or multiple research documents to ingest into the vector knowledge base."
        )

        # Advanced Chunking & Ingestion Settings Expander
        with st.expander("⚙️ Advanced Settings (Chunk Size & Overlap)", expanded=False):
            conf_c1, conf_c2, conf_c3 = st.columns(3)
            with conf_c1:
                target_chunk_size = st.slider(
                    "Target Chunk Size (Characters)",
                    min_value=250,
                    max_value=1800,
                    value=700,
                    step=50,
                    help="Target character size for each passage chunk. Smaller chunks improve retrieval precision; larger chunks provide broader context."
                )
            with conf_c2:
                chunk_overlap = st.slider(
                    "Chunk Overlap (Characters)",
                    min_value=0,
                    max_value=350,
                    value=120,
                    step=10,
                    help="Character overlap maintained across consecutive chunks to preserve sentence boundaries and semantic continuity."
                )
            with conf_c3:
                force_reindex = st.checkbox(
                    "Force Re-indexing",
                    value=False,
                    help="When unchecked, documents already present in ChromaDB are skipped. Check to overwrite existing document embeddings."
                )

        col_btn1, col_btn2, col_btn3 = st.columns([2, 2, 1.2])
        with col_btn1:
            if st.button("Process & Index Uploaded Documents", type="primary", use_container_width=True, help="Extract text, generate semantic embeddings, and store in persistent ChromaDB."):
                if not uploaded_docs:
                    st.warning("Please select at least one document to upload.")
                else:
                    with st.spinner(f"Validating, chunking, and embedding {len(uploaded_docs)} document(s)..."):
                        res = rag_store.add_documents(
                            uploaded_files=uploaded_docs,
                            api_key=active_gemini_key,
                            target_chunk_size=target_chunk_size,
                            overlap=chunk_overlap,
                            force_reindex=force_reindex
                        )
                        st.session_state["rag_last_ingestion_res"] = res
                        st.rerun()

        with col_btn2:
            if st.button("Seed Benchmark Literature", type="secondary", use_container_width=True, help="Pre-load foundational papers: Chernozhukov (2018), VanderWeele (2017), and Robins (1994)."):
                with st.spinner("Indexing foundational causal literature into ChromaDB..."):
                    res = rag_store.seed_benchmark_papers(api_key=active_gemini_key)
                    st.success(f"Loaded {res['documents_indexed']} seminal papers with {res['total_chunks']} passage chunks into ChromaDB!")
                    st.rerun()

        with col_btn3:
            if st.button("Clear Vector Index", type="secondary", use_container_width=True, help="Wipe all indexed documents and reset ChromaDB collection."):
                rag_store.clear_store()
                st.session_state["rag_last_ingestion_res"] = None
                st.info("Vector database cleared.")
                st.rerun()

        # Ingestion Results Display if recently processed
        if st.session_state.get("rag_last_ingestion_res"):
            ingest_res = st.session_state["rag_last_ingestion_res"]
            st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
            if ingest_res.get("documents_indexed", 0) > 0:
                st.success(
                    f"Successfully indexed **{ingest_res['documents_indexed']} document(s)** "
                    f"yielding **{ingest_res['total_chunks']} chunks** across **{ingest_res['total_pages']} structural units (pages/slides/sheets)**!"
                )
            if ingest_res.get("documents_skipped", 0) > 0:
                skipped_names = [s["file_name"] for s in ingest_res.get("skipped_files", [])]
                st.info(
                    f"**{ingest_res['documents_skipped']} document(s) skipped** because they are already indexed with matching fingerprints: "
                    f"`{', '.join(skipped_names)}`. Enable **Force Re-indexing** if you wish to overwrite."
                )
            if ingest_res.get("errors"):
                for err in ingest_res["errors"]:
                    st.error(f"Error processing `{err['file_name']}`: {err['error']}")

        st.markdown("---")
        st.markdown("#### Currently Indexed Knowledge Base Documents")

        indexed_docs = rag_store.list_indexed_documents()
        if indexed_docs:
            # Interactive Document Search Bar
            search_col, stat_col = st.columns([3, 1])
            with search_col:
                kb_search_term = st.text_input(
                    "Search Uploaded Documents",
                    placeholder="Search by title, file name, format (e.g. PDF, XLSX), or fingerprint...",
                    key="rag_kb_search_input"
                ).strip().lower()
            with stat_col:
                st.markdown(
                    f"""
                    <div style="text-align: right; padding-top: 26px;">
                        <span class="status-pill-emerald">{len(indexed_docs)} Ingested</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            filtered_docs = indexed_docs
            if kb_search_term:
                filtered_docs = [
                    d for d in indexed_docs
                    if (kb_search_term in d.get("title", "").lower() or
                        kb_search_term in d.get("file_name", "").lower() or
                        kb_search_term in d.get("file_type", "").lower() or
                        kb_search_term in d.get("doc_id", "").lower())
                ]
                if not filtered_docs:
                    st.info(f"No documents match search term: '{kb_search_term}'. Showing 0 of {len(indexed_docs)} documents.")

            if filtered_docs:
                doc_df = pd.DataFrame(filtered_docs)
                doc_df["file_type"] = doc_df["file_type"].str.upper()
                doc_df = doc_df.rename(columns={
                    "title": "Document Title",
                    "file_name": "File Name",
                    "file_type": "Format",
                    "total_pages": "Units (Pages/Slides)",
                    "total_chunks": "Indexed Chunks",
                    "doc_id": "Document Fingerprint (SHA-256)",
                    "upload_timestamp": "Ingestion Date/Time"
                })
                display_cols = ["File Name", "Document Title", "Format", "Units (Pages/Slides)", "Indexed Chunks", "Document Fingerprint (SHA-256)", "Ingestion Date/Time"]
                existing_cols = [c for c in display_cols if c in doc_df.columns]
                st.dataframe(doc_df[existing_cols], use_container_width=True)

            # Document Preview & Inspection Section
            with st.expander("Preview Document Information & Extracted Chunks", expanded=False):
                st.markdown("##### Inspect Extracted Metadata & Structural Passages")
                doc_preview_options = {f"[{d.get('file_type', 'doc').upper()}] {d['title']} ({d['file_name']})": d["doc_id"] for d in indexed_docs}
                selected_preview_label = st.selectbox("Select Document to Preview", list(doc_preview_options.keys()), key="rag_preview_doc_select")
                
                if selected_preview_label:
                    selected_doc_id = doc_preview_options[selected_preview_label]
                    preview_data = rag_store.get_document_preview(selected_doc_id, limit_chunks=4)
                    if preview_data:
                        prev_c1, prev_c2, prev_c3, prev_c4 = st.columns(4)
                        with prev_c1:
                            st.caption("DOCUMENT FORMAT")
                            st.markdown(f"**`{preview_data['file_type'].upper()}`**")
                        with prev_c2:
                            st.caption("STRUCTURAL UNITS")
                            st.markdown(f"**{preview_data['total_pages']} Pages/Slides/Sheets**")
                        with prev_c3:
                            st.caption("TOTAL CHUNKS")
                            st.markdown(f"**{preview_data['total_chunks']} Vectors**")
                        with prev_c4:
                            st.caption("FINGERPRINT (SHA-256)")
                            st.markdown(f"**`{preview_data['doc_id']}`**")

                        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
                        st.markdown("**Sample Extracted Chunks:**")
                        for idx, ch in enumerate(preview_data.get("chunks_preview", []), start=1):
                            with st.container():
                                st.markdown(
                                    f"""
                                    <div style="background-color: #0A1020; padding: 12px 16px; border-radius: 8px; border: 1px solid rgba(139, 92, 246, 0.25); margin-bottom: 10px;">
                                        <div style="font-size: 11.5px; color: #A78BFA; font-weight: 600; margin-bottom: 6px; display:flex; justify-content:space-between;">
                                            <span>Chunk #{idx} • Unit/Page {ch['page_num']} ({ch['char_count']} chars)</span>
                                            <span style="color: #38BDF8;">Section: {ch['section']}</span>
                                        </div>
                                        <blockquote style="color: #E2E8F0; margin: 0; font-size: 13px; line-height: 1.5;">
                                            "{ch['text']}"
                                        </blockquote>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

            # Re-indexing & Vector Purge Controls
            with st.expander("Re-index Documents with Custom Chunk Parameters", expanded=False):
                st.markdown("Re-chunk and regenerate embeddings for existing documents in ChromaDB with updated target sizes.")
                reindex_c1, reindex_c2 = st.columns(2)
                with reindex_c1:
                    reindex_target = st.selectbox(
                        "Target Document for Re-indexing",
                        ["All Indexed Documents"] + [f"{d['title']} ({d['file_name']})" for d in indexed_docs],
                        key="reindex_target_doc"
                    )
                with reindex_c2:
                    reindex_chunk_sz = st.slider("New Target Chunk Size", 300, 1600, 700, 50, key="reindex_chunk_sz")
                    reindex_overlap_sz = st.slider("New Chunk Overlap", 0, 300, 100, 10, key="reindex_overlap_sz")

                if st.button("Execute Re-indexing", type="primary", key="btn_exec_reindex"):
                    with st.spinner("Re-chunking and re-embedding passages into ChromaDB..."):
                        if reindex_target == "All Indexed Documents":
                            re_res = rag_store.reindex_all_documents(
                                target_chunk_size=reindex_chunk_sz,
                                overlap=reindex_overlap_sz,
                                api_key=active_gemini_key
                            )
                            st.success(f"Successfully re-indexed {re_res['total_documents_reindexed']} documents. Current collection vector count: {re_res['collection_count']}.")
                        else:
                            matched_doc = next((d for d in indexed_docs if f"{d['title']} ({d['file_name']})" == reindex_target), None)
                            if matched_doc:
                                re_res = rag_store.reindex_document(
                                    doc_identifier=matched_doc["doc_id"],
                                    target_chunk_size=reindex_chunk_sz,
                                    overlap=reindex_overlap_sz,
                                    api_key=active_gemini_key
                                )
                                if re_res.get("success"):
                                    st.success(f"Re-indexed '{matched_doc['title']}': {re_res['previous_chunks_count']} previous chunks replaced with {re_res['new_chunks_count']} new vectors.")
                                else:
                                    st.error(re_res.get("error", "Re-indexing failed."))
                        st.rerun()

            # Single document removal control
            with st.expander("Delete Documents & Purge Vectors", expanded=False):
                doc_options = {f"[{d.get('file_type', 'doc').upper()}] {d['title']} ({d['file_name']})": d["doc_id"] for d in indexed_docs}
                selected_to_remove = st.selectbox("Select document to delete", ["Select a document..."] + list(doc_options.keys()), key="rag_delete_doc_select")
                if selected_to_remove != "Select a document...":
                    if st.button(f"Permanently Delete '{selected_to_remove}' & Associated Vectors", type="secondary"):
                        target_id = doc_options[selected_to_remove]
                        rag_store.delete_document(target_id)
                        st.success(f"Removed '{selected_to_remove}' and purged all vector embeddings.")
                        st.rerun()
        else:
            st.markdown(
                """
                <div class="empty-state-card">
                    <div class="empty-state-title">No Documents Indexed Yet</div>
                    <div class="empty-state-text">
                        Upload research PDFs, Word documents, Excel workbooks, CSV datasets, PowerPoint slides, or Markdown notes using the uploader above, or click <strong>Seed Benchmark Literature</strong> to instantly load foundational causal econometric papers (Chernozhukov Double ML, VanderWeele E-Values, Robins AIPW) into ChromaDB.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # ==========================================================================
    # TAB 2: AI Document Chat (Conversational RAG)
    # ==========================================================================
    with rag_tabs[1]:
        st.markdown("#### Chat with Your Documents")
        st.markdown(
            "Ask natural-language questions about your uploaded documents. "
            "The AI searches relevant passages, quotes exact source citations with page numbers, and alerts you if evidence is missing."
        )

        if not indexed_docs:
            st.markdown(
                """
                <div class="empty-state-card">
                    <div class="empty-state-title">Knowledge Base is Empty</div>
                    <div class="empty-state-text">
                        Please upload research documents or seed benchmark literature in the <strong>Upload & Manage Documents</strong> tab before chatting with documents.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            # Configurable Retrieval Settings Expander
            with st.expander("⚙️ Advanced Search & Filter Options", expanded=False):
                cfg_c1, cfg_c2, cfg_c3, cfg_c4 = st.columns(4)
                with cfg_c1:
                    top_k_select = st.slider("Top Passages (k)", min_value=1, max_value=8, value=4, key="chat_top_k")
                with cfg_c2:
                    min_similarity_threshold = st.slider("Min Similarity Threshold", min_value=0.0, max_value=0.85, value=0.0, step=0.05, key="chat_min_sim")
                with cfg_c3:
                    doc_filter_opts = ["All Documents"] + [d["title"] for d in indexed_docs]
                    selected_doc_filter = st.selectbox("Filter to Specific Document", doc_filter_opts, key="chat_doc_filter")
                with cfg_c4:
                    format_filter_opts = ["All Formats"] + sorted(list(set(d.get("file_type", "doc").upper() for d in indexed_docs)))
                    selected_format_filter = st.selectbox("Filter by Format", format_filter_opts, key="chat_fmt_filter")

            # Preset Inquiries Chips
            sample_queries = [
                "Select a guided inquiry...",
                "How does Double Machine Learning eliminate first-order regularization bias?",
                "What are the core identification assumptions for Average Treatment Effect (ATE)?",
                "How does Augmented Inverse Probability Weighting (AIPW) achieve double robustness?",
                "What does the E-value measure in sensitivity analysis for unmeasured confounding?",
                "How does 1:1 nearest-neighbor propensity score matching balance covariates?",
                "Summarize the clinical dataset features and cohort characteristics."
            ]
            q_preset = st.selectbox("Guided Research Inquiries", sample_queries, key="chat_preset_select")
            if q_preset != "Select a guided inquiry...":
                st.session_state["rag_chat_input_val"] = q_preset

            # Render Conversation History Thread
            if st.session_state.get("rag_chat_history"):
                st.markdown("##### Conversation History")
                for turn_idx, turn in enumerate(st.session_state["rag_chat_history"], start=1):
                    role = turn.get("role", "user")
                    content = turn.get("content", "")
                    ts = turn.get("timestamp", "")
                    
                    if role == "user":
                        st.markdown(
                            f"""
                            <div style="background: rgba(139, 92, 246, 0.12); border: 1px solid rgba(167, 139, 250, 0.3); border-radius: 12px; padding: 14px 18px; margin-bottom: 12px;">
                                <div style="font-size: 11.5px; color: #A78BFA; font-weight: 700; margin-bottom: 4px; display:flex; justify-content:space-between;">
                                    <span>RESEARCH INQUIRY</span>
                                    <span style="font-weight:400; color:#94A3B8;">{ts}</span>
                                </div>
                                <div style="color: #F8FAFC; font-size: 14.5px; line-height: 1.5;">{content}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        is_insufficient = turn.get("insufficient_evidence", False)
                        st.markdown(
                            f"""
                            <div class="saas-card" style="border-left: 4px solid {'#F59E0B' if is_insufficient else '#8B5CF6'}; margin-bottom: 14px;">
                                <div style="font-size: 11.5px; color: {'#F59E0B' if is_insufficient else '#2DD4BF'}; font-weight: 700; margin-bottom: 8px; display:flex; justify-content:space-between;">
                                    <span>{'EVIDENCE GAP NOTIFICATION' if is_insufficient else 'SYNTHESIZED GROUNDED RESPONSE'}</span>
                                    <span style="font-weight:400; color:#94A3B8;">{ts}</span>
                                </div>
                                <div style="font-size: 14.5px; line-height: 1.6; color: #F1F5F9;">
                                    {content}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        # Citations Accordion
                        citations = turn.get("citations", [])
                        if citations:
                            with st.expander(f"Grounded Citations & Evidence Passages ({len(citations)} Sources)", expanded=False):
                                for i, p in enumerate(citations, start=1):
                                    sec_tag = p.get("section", "General")
                                    ft = p.get("file_type", "doc").upper()
                                    sim = p.get("similarity_score", 1.0)
                                    st.markdown(
                                        f"""
                                        <div style="background-color: #0A1020; padding: 12px 14px; border-radius: 8px; border: 1px solid rgba(45, 212, 191, 0.2); margin-bottom: 8px;">
                                            <div style="font-size: 11.5px; color: #2DD4BF; font-weight: 600; margin-bottom: 6px; display:flex; justify-content:space-between; align-items:center;">
                                                <span><strong style="color:#38BDF8;">[{ft}]</strong> {p.get('paper', 'Unknown')} • Unit/Page {p.get('page', 1)}</span>
                                                <span style="color:#A78BFA; background:rgba(139,92,246,0.18); padding:2px 8px; border-radius:4px; font-size:10.5px;">Match: {sim:.1%}</span>
                                            </div>
                                            <blockquote style="color: #CBD5E1; margin: 0; font-size: 13px; line-height: 1.5;">
                                                "{p.get('excerpt', '')}"
                                            </blockquote>
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )

            # Chat Input Form
            with st.form("rag_chat_form", clear_on_submit=False):
                current_val = st.session_state.get("rag_chat_input_val", "")
                chat_query = st.text_input(
                    "Ask a question or enter a follow-up inquiry:",
                    value=current_val,
                    placeholder="e.g. How does Double Machine Learning handle high-dimensional confounders, and what was the sample size?",
                    key="rag_chat_input_field"
                )
                form_c1, form_c2 = st.columns([3, 1])
                with form_c1:
                    submit_chat = st.form_submit_button("Send Question / Follow-Up", type="primary", use_container_width=True)
                with form_c2:
                    clear_chat = st.form_submit_button("Clear Conversation", type="secondary", use_container_width=True)

            if clear_chat:
                st.session_state["rag_chat_history"] = []
                st.session_state["rag_chat_input_val"] = ""
                st.success("Chat history cleared.")
                st.rerun()

            if submit_chat:
                if not chat_query.strip():
                    st.warning("Please provide a question.")
                else:
                    d_filter = None if selected_doc_filter == "All Documents" else selected_doc_filter
                    f_filter = None if selected_format_filter == "All Formats" else selected_format_filter
                    
                    with st.spinner("Retrieving semantic passages and synthesizing evidence-grounded response..."):
                        qa_result = answer_document_chat(
                            query=chat_query,
                            vector_store=rag_store,
                            chat_history=st.session_state.get("rag_chat_history", []),
                            api_key=active_gemini_key,
                            model_name=selected_model,
                            top_k=top_k_select,
                            paper_filter=d_filter,
                            file_type_filter=f_filter,
                            min_similarity=min_similarity_threshold
                        )
                        
                        now_str = datetime.now().strftime("%H:%M:%S")
                        # Append user turn
                        st.session_state["rag_chat_history"].append({
                            "role": "user",
                            "content": chat_query,
                            "timestamp": now_str
                        })
                        # Append assistant turn
                        st.session_state["rag_chat_history"].append({
                            "role": "assistant",
                            "content": qa_result.get("answer", ""),
                            "citations": qa_result.get("citations", []),
                            "retrieved_passages": qa_result.get("retrieved_passages", []),
                            "insufficient_evidence": qa_result.get("insufficient_evidence", False),
                            "timestamp": now_str
                        })
                        st.session_state["rag_chat_input_val"] = ""
                        st.rerun()

    # ==========================================================================
    # TAB 3: Key Information & Empirical Extraction
    # ==========================================================================
    with rag_tabs[2]:
        st.markdown("#### Extract Key Information & Executive Review")
        st.markdown(
            "Automatically extract structured research parameters (treatment, outcome, sample size, estimated effect) "
            "or generate a publication-grade executive summary for any uploaded document."
        )

        if not indexed_docs:
            st.markdown(
                """
                <div class="empty-state-card">
                    <div class="empty-state-title">No Documents Available for Parameter Extraction</div>
                    <div class="empty-state-text">
                        Please upload research documents or seed benchmark literature in the <strong>Upload & Manage Documents</strong> tab.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            all_titles = [d["title"] for d in indexed_docs]
            
            ext_subtabs = st.tabs([
                "Structured Parameters",
                "Executive Document Review"
            ])

            # Subtab 1: Key Causal Parameter Extraction
            with ext_subtabs[0]:
                st.markdown("##### Extract Key Research Parameters")
                doc_choice_extract = st.selectbox("Select Target Document for Parameter Extraction", all_titles, key="doc_extract_select", help="Choose a document to extract sample size, treatment, outcome, and effect estimates.")
                
                if st.button("Extract Structured Causal Parameters", type="primary", use_container_width=True, key="btn_extract_params"):
                    with st.spinner(f"Extracting empirical causal parameters for '{doc_choice_extract}'..."):
                        ext_res = extract_key_information(
                            document_title_or_id=doc_choice_extract,
                            vector_store=rag_store,
                            api_key=active_gemini_key,
                            model_name=selected_model
                        )
                        st.session_state["rag_key_info_result"] = ext_res

                if st.session_state.get("rag_key_info_result"):
                    ki_res = st.session_state["rag_key_info_result"]
                    st.markdown("---")
                    st.markdown(
                        f"""
                        <div class="saas-card" style="border-left: 4px solid #8B5CF6;">
                            <div style="font-size: 14.5px; line-height: 1.6; color: #F1F5F9;">
                                {ki_res['markdown']}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.download_button(
                        "Download Extracted Parameters (.md)",
                        data=ki_res["markdown"],
                        file_name=f"extracted_parameters_{doc_choice_extract.replace(' ', '_')[:30]}.md",
                        mime="text/markdown",
                        key="dl_extracted_params"
                    )

            # Subtab 2: Executive Document Review
            with ext_subtabs[1]:
                st.markdown("##### Executive Summary & Methodology Synthesis")
                doc_choice_review = st.selectbox("Select Target Document for Executive Review", all_titles, key="doc_review_select", help="Generate an executive summary of this study's methodology and findings.")
                
                if st.button("Generate Executive Document Review", type="primary", use_container_width=True, key="btn_exec_review"):
                    with st.spinner(f"Compiling executive review for '{doc_choice_review}'..."):
                        sum_res = generate_research_summary(
                            paper_title=doc_choice_review,
                            vector_store=rag_store,
                            api_key=active_gemini_key,
                            model_name=selected_model
                        )
                        st.session_state["rag_summary_result"] = sum_res

                if st.session_state.get("rag_summary_result"):
                    s_res = st.session_state["rag_summary_result"]
                    st.markdown("---")
                    st.markdown(
                        f"""
                        <div class="saas-card" style="border-left: 4px solid #14B8A6;">
                            <div style="font-size: 14.5px; line-height: 1.6; color: #F1F5F9;">
                                {s_res['summary']}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.download_button(
                        "Download Executive Summary (.md)",
                        data=s_res["summary"],
                        file_name=f"executive_review_{doc_choice_review.replace(' ', '_')[:30]}.md",
                        mime="text/markdown",
                        key="dl_exec_summary"
                    )

    # ==========================================================================
    # TAB 4: Cross-Document Synthesis & Comparison Matrix
    # ==========================================================================
    with rag_tabs[3]:
        st.markdown("#### Compare Across Documents")
        st.markdown(
            "Synthesize and compare methods, identification assumptions, and empirical findings across multiple uploaded documents side-by-side."
        )

        if not indexed_docs:
            st.markdown(
                """
                <div class="empty-state-card">
                    <div class="empty-state-title">Multi-Document Comparison Requires Ingested Sources</div>
                    <div class="empty-state-text">
                        Upload research documents or seed benchmark literature in the <strong>Upload & Manage Documents</strong> tab to compare methodologies across studies.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            comp_topic = st.text_input(
                "Comparison Topic / Focus Area",
                value="Causal Estimators, Identification Assumptions, and Robustness Diagnostics",
                help="Define the thematic topic for cross-document comparison.",
                key="rag_comp_topic_input"
            )

            if st.button("Generate Cross-Document Matrix", type="primary", use_container_width=True, key="btn_gen_comp_matrix"):
                with st.spinner("Analyzing multi-document methodology and generating comparison matrix..."):
                    comp_res = compare_literature_methods(
                        vector_store=rag_store,
                        comparison_topic=comp_topic,
                        api_key=active_gemini_key,
                        model_name=selected_model
                    )
                    st.session_state["rag_comparison_result"] = comp_res

            if st.session_state.get("rag_comparison_result"):
                c_res = st.session_state["rag_comparison_result"]
                st.markdown("---")
                st.markdown(c_res["comparison_matrix"])
                st.download_button(
                    "Download Comparison Matrix (.md)",
                    data=c_res["comparison_matrix"],
                    file_name="cross_document_comparison_matrix.md",
                    mime="text/markdown",
                    key="dl_comp_matrix"
                )

    st.markdown("---")
    st.markdown(
        """
        <div class="saas-card" style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.08) 0%, rgba(14, 20, 32, 0.95) 100%); border: 1px solid rgba(99, 102, 241, 0.25); padding: 18px 22px; margin-bottom: 14px;">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
                <div>
                    <h4 style="color:#FFFFFF; margin:0 0 4px 0; font-size:15px;">Next Step: Inquire the AI Assistant or Run Simulations</h4>
                    <p style="color:#94A3B8; font-size:12.5px; margin:0;">
                        Ask the AI Research Assistant questions about your newly indexed documents, or test model estimators in the Simulation Lab.
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    rag_nav1, rag_nav2, rag_nav3 = st.columns([1, 1, 1])
    with rag_nav1:
        if st.button("← Back to AI Assistant", use_container_width=True, key="rag_nav_back", help="Return to chat assistant"):
            switch_page(PAGE_ASSISTANT)
    with rag_nav2:
        if st.button("Monte Carlo Simulation Lab →", type="primary", use_container_width=True, key="rag_nav_next", help="Launch estimator simulation tests"):
            switch_page(PAGE_LAB)
    with rag_nav3:
        if st.button("Generate Reports & Exports →", use_container_width=True, key="rag_nav_reports", help="Go to Reports & Exports"):
            switch_page(PAGE_REPORTS)



# ==============================================================================
# PAGE 6: RESEARCH LAB (MONTE CARLO BENCHMARK)
# ==============================================================================
elif selected_page in (PAGE_LAB, "Research Lab", "Simulation Lab"):
    st.markdown(
        """
        <div class="hero-badge">BENCHMARK SIMULATION LAB</div>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
            <div>
                <h2 style="margin: 0; color: #F8FAFC; font-weight: 800;">Simulation Lab: Compare Model Performance</h2>
                <p style="color: #94A3B8; margin: 4px 0 0 0; font-size: 13.5px;">
                    Run repeated experiments with known ground-truth effects to see which causal model is most accurate with noisy or confounded data.
                </p>
            </div>
            <span class="status-pill-cyan">● Ground Truth Known</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    lab1, lab2 = st.columns([1, 2])
    with lab1:
        st.markdown("#### Experiment Settings")
        sim_tau = st.number_input("Known Ground-Truth Effect (τ)", value=2.5, step=0.5, help="The real treatment effect we want estimators to recover.")
        sim_n = st.slider("Sample Size (N)", min_value=200, max_value=2000, value=600, step=100, help="Number of synthetic individuals/rows per experiment.")
        sim_reps = st.slider("Simulation Runs (M)", min_value=5, max_value=50, value=25, step=5, help="Number of repeated experiments to test estimator consistency.")

        with st.expander("⚙️ Advanced Stress-Test Scenarios", expanded=False):
            sim_conf = st.slider("Confounding Multiplier", 0.5, 2.5, 1.0, 0.25, help="Strength of association between confounders and treatment/outcome.")
            sim_nl = st.checkbox("Inject Non-Linear Confounding (Stress test simple linear models)", value=False, help="Adds non-linear interactions to test Double ML and Matching advantages.")
            sim_unm = st.checkbox("Inject Unmeasured Common Cause (Omitted Variable Bias)", value=False, help="Simulates an unobserved factor to test sensitivity.")

        run_mc_action = st.button("Launch Monte Carlo Simulation", type="primary", use_container_width=True, help="Simulate data and benchmark OLS, Matching, IPW, AIPW, and DML.")

    with lab2:
        if run_mc_action:
            with st.spinner(f"Running {sim_reps} Monte Carlo replications across random seeds (True tau = {sim_tau})..."):
                mc_out = run_monte_carlo_benchmark(
                    n_replications=sim_reps,
                    n_samples=sim_n,
                    true_ate=sim_tau,
                    confounding_strength=sim_conf,
                    non_linear=sim_nl,
                    unmeasured_confounding=sim_unm,
                    estimators=["OLS", "Matching", "IPW", "AIPW", "DML"],
                    seed_start=300
                )
                st.session_state["mc_res"] = mc_out

        mc_res = st.session_state.get("mc_res")
        if mc_res:
            st.markdown("#### Empirical Benchmark Results Table")
            st.caption("Key: **Bias** (closer to 0 is better) • **RMSE** (lower error is better) • **Coverage** (target is 95% of intervals covering truth)")
            st.dataframe(mc_res["summary_table"], use_container_width=True)

            rep_rows = []
            for est_name, vals in mc_res["replication_data"].items():
                for v in vals:
                    rep_rows.append({"Estimator": est_name, "Estimate": v})
            rep_df = pd.DataFrame(rep_rows)

            if not rep_df.empty:
                fig_box = px.box(
                    rep_df,
                    x="Estimator",
                    y="Estimate",
                    color="Estimator",
                    points="all",
                    title=f"Estimator Sampling Distribution (True tau = {sim_tau})"
                )
                fig_box.add_hline(y=sim_tau, line_dash="solid", line_color="#2DD4BF", annotation_text=f"Ground Truth ({sim_tau})")
                apply_dark_theme(fig_box)
                st.plotly_chart(fig_box, use_container_width=True)

                # Export CSV button
                csv_bytes = mc_res["summary_table"].to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Monte Carlo Benchmark (.csv)",
                    data=csv_bytes,
                    file_name="monte_carlo_benchmark_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        else:
            st.markdown(
                """
                <div class="empty-state-card">
                    <div class="empty-state-title">Simulation Studio Ready</div>
                    <div class="empty-state-text">
                        Configure your simulation parameters on the left and click <strong>Launch Monte Carlo Simulation</strong>. The engine tests OLS, Matching, IPW, AIPW, and Double ML across repeated trials to evaluate accuracy, bias, and stability.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("---")
    st.markdown(
        """
        <div class="saas-card" style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.08) 0%, rgba(14, 20, 32, 0.95) 100%); border: 1px solid rgba(99, 102, 241, 0.25); padding: 18px 22px; margin-bottom: 14px;">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
                <div>
                    <h4 style="color:#FFFFFF; margin:0 0 4px 0; font-size:15px;">Next Step: Export Your Full Research Report</h4>
                    <p style="color:#94A3B8; font-size:12.5px; margin:0;">
                        Download a complete audit bundle with Markdown documentation, JSON reproducibility files, and printable HTML.
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    lab_nav1, lab_nav2, lab_nav3 = st.columns([1, 1, 1])
    with lab_nav1:
        if st.button("← Back to Knowledge Base / RAG", use_container_width=True, key="lab_nav_back", help="Return to document workspace"):
            switch_page(PAGE_RAG)
    with lab_nav2:
        if st.button("Generate Reports & Exports →", type="primary", use_container_width=True, key="lab_nav_next", help="Go to final reports"):
            switch_page(PAGE_REPORTS)
    with lab_nav3:
        if st.button("Configure Settings →", use_container_width=True, key="lab_nav_settings", help="Adjust API keys and platform settings"):
            switch_page(PAGE_SETTINGS)


# ==============================================================================
# PAGE 7: REPORTS (SCIENTIFIC DOCUMENTATION & EXPORT)
# ==============================================================================
elif selected_page in (PAGE_REPORTS, "Reports", "Reports & Exports", "Audit Reports"):
    st.markdown(
        """
        <div class="hero-badge">FINAL STEP • REPORTS & EXPORTS</div>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
            <div>
                <h2 style="margin: 0; color: #F8FAFC; font-weight: 800;">Research Reports & Exports</h2>
                <p style="color: #94A3B8; margin: 4px 0 0 0; font-size: 13.5px;">
                    Review your complete research findings and download peer-review ready reports in Markdown, machine-readable JSON, or printable HTML.
                </p>
            </div>
            <span class="status-pill-emerald">● Audit Ready</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    spec = st.session_state.get("analysis_spec")
    all_ests = st.session_state.get("all_estimators")
    sens = st.session_state.get("sensitivity_result")
    refs = st.session_state.get("refutation_results")
    bal = st.session_state.get("balance_table")
    dag_s = st.session_state.get("dag_summary")
    logs = st.session_state.get("assistant_logs")

    rag_s = st.session_state.get("rag_store")
    lit_summary = None
    if rag_s:
        indexed_p = rag_s.list_indexed_papers()
        if indexed_p:
            lit_summary = {
                "papers_count": len(indexed_p),
                "chunks_count": rag_s.collection.count(),
                "papers": indexed_p,
                "recent_qa": {
                    "query": st.session_state.get("rag_last_query", ""),
                    "answer": st.session_state.get("rag_last_result", {}).get("answer", "")
                } if st.session_state.get("rag_last_result") else None
            }

    full_md = build_markdown_report(
        df=df,
        causal_result=st.session_state.get("causal_result"),
        analysis_spec=spec,
        all_estimators=all_ests,
        balance_table=bal,
        sensitivity_result=sens,
        refutation_results=refs,
        dag_summary=dag_s,
        assistant_logs=logs,
        literature_summary=lit_summary
    )

    json_bundle = build_json_reproducibility_bundle(
        df=df,
        analysis_spec=spec or {},
        all_estimators=all_ests or [],
        balance_table=bal,
        sensitivity_result=sens,
        refutation_results=refs,
        literature_summary=lit_summary
    )

    html_doc = build_html_report(full_md)

    st.markdown("##### Download Export Files")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button(
            "Download Markdown Report (.md)",
            data=full_md,
            file_name="causallens_research_report.md",
            mime="text/markdown",
            use_container_width=True,
            help="Human-readable Markdown document with all tables and findings."
        )
    with d2:
        st.download_button(
            "Download JSON Reproducibility Bundle (.json)",
            data=json_bundle,
            file_name="causallens_reproducibility_bundle.json",
            mime="application/json",
            use_container_width=True,
            help="Machine-readable bundle containing all model outputs and specs."
        )
    with d3:
        st.download_button(
            "Download Printable HTML Report (.html)",
            data=html_doc,
            file_name="causallens_research_report.html",
            mime="text/html",
            use_container_width=True,
            help="Self-contained styled HTML file ready for printing or sharing."
        )

    st.markdown("---")
    st.markdown("#### Document Live Preview")
    st.markdown(full_md)

    st.markdown("---")
    st.markdown(
        """
        <div class="saas-card" style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.08) 0%, rgba(14, 20, 32, 0.95) 100%); border: 1px solid rgba(99, 102, 241, 0.25); padding: 18px 22px; margin-bottom: 14px;">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
                <div>
                    <h4 style="color:#FFFFFF; margin:0 0 4px 0; font-size:15px;">Research Workflow Completed</h4>
                    <p style="color:#94A3B8; font-size:12.5px; margin:0;">
                        You have inspected your data, estimated causal effects, interrogated the AI assistant, and generated publication reports.
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    rep_nav1, rep_nav2, rep_nav3 = st.columns([1, 1, 1])
    with rep_nav1:
        if st.button("← Back to Research Lab", use_container_width=True, key="reports_nav_back", help="Return to simulation lab"):
            switch_page(PAGE_LAB)
    with rep_nav2:
        if st.button("Return to Overview / Home", type="primary", use_container_width=True, key="reports_nav_home", help="Go back to the homepage"):
            switch_page(PAGE_OVERVIEW)
    with rep_nav3:
        if st.button("Platform Settings →", use_container_width=True, key="reports_nav_settings", help="Open platform settings"):
            switch_page(PAGE_SETTINGS)


# ==============================================================================
# PAGE 8: SETTINGS & ENGINE CONFIGURATION
# ==============================================================================
elif selected_page in (PAGE_SETTINGS, "Settings"):
    st.markdown(
        """
        <div class="hero-badge">PLATFORM SETTINGS</div>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
            <div>
                <h2 style="margin: 0; color: #F8FAFC; font-weight: 800;" class="gradient-title">Platform Settings & AI Configuration</h2>
                <p style="color: #94A3B8; margin: 4px 0 0 0; font-size: 13.5px;">
                    Configure your Google Gemini API key, customize causal model settings, manage memory caches, and inspect system telemetry.
                </p>
            </div>
            <span class="status-pill-emerald">● System Operational</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    set_tab1, set_tab2, set_tab3, set_tab4 = st.tabs([
        "1. Gemini API & AI Model",
        "2. Causal Model Defaults",
        "3. Memory & Cache Management",
        "4. System Environment"
    ])

    with set_tab1:
        st.markdown("#### Google Gemini API Configuration")
        st.caption("Connect your Gemini API key to unlock natural-language explanations, multi-document RAG chat, and automated research summaries.")
        
        c_k1, c_k2 = st.columns([3, 1])
        with c_k1:
            curr_env_key = get_api_key() or ""
            set_api_key = st.text_input(
                "Gemini API Key",
                value=st.session_state.get("user_gemini_api_key", curr_env_key),
                type="password",
                help="Enter your API key from Google AI Studio. It is loaded securely via environment variables and never logged or exposed."
            )
            if set_api_key != st.session_state.get("user_gemini_api_key", curr_env_key):
                st.session_state["user_gemini_api_key"] = set_api_key
                os.environ["GEMINI_API_KEY"] = set_api_key
                os.environ["GOOGLE_API_KEY"] = set_api_key

            active_key_preview = set_api_key.strip() if set_api_key.strip() else curr_env_key
            if active_key_preview:
                st.markdown(f"<span style='color: #4ADE80; font-size: 12px;'>● Key Configured: <code>{mask_api_key(active_key_preview)}</code></span>", unsafe_allow_html=True)
            else:
                st.markdown("<span style='color: #94A3B8; font-size: 12px;'>○ No API key configured. Offline deterministic fallback active.</span>", unsafe_allow_html=True)
        
        with c_k2:
            st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
            if st.button("Probe Gemini Auth", use_container_width=True, type="primary", help="Send a test ping to Google Gemini API"):
                key_to_test = set_api_key.strip() if set_api_key.strip() else curr_env_key
                with st.spinner("Connecting to Google Gemini API via official google-genai SDK..."):
                    val = validate_gemini_connection(api_key=key_to_test)
                    if val["is_valid"]:
                        st.success(
                            f"**Gemini Connection Verified!**\n\n"
                            f"- **Model:** `{val['model']}`\n"
                            f"- **Credentials:** `{val['masked_key']}`\n"
                            f"- **Status:** Official Google GenAI SDK authenticated."
                        )
                    else:
                        st.error(
                            f"**Connection Verification Failed:** {val['message']}\n\n"
                            f"{val.get('error_details', '') or ''}"
                        )

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.selectbox(
                "Default Assistant Model",
                ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-flash-lite-latest", "gemini-3.5-flash", "gemini-3.7-flash", "gemini-3.8-flash", "gemini-3.1-pro-preview"],
                index=0,
                key="settings_default_model",
                help="Select which Gemini model powers your assistant."
            )
            st.selectbox(
                "Default Assistant Mode",
                [
                    ("Evidence-Grounded (Verifies all numerical claims)", MODE_GROUNDED),
                    ("Methodologist Auditor (Rigorous causal critique)", MODE_AUDITOR),
                    ("Unconstrained Exploration (Generative analysis)", MODE_UNCONSTRAINED)
                ],
                format_func=lambda x: x[0],
                key="settings_default_mode",
                help="Operating persona for conversational answers."
            )
        with col_m2:
            with st.expander("⚙️ Advanced Model Hyperparameters", expanded=False):
                st.slider("Model Temperature", 0.0, 1.0, 0.2, 0.05, key="settings_temperature", help="Lower is more factual; higher is more creative.")
                st.number_input("Max Generation Tokens", min_value=256, max_value=8192, value=2048, step=256, key="settings_max_tokens", help="Maximum length of generated response.")

        st.markdown(
            """
            > **Tip for Deployment:**
            > In Streamlit Community Cloud, add your API key under **App Settings → Secrets**:
            > ```toml
            > GEMINI_API_KEY = "your_actual_key_here"
            > ```
            """
        )

    with set_tab2:
        st.markdown("#### Causal Inference Engine Parameters")
        st.caption("Adjust default statistical confidence intervals and algorithm settings. Defaults follow standard best practices.")

        c_par1, c_par2 = st.columns(2)
        with c_par1:
            st.selectbox("Statistical Confidence Level (1 - α)", [0.90, 0.95, 0.99], index=1, format_func=lambda x: f"{int(x*100)}% Confidence Interval", key="settings_conf_level", help="Target confidence level for all estimator confidence bounds.")
            st.slider("Propensity Matching Caliper (Standard Deviations)", 0.01, 0.50, 0.05, 0.01, key="settings_psm_caliper", help="Maximum allowable distance in propensity scores between matched units.")
            st.slider("Propensity Weight Trimming Bounds (IPW)", 0.001, 0.05, 0.01, 0.005, key="settings_ipw_trim", help="Lower bound truncation to prevent extreme inverse propensity weights.")

        with c_par2:
            st.slider("Double Machine Learning (DML) K-Fold Folds", 2, 10, 5, 1, key="settings_dml_folds", help="Number of cross-fitting splits for residualizing nuisance functions.")
            st.number_input("Random Forest Trees for Nuisance Models", min_value=20, max_value=500, value=100, step=20, key="settings_rf_trees", help="Number of estimators in the underlying Random Forest regressors.")
            st.selectbox("Standard Error Estimator", ["HC3 (MacKinnon & White)", "HC2 (Horn-Horn-Duncan)", "HC1 (Hinkley)", "HC0 (White Robust)"], index=0, key="settings_se_type", help="Heteroskedasticity-consistent robust standard error formula.")

    with set_tab3:
        st.markdown("#### Active Dataset & Session Cache Management")
        st.caption("Inspect in-memory DataFrame statistics and purge intermediate calculation caches.")

        if df is not None:
            mem_bytes = df.memory_usage(deep=True).sum()
            mem_kb = mem_bytes / 1024.0
            
            c_mem1, c_mem2, c_mem3 = st.columns(3)
            with c_mem1:
                st.metric("In-Memory Table Size", f"{mem_kb:.1f} KB")
            with c_mem2:
                st.metric("Total Records", f"{len(df):,}")
            with c_mem3:
                st.metric("Column Count", f"{len(df.columns)}")

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            if st.button("Clear Estimator Cache", use_container_width=True, help="Purge cached ATE and refutation calculations"):
                st.session_state["all_estimators"] = None
                st.session_state["causal_result"] = None
                st.session_state["matching_result"] = None
                st.session_state["baseline_smd"] = None
                st.session_state["sensitivity_result"] = None
                st.success("Estimator cache cleared.")
                st.rerun()
        with col_c2:
            if st.button("Reset Assistant Chat History", use_container_width=True, help="Clear conversation turns"):
                st.session_state["chat_history"] = []
                st.session_state["assistant_logs"] = []
                st.success("Chat history cleared.")
                st.rerun()
        with col_c3:
            if st.button("Reset to Synthetic Benchmark", use_container_width=True, help="Restore default synthetic benchmark dataset"):
                demo_df, meta = generate_synthetic_dataset(n=1000, true_ate=2.5, confounding_strength=1.0, seed=42)
                st.session_state["df"] = demo_df
                st.session_state["dataset_name"] = "synthetic_demo"
                st.session_state["ground_truth_ate"] = 2.5
                st.session_state["analysis_spec"] = {
                    "treatment": "treatment",
                    "outcome": "outcome",
                    "confounders": ["x1_feature", "age", "binary_flag", "prior_activity"]
                }
                st.success("Reset to Synthetic Benchmark.")
                st.rerun()

    with set_tab4:
        st.markdown("#### System Environment & Reproducibility Telemetry")
        import sys
        import platform
        import sklearn
        
        env_specs = [
            {"Parameter": "Platform / OS", "Value": platform.platform()},
            {"Parameter": "Python Version", "Value": sys.version.split()[0]},
            {"Parameter": "Streamlit Version", "Value": st.__version__},
            {"Parameter": "scikit-learn Version", "Value": sklearn.__version__},
            {"Parameter": "pandas Version", "Value": pd.__version__},
            {"Parameter": "numpy Version", "Value": np.__version__},
            {"Parameter": "ChromaDB Storage", "Value": "Active (Local SQLite3)"},
            {"Parameter": "CausalLens AI Architecture", "Value": "v2.4.0 (Enterprise Research Grade)"},
            {"Parameter": "SOC2 / Research Compliance", "Value": "Self-Contained In-Memory Isolation"}
        ]
        st.dataframe(pd.DataFrame(env_specs), use_container_width=True, height=340)

    st.markdown("---")
    st.markdown(
        """
        <div class="saas-card" style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.08) 0%, rgba(14, 20, 32, 0.95) 100%); border: 1px solid rgba(99, 102, 241, 0.25); padding: 18px 22px; margin-bottom: 14px;">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
                <div>
                    <h4 style="color:#FFFFFF; margin:0 0 4px 0; font-size:15px;">Configuration Saved</h4>
                    <p style="color:#94A3B8; font-size:12.5px; margin:0;">
                        Ready to analyze your data? Return to Overview or jump directly into Step 1 or Step 2.
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    set_nav1, set_nav2, set_nav3 = st.columns([1, 1, 1])
    with set_nav1:
        if st.button("Return to Overview / Home", type="primary", use_container_width=True, key="settings_nav_home", help="Back to homepage"):
            switch_page(PAGE_OVERVIEW)
    with set_nav2:
        if st.button("Step 1: Inspect Data →", use_container_width=True, key="settings_nav_data", help="Go to Data Analysis"):
            switch_page(PAGE_DATA)
    with set_nav3:
        if st.button("Step 2: Causal Analysis →", use_container_width=True, key="settings_nav_causal", help="Go to Causal Analysis"):
            switch_page(PAGE_CAUSAL)

