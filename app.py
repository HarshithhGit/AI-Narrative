"""
Module Name: app.py
Version: 4.0.0 (Enterprise Production Framework)
Description: Main orchestration entrypoint for the Enterprise AI Data Analyst Suite PRO.
             Manages state mutation, responsive UI rendering loop, global cross-filtering, 
             and tabbed layout matrix synchronization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import traceback

# Core Application Module Imports
from etl_pipeline import AutomatedETLPipeline, OutlierEngine, FeatureSchemaBlueprintGenerator
from bi_engine import AdvancedBIEngine
from insights import DeepStatisticalInsightsEngine
from automl import AutonomousMLEngine
from reporting import EnterprisePDFCompiler
from dashboard_report import compute_report_facts, DashboardPDFReport
from llm_narrator import LLMNarrator, provider_names, default_model, env_key_for, OFFLINE
import os
from cv_auditor import render_cv_workspace
from manifest_logs import log_event, render_manifest_workspace, ensure_log

# Initialize high-fidelity layout config
st.set_page_config(
    page_title="Enterprise AI Data Analyst Suite PRO",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Fidelity CSS styling for Power BI emulation layout
st.markdown("""
<style>
    body {
        color: #0F172A;
        background-color: #F8FAFC;
    }
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .bi-card-metric {
        background: #FFFFFF;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        text-align: center;
    }
    .bi-metric-value {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #1E3A8A !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #F1F5F9;
        padding: 6px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: 600;
        color: #475569;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #2563EB !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_states():
    """Ensures consistent registration of analytical session states across runtime boundaries."""
    state_defaults = {
        "raw_data": None,
        "production_pipeline": None,
        "filtered_pipeline": None,
        "cross_filter_target_dim": "None",
        "cross_filter_active_bounds": [],
        "interactive_drilldown_history": [],
        "active_model_registry": {},
        "etl_transformation_log": [],
        "last_selected_chart_interaction": None,
        "report_pdf_bytes": None,
        "report_narrative": None,
        "report_meta": None,
        "dataset_sig": None,
        "dataset_name": None,
        "reports_generated": 0,
        "images_audited": 0
    }
    for key, default in state_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

initialize_session_states()
ensure_log()

st.title("📊 Enterprise AI Data Analyst Suite PRO")
st.caption("Next-Generation Multi-Dimensional BI Engine, Autonomous Statistical Optimization & Automated Machine Learning Pipeline")

# Core Side Panels Configuration Workspace
st.sidebar.header("📁 Operational Workspace Manager")
workspace_mode = st.sidebar.radio(
    "Select Operating Environment Mode",
    ["Analytical Engine Core", "Multimodal CV Matrix Auditor", "Enterprise Pipeline Manifest Logs"]
)

if workspace_mode == "Analytical Engine Core":
    uploaded_file = st.sidebar.file_uploader(
        "📥 Inject Corporate Transactional Dataset (CSV)",
        type=["csv"],
        help="Upload structural transaction matrices to activate the multi-tier reporting framework loops."
    )
    
    if not uploaded_file:
        st.info("💡 Pipeline Ready for Execution. Provide a CSV dataset within the panel control workspace to launch processing passes.")
    else:
        # Load and parse safely via internal pipeline engine
        # reload whenever a DIFFERENT file is uploaded (name/size signature changes)
        file_sig = (uploaded_file.name, getattr(uploaded_file, "size", None))
        if st.session_state.raw_data is None or st.session_state.dataset_sig != file_sig:
            try:
                st.session_state.raw_data = pd.read_csv(uploaded_file)
                st.session_state.production_pipeline = st.session_state.raw_data.copy()
                st.session_state.filtered_pipeline = st.session_state.raw_data.copy()
                st.session_state.dataset_sig = file_sig
                st.session_state.dataset_name = uploaded_file.name
                st.session_state.report_pdf_bytes = None
                log_event("Data", "Dataset loaded", uploaded_file.name, None, st.session_state.raw_data.shape)
            except Exception as e:
                log_event("Error", "Dataset load failed", str(e), status="error")
                st.error(f"Critical Parsing Exception Triggered during base frame load: {str(e)}")
                st.stop()
                
        df_working = st.session_state.production_pipeline
        
        # Clean the column headers instantly to avoid string spaces crashing Plotly parameters maps
        df_working = AutomatedETLPipeline.clean_column_identifiers(df_working)
        
        # Parse schema dynamic properties mapping natively
        num_cols = df_working.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df_working.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        # Universal Cross-Filtering Controls Configuration Panel
        st.sidebar.markdown("---")
        st.sidebar.subheader("🎯 Cross-Filtering Interaction Control Plane")
        
        if len(cat_cols) > 0:
            global_filter_dim = st.sidebar.selectbox("Anchor Filtering Dimension", ["None"] + cat_cols)
            if global_filter_dim != "None":
                unique_bounds = df_working[global_filter_dim].dropna().unique().tolist()
                selected_bounds = st.sidebar.multiselect(
                    f"Filter Bounds: {global_filter_dim}",
                    unique_bounds,
                    default=unique_bounds
                )
                st.session_state.filtered_pipeline = df_working[df_working[global_filter_dim].isin(selected_bounds)]
            else:
                st.session_state.filtered_pipeline = df_working.copy()
        else:
            st.session_state.filtered_pipeline = df_working.copy()
            
        # Global Numeric Slicing Control Allocation Array
        if len(num_cols) > 0:
            slice_col = st.sidebar.selectbox("Continuous Slice Target Axis", ["None"] + num_cols)
            if slice_col != "None":
                min_s = float(st.session_state.filtered_pipeline[slice_col].min())
                max_s = float(st.session_state.filtered_pipeline[slice_col].max())
                if min_s != max_s:
                    chosen_range = st.sidebar.slider(f"Range: {slice_col}", min_s, max_s, (min_s, max_s))
                    st.session_state.filtered_pipeline = st.session_state.filtered_pipeline[
                        (st.session_state.filtered_pipeline[slice_col] >= chosen_range[0]) & 
                        (st.session_state.filtered_pipeline[slice_col] <= chosen_range[1])
                    ]

        # Application Workspace Tab Layout Orchestration Routing Array
        t_summary, t_etl, t_bi_studio, t_stats, t_ml, t_publish = st.tabs([
            "🏠 Operations Overview",
            "🧹 Data Pipeline Studio",
            "📊 Native BI Canvas (15 Charts)",
            "🔬 Diagnostic Statistics",
            "🧠 Predictive ML Lab",
            "📥 Enterprise Publisher"
        ])
        
        # ----------------------------------------------------
        # TAB 1: EXECUTIVE SUMMARY MAPPING
        # ----------------------------------------------------
        with t_summary:
            st.subheader("📋 Executive Operational Summary Dossier View")
            c_observations = st.session_state.filtered_pipeline
            
            mc1, mc2, mc3, mc4 = st.columns(4)
            with mc1:
                st.markdown(f"<div class='bi-card-metric'><div style='font-size:0.85rem;color:#64748B;'>OBSERVATION VOLUME</div><div class='bi-metric-value'>{c_observations.shape[0]:,}</div></div>", unsafe_allow_html=True)
            with mc2:
                st.markdown(f"<div class='bi-card-metric'><div style='font-size:0.85rem;color:#64748B;'>SCHEMA DIMENSIONALITY</div><div class='bi-metric-value'>{c_observations.shape[1]}</div></div>", unsafe_allow_html=True)
            with mc3:
                null_density = int(c_observations.isnull().sum().sum())
                st.markdown(f"<div class='bi-card-metric'><div style='font-size:0.85rem;color:#64748B;'>NULL CELL CONCENTRATION</div><div class='bi-metric-value'>{null_density:,}</div></div>", unsafe_allow_html=True)
            with mc4:
                redundancies = int(c_observations.duplicated().sum())
                st.markdown(f"<div class='bi-card-metric'><div style='font-size:0.85rem;color:#64748B;'>DUPLICATED SIGNATURES</div><div class='bi-metric-value'>{redundancies:,}</div></div>", unsafe_allow_html=True)
                
            st.markdown("---")
            col_left, col_right = st.columns([7, 3])
            with col_left:
                st.markdown("### 🔍 Live Frame Workspace Target Monitor")
                st.dataframe(c_observations.head(100), use_container_width=True)
            with col_right:
                st.markdown("### 🤖 Algorithmic System Summary Engine")
                with st.container(border=True):
                    # Call automated statistical insights core directly
                    system_insights = DeepStatisticalInsightsEngine.compute_topology_insights(c_observations, num_cols, cat_cols)
                    for item in system_insights[:5]:
                        st.info(item)
                        
        # ----------------------------------------------------
        # TAB 2: DATA PIPELINE STUDIO (ETL)
        # ----------------------------------------------------
        with t_etl:
            st.subheader("🧹 Production ETL Strategy & Schema Sanitization")
            col_etl_in, col_etl_out = st.columns(2)
            
            with col_etl_in:
                st.markdown("#### Transform Actions Menu Selector")
                impute_method = st.selectbox("Missing Cell Imputation Policy Framework", ["Impute via Mean/Mode Combination Strategy", "Drop Degenerate Structural Invariants"])
                scaling_method = st.selectbox("Continuous Quantitative Scaling Normalization Matrix", ["Omit Normalization Transformations", "StandardScaler", "MinMaxScaler", "RobustScaler", "QuantileTransformer", "PowerTransformer"])
                drop_dupes = st.checkbox("Execute Automatic Redundancy Deduplication Cleanse", value=True)
                noise_suppression = st.checkbox("Execute Low Frequency Categorical Noise Pooling", value=False)
                
                if st.button("⚡ Execute Mutation Sequences across Active Production State", use_container_width=True):
                    with st.spinner("Mutating production pipeline tensor array spaces..."):
                        mutated_df = st.session_state.production_pipeline.copy()
                        _shape_before = mutated_df.shape
                        _miss_before = int(mutated_df.isnull().sum().sum())
                        
                        # Apply data transformations pipelines
                        mutated_df = AutomatedETLPipeline.impute_missing_values(mutated_df, num_cols, cat_cols, impute_method)
                        if noise_suppression:
                            mutated_df = AutomatedETLPipeline.suppress_categorical_noise(mutated_df, cat_cols)
                        mutated_df = AutomatedETLPipeline.execute_feature_scaling(mutated_df, num_cols, scaling_method)
                        
                        if drop_dupes:
                            mutated_df.drop_duplicates(inplace=True)
                            
                        log_event("ETL", "Mutation sequence executed",
                                  f"impute={impute_method}; scaling={scaling_method}; dedupe={drop_dupes}; "
                                  f"noise_pooling={noise_suppression}; missing {_miss_before:,} -> {int(mutated_df.isnull().sum().sum()):,}",
                                  _shape_before, mutated_df.shape)
                        st.session_state.production_pipeline = mutated_df
                        st.session_state.filtered_pipeline = mutated_df.copy()
                        st.success("Production data pipeline schema execution sequence processed flawlessly.")
                        st.rerun()
                        
            with col_etl_out:
                st.markdown("#### Pipeline Variable Integrity Matrix Tracker")
                profile_frame = FeatureSchemaBlueprintGenerator.run_comprehensive_profile(st.session_state.production_pipeline)
                st.dataframe(profile_frame, use_container_width=True)
                
        # ----------------------------------------------------
        # TAB 3: NATIVE BI CANVAS ENGINE (15 WIDGET FORMATS)
        # ----------------------------------------------------
        with t_bi_studio:
            st.subheader("📊 Enterprise Visualization Studio Grid View Layout Canvas")
            
            bi_selector = st.selectbox("Target Native Visual Component Asset Mode", [
                "Clustered Bar Chart", "Clustered Column Chart", "Line Chart",
                "Pie Chart", "Donut Chart", "Treemap", "Scatter Plot",
                "Table", "Matrix", "Card", "KPI", "Gauge", 
                "Waterfall Chart", "Funnel Chart", "Map"
            ])
            
            # Delegate chart construction and reactive loops down to independent module layer
            AdvancedBIEngine.render_chart_component(bi_selector, st.session_state.filtered_pipeline, num_cols, cat_cols)
            
        # ----------------------------------------------------
        # TAB 4: DIAGNOSTIC STATISTICS ARCHITECTURE
        # ----------------------------------------------------
        with t_stats:
            st.subheader("🔬 Diagnostic Statistics & Mathematical Inference Testing Workspace")
            DeepStatisticalInsightsEngine.render_statistical_workspace(st.session_state.filtered_pipeline, num_cols, cat_cols)
                
        # ----------------------------------------------------
        # TAB 5: PREDICTIVE ML LAB OPERATIONAL CORE
        # ----------------------------------------------------
        with t_ml:
            st.subheader("🧠 High-Performance Predictive Optimization Workspace Machine Learning Frameworks")
            AutonomousMLEngine.render_ml_studio_workspace(st.session_state.filtered_pipeline, num_cols, cat_cols)
            
        # ----------------------------------------------------
        # TAB 6: ENTERPRISE PUBLISHING ENGINE AND REPORTING
        # ----------------------------------------------------
        with t_publish:
            st.subheader("📥 AI-Narrated Dashboard PDF Publisher")
            st.caption("Computes statistics, ML baselines and charts, asks an LLM to write the narrative, "
                       "then compiles an 8-page dashboard PDF that explains every feature of this suite.")
            doc_subject = st.text_input("Report Title", "Operational Performance Intelligence Briefing Report")

            st.markdown("#### 🤖 LLM Narrator Settings")
            lc1, lc2, lc3 = st.columns([1.2, 1.2, 1.6])
            with lc1:
                provider = st.selectbox("LLM Provider", provider_names(), index=0,
                                        help="Choose 'Offline' to run without any API key.")
            with lc2:
                model_name = st.text_input("Model", value=default_model(provider), key=f"model_{provider}")
            with lc3:
                secret_key = ""
                try:
                    env_name = {"Anthropic Claude": "ANTHROPIC_API_KEY", "OpenAI": "OPENAI_API_KEY",
                                "Groq (free tier)": "GROQ_API_KEY", "Google Gemini": "GEMINI_API_KEY"}.get(provider, "")
                    secret_key = st.secrets.get(env_name, "") if env_name else ""
                except Exception:
                    secret_key = ""
                api_key = st.text_input("API Key", type="password",
                                        value=secret_key or env_key_for(provider),
                                        disabled=(provider == OFFLINE),
                                        help="Stored only in this session. Or set it in .streamlit/secrets.toml / environment variables.")
            st.info("🔒 Privacy: only aggregated statistics (counts, means, test results) are sent to the LLM. "
                    "Raw rows never leave the app.")

            if st.button("🖨️ Generate AI Dashboard Report (PDF)", use_container_width=True, type="primary"):
                try:
                    report_df = st.session_state.filtered_pipeline
                    with st.spinner("1/3 Computing statistics, correlations and ML baselines..."):
                        facts, artifacts = compute_report_facts(report_df, num_cols, cat_cols)
                    with st.spinner("2/3 LLM is writing the narrative..."):
                        narrative, meta = LLMNarrator(provider, api_key, model_name).generate(facts)
                    with st.spinner("3/3 Rendering dashboard PDF..."):
                        pdf_io = DashboardPDFReport.build(report_df, num_cols, cat_cols, doc_subject,
                                                          narrative, facts, artifacts, meta)
                    st.session_state.report_pdf_bytes = pdf_io.getvalue()
                    st.session_state.report_narrative = narrative
                    st.session_state.report_meta = meta
                    st.session_state.reports_generated += 1
                    log_event("Report", "AI dashboard PDF generated",
                              f"provider={meta.get('provider')}; model={meta.get('model')}; llm_used={meta.get('used_llm')}",
                              report_df.shape, report_df.shape)
                except Exception as e_rep:
                    log_event("Error", "Report generation failed", str(e_rep), status="error")
                    st.error(f"Report generation failed: {e_rep}")
                    st.code(traceback.format_exc())

            if st.session_state.report_pdf_bytes:
                meta = st.session_state.report_meta or {}
                if meta.get("used_llm"):
                    st.success(f"✅ Narrative written by {meta['provider']} ({meta['model']}).")
                else:
                    st.warning(meta.get("error") or "Narrative generated by the offline rule engine.")
                st.download_button(
                    label="💾 Download Dashboard PDF",
                    data=st.session_state.report_pdf_bytes,
                    file_name="AI_Dashboard_Report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
                with st.expander("📝 Preview the narrative text"):
                    nar = st.session_state.report_narrative or {}
                    for key, val in nar.items():
                        st.markdown(f"**{key.replace('_', ' ').title()}**")
                        if isinstance(val, list):
                            for item in val:
                                st.markdown(f"- {item}")
                        else:
                            st.write(val)

            with st.expander("📄 Classic tabular PDF (legacy report)"):
                if st.button("Build classic PDF"):
                    legacy = EnterprisePDFCompiler.build_pdf_document_stream(
                        st.session_state.filtered_pipeline, num_cols, cat_cols, doc_subject)
                    st.download_button("💾 Download classic PDF", legacy.getvalue(),
                                       file_name="Classic_Report.pdf", mime="application/pdf")

elif workspace_mode == "Multimodal CV Matrix Auditor":
    render_cv_workspace()

else:
    render_manifest_workspace()
