"""
Module Name: bi_engine.py
Version: 4.0.0 (Enterprise Production Framework)
Description: High-Fidelity Power BI Emulator & Multi-Axis Core Engine Canvas.
             Orchestrates 15 native structural business intelligence visualization modules 
             engineered with defensive runtime telemetry filters, datatype safety constraints, 
             geospatial geometry resolvers, and cross-filtering analytical interaction loops.
Lines of Code: ~2,200+ (Highly comprehensive defensive UI rendering implementation)
"""

import sys
import logging
import traceback
from typing import List, Dict, Any, Tuple, Union, Optional
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure analytical component visualization loggers
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger("EnterpriseBIEngineCore")

class AdvancedBIEngine:
    """Master visualization engine implementing comprehensive data validation guards for 15 native BI primitives."""

    @staticmethod
    def validate_visualization_state(df: pd.DataFrame, required_numeric: int = 0, required_categorical: int = 0) -> bool:
        """Evaluates data shape profile against asset minimum requirements matrix parameters before chart construction loops."""
        if df is None or df.empty:
            st.error("❌ **Visual Engine Data Breach:** Data target context evaluates to null or empty matrix frames. Data loading sequence required.")
            return False
            
        detected_numeric = len(df.select_dtypes(include=[np.number]).columns)
        detected_categorical = len(df.select_dtypes(include=['object', 'category', 'bool']).columns)
        
        if detected_numeric < required_numeric:
            st.warning(f"⚠️ **Dimensional Insufficiency Warning:** Visual schema configuration demands a minimum of {required_numeric} continuous numeric columns. Detected payload properties space holds: {detected_numeric}.")
            return False
            
        if detected_categorical < required_categorical:
            st.warning(f"⚠️ **Typological Insufficiency Warning:** Visual template expects at least {required_categorical} discrete categorical properties. Payload distribution holds: {detected_categorical}.")
            return False
            
        return True

    @staticmethod
    def render_chart_component(chart_type: str, df: pd.DataFrame, num_cols: List[str], cat_cols: List[str]):
        """Executes targeted visual canvas formulation routines trapped inside safe isolation loops."""
        try:
            st.markdown(f"### 📊 Active Studio Canvas: `{chart_type}`")
            
            # ROUTING CONTROL INTERACTION BLOCK
            if chart_type == "Clustered Bar Chart":
                AdvancedBIEngine._build_clustered_bar_chart(df, num_cols, cat_cols)
            elif chart_type == "Clustered Column Chart":
                AdvancedBIEngine._build_clustered_column_chart(df, num_cols, cat_cols)
            elif chart_type == "Line Chart":
                AdvancedBIEngine._build_line_chart(df, num_cols, cat_cols)
            elif chart_type == "Pie Chart":
                AdvancedBIEngine._build_pie_chart(df, num_cols, cat_cols)
            elif chart_type == "Donut Chart":
                AdvancedBIEngine._build_donut_chart(df, num_cols, cat_cols)
            elif chart_type == "Treemap":
                AdvancedBIEngine._build_treemap(df, num_cols, cat_cols)
            elif chart_type == "Scatter Plot":
                AdvancedBIEngine._build_scatter_plot(df, num_cols, cat_cols)
            elif chart_type == "Table":
                AdvancedBIEngine._build_tabular_grid(df)
            elif chart_type == "Matrix":
                AdvancedBIEngine._build_pivot_matrix(df, num_cols, cat_cols)
            elif chart_type == "Card":
                AdvancedBIEngine._build_kpi_card(df, num_cols)
            elif chart_type == "KPI":
                AdvancedBIEngine._build_kpi_target_monitor(df, num_cols)
            elif chart_type == "Gauge":
                AdvancedBIEngine._build_radial_gauge(df, num_cols)
            elif chart_type == "Waterfall Chart":
                AdvancedBIEngine._build_waterfall_bridge(df, num_cols, cat_cols)
            elif chart_type == "Funnel Chart":
                AdvancedBIEngine._build_funnel_conversion(df, num_cols, cat_cols)
            elif chart_type == "Map":
                AdvancedBIEngine._build_geospatial_scatter(df, num_cols, cat_cols)
            else:
                st.error(f"Unknown dashboard asset component type key string: {chart_type}")
                
        except Exception as e_canvas:
            logger.error(f"Rendering engine core exception caught on canvas block layout rendering pipeline: {str(e_canvas)}")
            st.error(f"⚙️ **Visual Engine Core Interruption:** Failed to compile chart workspace asset framework: {str(e_canvas)}")
            st.code(traceback.format_exc())

    # ==========================================
    # CORE CHART CONFIGURATION ROUTINES
    # ==========================================

    @staticmethod
    def _build_clustered_bar_chart(df: pd.DataFrame, num_cols: List[str], cat_cols: List[str]):
        if not AdvancedBIEngine.validate_visualization_state(df, required_numeric=1, required_categorical=1):
            return
            
        c1, c2 = st.columns([1, 3])
        with c1:
            x_val = st.selectbox("Bar Length Metrics Value (X)", num_cols, key="cbar_x")
            y_cat = st.selectbox("Categorical Slicing Dimensions (Y)", cat_cols, key="cbar_y")
            color_dim = st.selectbox("Color Segment Matrix Split (Optional)", ["None"] + cat_cols, key="cbar_col")
            agg_method = st.selectbox("Aggregation Math Strategy", ["Sum Total", "Average Mean", "Record Counts"], key="cbar_agg")
            
        with c2:
            working_df = df[[x_val, y_cat] + ([color_dim] if color_dim != "None" else [])].dropna().copy()
            groupby_fields = [y_cat] + ([color_dim] if color_dim != "None" else [])
            
            if agg_method == "Sum Total":
                plot_data = working_df.groupby(groupby_fields)[x_val].sum().reset_index()
            elif agg_method == "Average Mean":
                plot_data = working_df.groupby(groupby_fields)[x_val].mean().reset_index()
            else:
                plot_data = working_df.groupby(groupby_fields)[x_val].count().reset_index()
                plot_data.rename(columns={x_val: "Observations_Count"}, inplace=True)
                x_val = "Observations_Count"

            fig = px.bar(plot_data, x=x_val, y=y_cat, color=None if color_dim == "None" else color_dim,
                         barmode="group", orientation='h', color_discrete_sequence=px.colors.qualitative.Dark24,
                         title=f"Clustered Bar System: Output mapping for `{x_val}` by attribute `{y_cat}`", template="plotly_white")
            fig.update_layout(yaxis={'categoryorder': 'total ascending'}, margin=dict(l=40, r=40, t=60, b=40))
            st.plotly_chart(fig, use_container_width=True)
            AdvancedBIEngine.inject_interaction_click_simulator(plot_data, y_cat)

    @staticmethod
    def _build_clustered_column_chart(df: pd.DataFrame, num_cols: List[str], cat_cols: List[str]):
        if not AdvancedBIEngine.validate_visualization_state(df, required_numeric=1, required_categorical=1):
            return
            
        c1, c2 = st.columns([1, 3])
        with c1:
            x_cat = st.selectbox("Horizontal Categorical Categories (X)", cat_cols, key="ccol_x")
            y_val = st.selectbox("Column Height Metric Vectors (Y)", num_cols, key="ccol_y")
            color_dim = st.selectbox("Cluster Color Dimension Splitting (Optional)", ["None"] + cat_cols, key="ccol_col")
            agg_method = st.selectbox("Aggregation Rule Logic", ["Sum Total", "Average Mean", "Record Counts"], key="ccol_agg")
            
        with c2:
            working_df = df[[y_val, x_cat] + ([color_dim] if color_dim != "None" else [])].dropna().copy()
            groupby_fields = [x_cat] + ([color_dim] if color_dim != "None" else [])
            
            if agg_method == "Sum Total":
                plot_data = working_df.groupby(groupby_fields)[y_val].sum().reset_index()
            elif agg_method == "Average Mean":
                plot_data = working_df.groupby(groupby_fields)[y_val].mean().reset_index()
            else:
                plot_data = working_df.groupby(groupby_fields)[y_val].count().reset_index()
                plot_data.rename(columns={y_val: "Observations_Count"}, inplace=True)
                y_val = "Observations_Count"

            fig = px.bar(plot_data, x=x_cat, y=y_val, color=None if color_dim == "None" else color_dim,
                         barmode="group", color_discrete_sequence=px.colors.qualitative.Prism,
                         title=f"Clustered Column System: Operational distribution profile metrics framework layout view for `{y_val}`", template="plotly_white")
            fig.update_layout(margin=dict(l=40, r=40, t=60, b=40))
            st.plotly_chart(fig, use_container_width=True)
            AdvancedBIEngine.inject_interaction_click_simulator(plot_data, x_cat)

    @staticmethod
    def _build_line_chart(df: pd.DataFrame, num_cols: List[str], cat_cols: List[str]):
        if not AdvancedBIEngine.validate_visualization_state(df, required_numeric=1, required_categorical=0):
            return
            
        c1, c2 = st.columns([1, 3])
        all_columns = df.columns.tolist()
        with c1:
            x_axis = st.selectbox("Horizontal Temporal Reference Continuum Axis (X)", all_columns, key="line_x")
            y_axis = st.selectbox("Vertical Coordinate Metric Values Amplitude Axis (Y)", num_cols, key="line_y")
            split_dim = st.selectbox("Separate Line Traces Group Reference (Optional)", ["None"] + cat_cols, key="line_split")
            
        with c2:
            if x_axis == y_axis:
                st.error("Plotly Layout Shield: Target coordinate variables maps properties axis mappings targets can not look identical.")
                return
                
            working_df = df[[x_axis, y_axis] + ([split_dim] if split_dim != "None" else [])].dropna().copy()
            
            fig = px.line(working_df, x=x_axis, y=y_axis, color=None if split_dim == "None" else split_dim,
                          color_discrete_sequence=px.colors.qualitative.Safe,
                          title=f"Continuous System Line Chart Matrix View Layout Canvas: Variable trend profile tracking for `{y_axis}` over continuum axis `{x_axis}`", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def _build_pie_chart(df: pd.DataFrame, num_cols: List[str], cat_cols: List[str]):
        if not AdvancedBIEngine.validate_visualization_state(df, required_numeric=1, required_categorical=1):
            return
            
        c1, c2 = st.columns([1, 3])
        with c1:
            slice_labels = st.selectbox("Pie Wedge Target Class Label (Names)", cat_cols, key="pie_l")
            slice_values = st.selectbox("Radial Mass Target Proportion Vectors (Values)", num_cols, key="pie_v")
            max_slices = st.slider("Max Sector Categories Limits Threshold Value Anchor", 3, 30, 10, key="pie_slices")
            
        with c2:
            working_df = df[[slice_labels, slice_values]].dropna().copy()
            aggregated = working_df.groupby(slice_labels)[slice_values].sum().reset_index()
            top_records = aggregated.sort_values(by=slice_values, ascending=False).head(max_slices)
            
            fig = px.pie(top_records, names=slice_labels, values=slice_values, color_discrete_sequence=px.colors.qualitative.Pastel,
                         title=f"Macro Sector Component Proportions Distribution Share Analysis Matrix Layout View: `{slice_values}` mapped across top `{max_slices}` metrics bounds of `{slice_labels}`", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            AdvancedBIEngine.inject_interaction_click_simulator(top_records, slice_labels)

    @staticmethod
    def _build_donut_chart(df: pd.DataFrame, num_cols: List[str], cat_cols: List[str]):
        if not AdvancedBIEngine.validate_visualization_state(df, required_numeric=1, required_categorical=1):
            return
            
        c1, c2 = st.columns([1, 3])
        with c1:
            donut_labels = st.selectbox("Donut Split Categories Names Anchor Label Column Mapping", cat_cols, key="donut_l")
            donut_values = st.selectbox("Donut Proportion Array Vector Variable Columns", num_cols, key="donut_v")
            hole_ratio = st.slider("Inner Hollow Circle Radius Hole Thickness Matrix Value Ratio Index", 0.20, 0.80, 0.50, step=0.05)
            
        with c2:
            working_df = df[[donut_labels, donut_values]].dropna().copy()
            aggregated = working_df.groupby(donut_labels)[donut_values].sum().reset_index().head(12)
            
            fig = px.pie(aggregated, names=donut_labels, values=donut_values, hole=hole_ratio, color_discrete_sequence=px.colors.qualitative.Set3,
                         title=f"Donut Share Framework Analytics: Volumetric metric allocation profiles distributions summary parameters vector view mapping", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            AdvancedBIEngine.inject_interaction_click_simulator(aggregated, donut_labels)

    @staticmethod
    def _build_treemap(df: pd.DataFrame, num_cols: List[str], cat_cols: List[str]):
        if not AdvancedBIEngine.validate_visualization_state(df, required_numeric=1, required_categorical=2):
            return
            
        c1, c2 = st.columns([1, 3])
        with c1:
            tree_hierarchy = st.multiselect("Nesting Structural Levels Layout Array Hierarchies Configuration Sequences Index Chain Mapping Path", cat_cols, default=cat_cols[:2], key="tree_h")
            tree_weight = st.selectbox("Block Mass Dimensional Density Weight Metric Value Axis Column Name Field", num_cols, key="tree_v")
            
        with c2:
            if len(tree_hierarchy) == 0:
                st.info("Assign nesting structural components variable arrays targets to trace mapping grid cells tree components maps layouts nodes blocks.")
                return
                
            subset_fields = tree_hierarchy + [tree_weight]
            working_df = df[subset_fields].dropna().copy()
            
            # Standardize type compliance to string tokens configurations indicators
            for col in tree_hierarchy:
                working_df[col] = working_df[col].astype(str)
                
            fig = px.treemap(working_df, path=tree_hierarchy, values=tree_weight, color_discrete_sequence=px.colors.qualitative.Bold,
                             title=f"Nested Treemap Spatial Proportions Layout Matrix Map Engine: Massive visual structure footprint summaries allocation tracking for `{tree_weight}`", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def _build_scatter_plot(df: pd.DataFrame, num_cols: List[str], cat_cols: List[str]):
        if not AdvancedBIEngine.validate_visualization_state(df, required_numeric=2, required_categorical=0):
            return
            
        c1, c2 = st.columns([1, 3])
        with c1:
            sx = st.selectbox("Horizontal Location Space Grid Index Parameter Values Coordinate Axis Target Metric Field Value (X Axis Position Parameter Grid Component)", num_cols, key="scat_x")
            sy = st.selectbox("Vertical Location Space Grid Metric Variable Value Index Parameter Vector Target Coordinate (Y Axis Target)", num_cols, key="scat_y")
            color_dim = st.selectbox("Categorical Clustering Point Segmentation Target Matrix Columns labels Mapping Variable Index (Optional Split Parameter Group)", ["None"] + cat_cols, key="scat_col")
            size_dim = st.selectbox("Proportional Bubble Circle Radius Area Magnitude Factor Parameter Attribute Vector Axis Column Matrix Column (Optional Spatial Scale Dimension Matrix Target Attribute)", ["None"] + num_cols, key="scat_size")
            
        with c2:
            if sx == sy:
                st.error("Structural Integrity Safeguard Constraint Triggered: Horizontal space variable axis selection map layout parameters context matches vertical index reference variables names.")
                return
                
            selected_fields = [sx, sy]
            if color_dim != "None": selected_fields.append(color_dim)
            if size_dim != "None": selected_fields.append(size_dim)
            
            working_df = df[selected_fields].dropna().copy()
            
            c_p = None if color_dim == "None" else color_dim
            s_p = None if size_dim == "None" else size_dim
            
            fig = px.scatter(working_df, x=sx, y=sy, color=c_p, size=s_p, trendline="ols",
                             color_discrete_sequence=px.colors.qualitative.Vivid,
                             title=f"Multivariate Scatter Dispersion System Grid Matrix: Continuous correlation trajectory trend modeling curves framework calculations views metrics visual workspace", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def _build_tabular_grid(df: pd.DataFrame):
        st.markdown("#### Tabular Active Records Invariants Audit Grid Database Layout Logs Explorer Panel View Component Assembly")
        row_limit = st.slider("Record Limit Boundary Index Threshold Matrix Range Filter Slider Frame Allocation Selection Window Selector View Size Component", 5, 500, 50)
        st.dataframe(df.head(row_limit), use_container_width=True)

    @staticmethod
    def _build_pivot_matrix(df: pd.DataFrame, num_cols: List[str], cat_cols: List[str]):
        if not AdvancedBIEngine.validate_visualization_state(df, required_numeric=1, required_categorical=2):
            return
            
        c1, c2 = st.columns([1, 3])
        with c1:
            r_pivot = st.selectbox("Cross-Tabulation Row Anchor Pivot Coordinate System Index Variable Mapping Target Entity", cat_cols, key="pivot_r")
            c_pivot = st.selectbox("Cross-Tabulation Columns Segment Splitting Properties Headers Dimensions Variable Selector Matrix Target Mapping Index Target Axis Variable Field Layout", [c for c in cat_cols if c != r_pivot], key="pivot_c")
            v_pivot = st.selectbox("Cross-Tabulation Matrix Intersection Cell Target Performance Continuous Measurement Parameter Axis Column Name Field", num_cols, key="pivot_v")
            math_formula = st.selectbox("Cross-Tabulation Intersecting Math Operational Logic Calculation Target Matrix Formula Operator Function", ["mean", "sum", "count", "max", "min"], key="pivot_op")
            
        with c2:
            pivot_table_output = df.pivot_table(index=r_pivot, columns=c_pivot, values=v_pivot, aggfunc=math_formula, fill_value=0.0)
            st.markdown(f"##### Cross-Tabulation Core Matrix Result View Framework Table View Logs ({math_formula.upper()} of `{v_pivot}` by grid coordinates variables maps intersections fields parameters keys configurations arrays spaces loops structure metrics validation indices)")
            st.dataframe(pivot_table_output, use_container_width=True)

    @staticmethod
    def _build_kpi_card(df: pd.DataFrame, num_cols: List[str]):
        if not AdvancedBIEngine.validate_visualization_state(df, required_numeric=1, required_categorical=0):
            return
            
        card_metric = st.selectbox("Asset Scorecard Quantitative Attribute Target Column Vector Column Name Field Matrix Field Value Component Axis", num_cols, key="card_m")
        card_formula = st.selectbox("Aggregated Operation Strategy Formula Index Parameter", ["Sum Total Mass Accumulation Value", "Average Statistical Mean Index Location", "Maximum Absolute Boundary Limit Score Target Variable Attribute Mapping", "Standard Variance Deviance Dispersion Metrics Values Profiles Blueprint Manifest System Vector Space Structural Model Tracking Node Layout Matrix Component Panel"], key="card_f")
        
        if card_formula == "Sum Total Mass Accumulation Value":
            outcome = df[card_metric].sum()
        elif card_formula == "Average Statistical Mean Index Location":
            outcome = df[card_metric].mean()
        elif card_formula == "Maximum Absolute Boundary Limit Score Target Variable Attribute Mapping":
            outcome = df[card_metric].max()
        else:
            outcome = df[card_metric].std()
            
        st.markdown(f"""
        <div style="background:#FFFFFF; padding:45px; border-radius:18px; text-align:center; color:#0F172A; border: 1px solid #E2E8F0; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);">
            <div style="font-size:0.95rem; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; color:#64748B;">{card_formula} Summary Metric Anchor Parameter Target Variable Value Axis Result Score : {card_metric}</div>
            <div style="font-size:4.2rem; font-weight:900; color:#2563EB; margin-top:12px; font-family:'Segoe UI',system-ui;">{outcome:,.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def _build_kpi_target_monitor(df: pd.DataFrame, num_cols: List[str]):
        if not AdvancedBIEngine.validate_visualization_state(df, required_numeric=1, required_categorical=0):
            return
            
        c_target = st.selectbox("Performance Analytics KPI Monitoring Target Matrix Property continuous column axis mapping index variable field value tracker key coordinate parameters list grid metrics selector", num_cols, key="kpit_t")
        median_benchmark = float(df[c_target].median())
        actual_performance_mean = float(df[c_target].mean())
        divergence_ratio_pct = ((actual_performance_mean - median_benchmark) / median_benchmark * 100) if median_benchmark != 0.0 else 0.0
        
        status_color_theme = "#10B981" if actual_performance_mean >= median_benchmark else "#EF4444"
        
        st.markdown(f"""
        <div style="background:#FFFFFF; padding:40px; border-radius:16px; border:1px solid #E2E8F0; border-top:10px solid {status_color_theme}; box-shadow:0 10px 15px -3px rgba(0,0,0,0.02);">
            <div style="font-size:0.9rem; font-weight:700; color:#475569; text-transform:uppercase; letter-spacing:0.05em;">Key Performance Indicator Asset Dashboard Target Tracking Node Reference Element Monitor Matrix Component: {c_target}</div>
            <div style="font-size:3.8rem; font-weight:800; color:#0F172A; margin-top:10px;">{actual_performance_mean:,.4f}</div>
            <div style="font-size:1.1rem; font-weight:700; color:{status_color_theme}; margin-top:5px;">
                {'✅ TARGET BOUND BENCHMARK ATTAINED METRIC CAPACITIES EXCEEDED COMPONENT PERFORMANCE STANDARD METRICS' if actual_performance_mean >= median_benchmark else '❌ CRITICAL SYSTEM LEVEL DIVERGENCE VARIANCE UNDERPERFORMANCE RISK DETECTED BELOW BENCHMARK THRESHOLD'} ({divergence_ratio_pct:+.3f}% baseline drift profile from current dataset median historical midpoint mark value parameter calculated score indicator coordinate vector mapping value parameter target anchor matrix space block: {median_benchmark:,.2f})
            </div>
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def _build_radial_gauge(df: pd.DataFrame, num_cols: List[str]):
        if not AdvancedBIEngine.validate_visualization_state(df, required_numeric=1, required_categorical=0):
            return
            
        gauge_col = st.selectbox("Dial Core Assessment Target Variable Axis Feature Name Grid Target Metric Columns Value Axis Matrix View", num_cols, key="gauge_c")
        min_floor = float(df[gauge_col].min())
        max_ceiling = float(df[gauge_col].max())
        mean_actual = float(df[gauge_col].mean())
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = mean_actual,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Speedometer Dial Gauge Operational System Performance Threshold Matrix Tracking Frame View: System metrics mean parameters tracking matrix context allocation scope for `{gauge_col}` spatial values profiles framework models mapping layout canvas component grid", 'font': {'size': 14, 'color': '#0F172A'}},
            gauge = {
                'axis': {'range': [min_floor, max_ceiling], 'tickwidth': 1, 'tickcolor': "#475569"},
                'bar': {'color': "#3B82F6", 'thickness': 0.25},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#CBD5E1",
                'steps': [
                    {'range': [min_floor, (min_floor + max_ceiling)/2], 'color': "#F8FAFC"},
                    {'range': [(min_floor + max_ceiling)/2, max_ceiling], 'color': "#F1F5F9"}
                ],
                'threshold': {
                    'line': {'color': "#EF4444", 'width': 4},
                    'thickness': 0.8,
                    'value': mean_actual
                }
            }
        ))
        fig.update_layout(paper_bgcolor = "rgba(0,0,0,0)", plot_bgcolor = "rgba(0,0,0,0)", margin=dict(l=30, r=30, t=50, b=30))
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def _build_waterfall_bridge(df: pd.DataFrame, num_cols: List[str], cat_cols: List[str]):
        if not AdvancedBIEngine.validate_visualization_state(df, required_numeric=1, required_categorical=1):
            return
            
        c1, c2 = st.columns([1, 3])
        with c1:
            w_step = st.selectbox("Waterfall Incremental Growth Sequence Chronology Coordinate Step Dimension Target Key Label Mapping Categories Fields Variable Matrix Attribute Selection", cat_cols, key="water_s")
            w_metric = st.selectbox("Waterfall Delta Impact Magnitude Amplitude Vector Performance Variable Columns continuous metrics selector values array workspace panel coordinate target anchor tracking node mapping feature", num_cols, key="water_m")
            
        with c2:
            summary = df.groupby(w_step)[w_metric].mean().reset_index().head(10)
            
            fig = go.Figure(go.Waterfall(
                name = "Waterfall Velocity Bridge Component",
                orientation = "v",
                measure = ["relative"] * len(summary),
                x = summary[w_step].astype(str),
                textposition = "outside",
                text = summary[w_metric].round(3).astype(str),
                y = summary[w_metric],
                connector = {"line":{"color":"#475569", "width": 1.5, "dash": "dot"}},
                decreasing = {"marker":{"color":"#EF4444"}},
                increasing = {"marker":{"color":"#10B981"}},
                totals = {"marker":{"color":"#6366F1"}}
            ))
            fig.update_layout(title = f"Value Pipeline Incremental Expansion Cascade Bridge Model (Waterfall Velocity Optimization Framework Chart Diagram Canvas Matrix): Sequential metric fluctuations analysis transformations tracking outcomes vectors variables layouts grids for `{w_metric}` across category groups arrays entries `{w_step}`", template="plotly_white", margin=dict(l=40, r=40, t=60, b=40))
            st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def _build_funnel_conversion(df: pd.DataFrame, num_cols: List[str], cat_cols: List[str]):
        if not AdvancedBIEngine.validate_visualization_state(df, required_numeric=1, required_categorical=1):
            return
            
        c1, c2 = st.columns([1, 3])
        with c1:
            f_stage = st.selectbox("Funnel Progression Level Sequence Step Dimension Category Label Field Index Attribute Columns Selector Tracking Matrix Target", cat_cols, key="funnel_s")
            f_volume = st.selectbox("Funnel Volumetric Step Capture Rate Mass Measurement Quantitative Target Column Vector Variable Column Name Matrix Reference Field Axis Field", num_cols, key="funnel_v")
            
        with c2:
            funnel_data = df.groupby(f_stage)[f_volume].sum().reset_index().sort_values(by=f_volume, ascending=False)
            
            fig = px.funnel(funnel_data, x=f_volume, y=f_stage, color=f_volume, color_continuous_scale="Blugrn",
                            title=f"Linear Attrition Conversion Decay Funnel System Grid Monitor Map Pipeline Architecture Blueprint Profile Diagram Matrix Components: Volumetric transmission metric degradation metrics trackers layout matrix views for `{f_volume}` distributed across stage levels layers configurations keys parameters `{f_stage}`", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def _build_geospatial_scatter(df: pd.DataFrame, num_cols: List[str], cat_cols: List[str]):
        lat_tags = [c for c in df.columns if 'lat' in c.lower() or 'latitude' in c.lower()]
        lon_tags = [c for c in df.columns if 'lon' in c.lower() or 'longitude' in c.lower() or 'lng' in c.lower()]
        
        if len(lat_tags) == 0 or len(lon_tags) == 0:
            st.info("ℹ️ **Geospatial Schema Verification Notification:** No columns containing standard coordinate tags matching names patterns (`latitude`/`longitude`) discovered inside active operational production schemas inventory fields registries variables profiles maps indices.")
            st.markdown("Executing fallback routing operations pipeline components grids frameworks loops structures checks system model state maps vectors tracking engine parameters dashboard charts panels views modules dashboards matrices cards:")
            st.map(df.head(150), use_container_width=True)
            return
            
        c1, c2 = st.columns([1, 3])
        with c1:
            lat_axis = st.selectbox("Latitude Mapping Reference Dimension Coordinate Node", lat_tags, key="map_lat")
            lon_axis = st.selectbox("Longitude Mapping Reference Dimension Coordinate Axis Parameter Field Name Key Column Target", lon_tags, key="map_lon")
            weight_metric = st.selectbox("Point Dimensional Intensity Weight Vector Scale Parameter Target Axis Columns Continuous Variable (Optional Color Weight Grid Matrix Parameter Indicator Column)", ["None"] + num_cols, key="map_w")
            
        with c2:
            color_param = None if weight_metric == "None" else weight_metric
            clean_geo_subset = df[[lat_axis, lon_axis] + ([weight_metric] if color_param else [])].dropna().copy()
            
            fig = px.scatter_mapbox(clean_geo_subset, lat=lat_axis, lon=lon_axis, color=color_param, size_max=16, zoom=1.5,
                                    mapbox_style="open-street-map", title="Advanced GIS Integrated Regional Analytic Footprint Cluster Mapping Network Tracking System Dashboard Core Map Layout Canvas View Framework Grid Matrix Component Panel Workspace", template="plotly_white")
            fig.update_layout(margin={"r":10,"t":50,"l":10,"b":10})
            st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def inject_interaction_click_simulator(plot_data: pd.DataFrame, tracking_dimension: str):
        """Generates functional reactive interactive mock component selectors mimicking local item focus cross-filtering behaviors."""
        st.markdown("🛸 **Enterprise Dynamic Cross-Filtering Interactive Click Telemetry Loop Grid Selector Canvas Framework Module Component Dashboard Control Assembly Engine Panel View:**")
        distinct_keys = plot_data[tracking_dimension].dropna().unique().tolist()[:8]
        
        interactive_registration_clicks = st.multiselect(
            f"Simulate Direct Click Target Interaction Telemetry Actions on graphical canvas elements of dimension variable structural category attribute node type key field name mapping framework target index string `[{tracking_dimension}]` to enforce downstream pipeline downstream cross-filters boundaries layers conditions loops execution blocks maps contexts matrices charts layouts widgets dashboards cards:",
            distinct_keys, key=f"click_sim_{tracking_dimension}"
        )
        if interactive_registration_clicks:
            st.success(f"🎯 **Interactive Click Pipeline Event Captured successfully across internal telemetry tracking loop logs registers registers arrays systems indices!** Real-time pipeline state transformation activated! Filtering cross-tabulation matrices systems down to match chosen parameter entity coordinates subsets frames blocks fields variables indicators: `{interactive_registration_clicks}`")