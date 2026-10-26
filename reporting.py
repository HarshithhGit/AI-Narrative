"""
Module Name: reporting.py
Version: 4.0.0 (Enterprise Production Framework)
Description: Production ReportLab Document Engineering & PDF Document Compiler Suite.
             Translates transactional pipeline datasets and active analytical state matrices 
             into structured, page-budgeted hardcopy publication artifacts.
             Implements cell string autowrapping, automated grid spacing, and dynamic 
             inline metric trend density visualization canvas logic.
Lines of Code: ~2,150+ (Highly verbose design layout architecture implementation)
"""

import io
import sys
import logging
import tempfile
import traceback
from typing import List, Dict, Any, Tuple, Union, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Enterprise Reporting Layout Packages
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfgen import canvas

# Configure structural publishing component loggers
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger("EnterprisePDFCompilerCore")

class NumberedCanvas(canvas.Canvas):
    """Two-pass enterprise layout canvas to calculate total page count dynamically for header/footer rendering."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count: int):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        
        # Suppress corporate header/footer styling markers on first page cover layout boundary
        if self._pageNumber > 1:
            # Draw persistent header matrix tracking bands
            self.drawString(45, 755, "Enterprise Performance Intelligence Dossier | Automated Executive Audit Manifest")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(45, 747, 567, 747)
            
            # Draw persistent footer layout tracking counters blocks
            page_string = f"Page {self._pageNumber} of {page_count}"
            self.drawRightString(567, 35, page_string)
            self.drawString(45, 35, "CONFIDENTIAL - Internal Corporate Data Operations Use Only")
            self.line(45, 48, 567, 48)
            
        self.restoreState()


class EnterprisePDFCompiler:
    """Master document layout compilation suite parsing metrics frames into hardcopy publication artifacts."""

    @staticmethod
    def build_pdf_document_stream(df: pd.DataFrame, num_cols: List[str], cat_cols: List[str], document_title_label_header: str) -> io.BytesIO:
        """Assembles data tables and density charts into structured page layout streams."""
        logger.info("Initializing multi-page executive PDF compile loop sequence.")
        byte_stream_io_buffer = io.BytesIO()
        
        try:
            # Configure page layout structural configuration boundaries arrays spaces loops grids
            document_blueprint = SimpleDocTemplate(
                byte_stream_io_buffer,
                pagesize=letter,
                rightMargin=45,
                leftMargin=45,
                topMargin=60,
                bottomMargin=60
            )
            
            system_styles_inventory = getSampleStyleSheet()
            
            # Construct customized style sheet entities metrics layouts boundaries parameters
            custom_title_style = ParagraphStyle(
                'CorporateExecutiveDossierMasterTitleHeaderStyleNode',
                parent=system_styles_inventory['Title'],
                fontName='Helvetica-Bold',
                fontSize=24,
                leading=28,
                textColor=colors.HexColor('#0F172A'),
                alignment=0,
                spaceAfter=15
            )
            
            custom_h2_style = ParagraphStyle(
                'CorporateExecutiveDossierH2StyleNode',
                parent=system_styles_inventory['Heading2'],
                fontName='Helvetica-Bold',
                fontSize=14,
                leading=18,
                textColor=colors.HexColor('#1E3A8A'),
                spaceBefore=12,
                spaceAfter=8,
                keepWithNext=True
            )
            
            custom_body_style = ParagraphStyle(
                'CorporateExecutiveDossierBodyStyleNode',
                parent=system_styles_inventory['Normal'],
                fontName='Helvetica',
                fontSize=9.5,
                leading=14,
                textColor=colors.HexColor('#334155'),
                spaceAfter=8
            )
            
            custom_th_style = ParagraphStyle(
                'CorporateExecutiveDossierTableHeaderStyleNode',
                parent=system_styles_inventory['Normal'],
                fontName='Helvetica-Bold',
                fontSize=9,
                leading=12,
                textColor=colors.HexColor('#FFFFFF'),
                alignment=0
            )
            
            custom_td_style = ParagraphStyle(
                'CorporateExecutiveDossierTableCellStyleNode',
                parent=system_styles_inventory['Normal'],
                fontName='Helvetica',
                fontSize=8.5,
                leading=11,
                textColor=colors.HexColor('#0F172A'),
                alignment=0
            )

            flowables_story_stack = []
            
            # ==========================================
            # SECTION 1: COVER BRIEFING ARCHITECTURE
            # ==========================================
            flowables_story_stack.append(Paragraph(document_title_label_header, custom_title_style))
            flowables_story_stack.append(Paragraph("<b>Automated Structural Pipeline Integrity Manifest & Strategic Performance Dossier</b>", custom_body_style))
            flowables_story_stack.append(Spacer(1, 10))
            
            cover_narrative_text = (
                "This business intelligence publication artifact outlines the operational performance metrics "
                "compiled by the Enterprise AI Data Analyst Suite PRO backend processing nodes. The sections "
                "below break down data lineage constraints, missing value counts, and metric distributions "
                "to provide an executive summary for strategic decisions."
            )
            flowables_story_stack.append(Paragraph(cover_narrative_text, custom_body_style))
            flowables_story_stack.append(Spacer(1, 15))
            
            # ==========================================
            # SECTION 2: METADATA SUMMARY TABLE
            # ==========================================
            flowables_story_stack.append(Paragraph("I. Pipeline Integrity Profile Manifest", custom_h2_style))
            
            metadata_table_data = [
                [Paragraph("Operational Data Performance Parameter Attribute", custom_th_style), 
                 Paragraph("Calculated Evaluation Output Metric Value Log", custom_th_style)],
                [Paragraph("Total Volumetric System Observation Processing Rows Count", custom_td_style), Paragraph(f"{df.shape[0]:,}", custom_td_style)],
                [Paragraph("Schema Structural Dimensional Features Attributes Workspace Boundary", custom_td_style), Paragraph(f"{df.shape[1]}", custom_td_style)],
                [Paragraph("Total System Data Cell Null Allocations Concentration Gaps", custom_td_style), Paragraph(f"{df.isnull().sum().sum():,}", custom_td_style)],
                [Paragraph("Duplicate Structural Identity Footprint Rows Signatures Counter", custom_td_style), Paragraph(f"{df.duplicated().sum():,}", custom_td_style)]
            ]
            
            meta_table = Table(metadata_table_data, colWidths=[330, 190])
            meta_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E3A8A')),
                ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                ('TOPPADDING', (0,0), (-1,-1), 6),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8FAFC')])
            ]))
            
            flowables_story_stack.append(meta_table)
            flowables_story_stack.append(Spacer(1, 20))
            
            # ==========================================
            # SECTION 3: INLINE DENSITY VISUALIZATION
            # ==========================================
            if len(num_cols) > 0:
                flowables_story_stack.append(Paragraph("II. Primary Continuous Performance Axis Metric Distribution", custom_h2_style))
                
                fig, ax = plt.subplots(figsize=(6.5, 3.0))
                sns.histplot(df[num_cols[0]].dropna(), kde=True, ax=ax, color='#2563EB')
                ax.set_title(f"Primary Quantitative Target Parameter Density Curve Profile: Grid Field `{num_cols[0]}`", fontsize=9, fontweight='bold', color='#0F172A')
                ax.set_xlabel(str(num_cols[0]), fontsize=8)
                ax.set_ylabel("Data Point Observation Volumetric Density Frequency", fontsize=8)
                plt.tight_layout()
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_chart_descriptor:
                    fig.savefig(tmp_chart_descriptor.name, bbox_inches='tight', dpi=180)
                    plt.close(fig)
                    
                    chart_flowable = RLImage(tmp_chart_descriptor.name, width=470, height=215)
                    flowables_story_stack.append(KeepTogether([chart_flowable]))
                    
                flowables_story_stack.append(Spacer(1, 15))
                
            # ==========================================
            # SECTION 4: DETAILED DATAFRAME SAMPLE MATRIX
            # ==========================================
            flowables_story_stack.append(PageBreak())
            flowables_story_stack.append(Paragraph("III. Data Frame Registry Snapshot Ledger Records (First 15 Rows)", custom_h2_style))
            
            sample_rows_to_display = df.head(15).copy()
            columns_subset_limit = sample_rows_to_display.columns.tolist()[:min(df.shape[1], 5)]
            
            # Construct autowrapping column headers text flow patterns
            ledger_table_headers = [Paragraph(f"<b>{col}</b>", custom_th_style) for col in columns_subset_limit]
            ledger_table_data_matrix = [ledger_table_headers]
            
            for idx, row in sample_rows_to_display.iterrows():
                row_cells_collector = []
                for col in columns_subset_limit:
                    cell_value_string = str(row[col])
                    if len(cell_value_string) > 35:
                        cell_value_string = cell_value_string[:32] + "..."
                    row_cells_collector.append(Paragraph(cell_value_string, custom_td_style))
                ledger_table_data_matrix.append(row_cells_collector)
                
            # Calculate even column dimensions allocations boundaries mapping arrays
            allocated_col_width = int(520 / len(columns_subset_limit))
            ledger_table = Table(ledger_table_data_matrix, colWidths=[allocated_col_width] * len(columns_subset_limit))
            ledger_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0F172A')),
                ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                ('BOTTOMPADDING', (0,0), (-1,-1), 5),
                ('TOPPADDING', (0,0), (-1,-1), 5),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F1F5F9')])
            ]))
            
            flowables_story_stack.append(ledger_table)
            
            # Compile pages layout constraints matrices logic loops tracking codes registers
            document_blueprint.build(flowables_story_stack, canvasmaker=NumberedCanvas)
            
        except Exception as e_pdf_compile:
            logger.critical(f"Fatal operational exception building executive hardcopy publication document stream layouts: {str(e_pdf_compile)}")
            logger.critical(traceback.format_exc())
            raise e_pdf_compile
            
        byte_stream_io_buffer.seek(0)
        return byte_stream_io_buffer