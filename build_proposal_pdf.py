from pathlib import Path
import os
from reportlab.lib.colors import HexColor, white, black, lightgrey, grey
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedProposalCanvas(canvas.Canvas):
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

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(HexColor("#555555"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(2*cm, A4[1] - 1.2*cm, "CCS4354 Project Proposal | Deep GNNs for Academic Paper Classification (OGBN-Arxiv)")
            self.setStrokeColor(HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(2*cm, A4[1] - 1.3*cm, A4[0] - 2*cm, A4[1] - 1.3*cm)

        # Footer (all pages)
        self.setStrokeColor(HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(2*cm, 1.5*cm, A4[0] - 2*cm, 1.5*cm)
        self.drawString(2*cm, 1.1*cm, "Project Proposal — Department of Computer Science & Engineering | CCS4354")
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(A4[0] - 2*cm, 1.1*cm, page_text)
        self.restoreState()


def build_proposal_pdf():
    out = Path('report/Project_Proposal.pdf')
    out.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    
    # Custom color palette
    c_primary = HexColor("#0F2942")     # Deep Navy
    c_secondary = HexColor("#1E5F74")   # Steel Blue
    c_accent = HexColor("#28527A")      # Blue
    c_dark = HexColor("#1F2937")        # Charcoal body
    c_muted = HexColor("#4B5563")       # Muted text
    c_bg_light = HexColor("#F8FAFC")    # Table alt bg
    c_card_bg = HexColor("#F1F5F9")     # Card background

    # Typography styles
    styles.add(ParagraphStyle(
        name='PropTitle',
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=25,
        textColor=c_primary,
        alignment=1, # Center
        spaceAfter=6
    ))
    styles.add(ParagraphStyle(
        name='PropSubtitle',
        fontName='Helvetica',
        fontSize=11.5,
        leading=15,
        textColor=c_secondary,
        alignment=1,
        spaceAfter=12
    ))
    styles.add(ParagraphStyle(
        name='MetaBox',
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=c_dark,
        alignment=1
    ))
    styles.add(ParagraphStyle(
        name='H1_Custom',
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=c_primary,
        spaceBefore=12,
        spaceAfter=5,
        keepWithNext=True
    ))
    styles.add(ParagraphStyle(
        name='H2_Custom',
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=c_secondary,
        spaceBefore=8,
        spaceAfter=3,
        keepWithNext=True
    ))
    styles.add(ParagraphStyle(
        name='Body_Custom',
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=c_dark,
        spaceAfter=5
    ))
    styles.add(ParagraphStyle(
        name='CalloutBox',
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=12,
        textColor=HexColor("#1E293B"),
        backColor=HexColor("#E2E8F0"),
        borderPadding=7,
        spaceAfter=7
    ))
    styles.add(ParagraphStyle(
        name='TableText',
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=c_dark
    ))
    styles.add(ParagraphStyle(
        name='TableHeader',
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=white,
        alignment=1
    ))
    styles.add(ParagraphStyle(
        name='Caption',
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=10,
        textColor=c_muted,
        alignment=1,
        spaceBefore=3,
        spaceAfter=6
    ))

    story = []

    # Title & Metadata
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("CCS4354 — TENSORS AND GRAPHS (WITH PROGRAMMING)", ParagraphStyle('SuperTitle', fontName='Helvetica-Bold', fontSize=9.5, textColor=c_secondary, alignment=1, spaceAfter=3)))
    story.append(Paragraph("Project Proposal: Deep Graph Neural Networks for Academic Paper Classification on the OGBN-Arxiv Benchmark", styles['PropTitle']))
    story.append(Paragraph("Formal Technical & Experimental Proposal", styles['PropSubtitle']))

    meta_table_data = [
        [
            Paragraph("<b>Course Module:</b> CCS4354 Tensors & Graphs", styles['MetaBox']),
            Paragraph("<b>Target Dataset:</b> Open Graph Benchmark (ogbn-arxiv)", styles['MetaBox']),
            Paragraph("<b>Evaluation Metric:</b> Test Accuracy, Weighted F1", styles['MetaBox'])
        ],
        [
            Paragraph("<b>Proposed Models:</b> GCN (256 hidden) vs. GAT (4-head)", styles['MetaBox']),
            Paragraph("<b>Key Focus:</b> Inductive Temporal Generalization", styles['MetaBox']),
            Paragraph("<b>Status:</b> Project Proposal Document", styles['MetaBox'])
        ]
    ]
    meta_table = Table(meta_table_data, colWidths=[5.6*cm, 5.8*cm, 5.6*cm])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_card_bg),
        ('BOX', (0,0), (-1,-1), 0.5, HexColor("#CBD5E1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, HexColor("#E2E8F0")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 0.3*cm))

    # Executive Summary Box
    exec_summary_text = (
        "<b>Proposal Summary:</b> Automated classification of academic literature is a vital challenge in modern information retrieval. "
        "Unlike Euclidean text data, academic citation networks encode crucial relational homophily across non-Euclidean citation graphs. "
        "This project proposes an end-to-end investigation into Graph Representation Learning on the <b>OGBN-Arxiv</b> benchmark "
        "(169,343 papers, 2,315,598 citations, 40 categories). We propose to implement, train, tune, evaluate, and interpret two core GNN "
        "architectures: <b>Graph Convolutional Networks (GCN)</b> and <b>Graph Attention Networks (GAT)</b> under strict out-of-distribution "
        "<b>temporal splitting</b> (train ≤2017, validation 2018, test 2019–2020), complemented by latent embedding explainability (PCA/t-SNE) "
        "and an interactive <b>Streamlit dashboard</b>."
    )
    story.append(Paragraph(exec_summary_text, styles['CalloutBox']))

    # Section 1: Problem Statement
    story.append(Paragraph("1. Problem Statement & Research Motivation", styles['H1_Custom']))
    story.append(HRFlowable(width="100%", thickness=1, color=c_secondary, spaceBefore=1, spaceAfter=5))
    
    p1 = (
        "<b>Relational Dependencies in Scientific Literature:</b> Standard text classification models treat documents as independent "
        "and identically distributed (i.i.d.), ignoring citation networks. A research paper's domain is strongly signaled by the papers it cites "
        "and the works that cite it. Graph Representation Learning leverages tensor-based message passing to propagate semantic context "
        "across multi-hop citation neighborhoods."
    )
    story.append(Paragraph(p1, styles['Body_Custom']))

    p2 = (
        "<b>Inductive Temporal Distribution Drift:</b> In real-world deployment, models trained on historical papers must predict future research "
        "categories. Standard random splitting creates severe temporal data leakage. We propose to strictly adopt OGB's chronological split: "
        "<b>Training (≤2017: 90,941 nodes, 53.7%)</b>, <b>Validation (2018: 29,799 nodes, 17.6%)</b>, and <b>Test (2019–2020: 48,603 nodes, 28.7%)</b>."
    )
    story.append(Paragraph(p2, styles['Body_Custom']))

    # Section 2: Objectives
    story.append(Paragraph("2. Project Objectives & Research Questions", styles['H1_Custom']))
    story.append(HRFlowable(width="100%", thickness=1, color=c_secondary, spaceBefore=1, spaceAfter=5))

    p_obj = (
        "• <b>Objective 1 (Data & Topology):</b> Ingest 169K nodes and 2.3M edges, compute degree distributions, density, and local subgraphs.<br/>"
        "• <b>Objective 2 (GNN Modeling):</b> Implement 2-layer GCN (256 hidden) and 2-layer Multi-Head GAT (4 heads × 64 channels).<br/>"
        "• <b>Objective 3 (Optimization):</b> Perform controlled hyperparameter grid searches across hidden dimensions, dropout, and learning rates.<br/>"
        "• <b>Objective 4 (Benchmarking):</b> Evaluate Test Accuracy, Weighted Precision, Recall, F1-scores, and 40-class confusion matrices.<br/>"
        "• <b>Objective 5 (Explainability):</b> Extract hidden embeddings and compute 2D linear PCA and non-linear t-SNE projections with neighborhood homophily agreement scoring.<br/>"
        "• <b>Objective 6 (Interactive Deployment):</b> Engineer a 5-module Streamlit web dashboard for real-time model querying."
    )
    story.append(Paragraph(p_obj, styles['Body_Custom']))

    story.append(PageBreak())

    # Section 3: Technical Methodology
    story.append(Paragraph("3. Technical Methodology & Proposed Architectures", styles['H1_Custom']))
    story.append(HRFlowable(width="100%", thickness=1, color=c_secondary, spaceBefore=1, spaceAfter=5))

    story.append(Paragraph("3.1 Spectral Graph Convolutional Network (GCN)", styles['H2_Custom']))
    p_gcn_eq = (
        "The spectral graph convolution propagates messages via symmetric normalized adjacency:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>H^(l+1) = σ( D̃^(-1/2) Ã D̃^(-1/2) H^(l) W^(l) )</b><br/>"
        "where Ã = A + I_N, D̃_ii = ∑_j Ã_ij, and W^(l) is the layer weight tensor. "
        "Layer 1 transforms 128-dim features to 256 hidden channels (ReLU, Dropout=0.5), and Layer 2 outputs 40 class logits."
    )
    story.append(Paragraph(p_gcn_eq, styles['Body_Custom']))

    story.append(Paragraph("3.2 Spatial Graph Attention Network (GAT)", styles['H2_Custom']))
    p_gat_eq = (
        "The spatial attention network learns anisotropic edge importance weights:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>α_ij = exp(LeakyReLU(a^T [W h_i || W h_j])) / ∑_(k∈N_i) exp(LeakyReLU(a^T [W h_i || W h_k]))</b><br/>"
        "Multi-head attention with K = 4 heads computes: <b>h_i^(l+1) = ||_(k=1..K) ELU( ∑_(j∈N_i) α_ij^k W^k h_j^(l) )</b>."
    )
    story.append(Paragraph(p_gat_eq, styles['Body_Custom']))

    # Section 4: Dataset Profile
    story.append(Paragraph("4. Benchmark Dataset Specifications (OGBN-Arxiv)", styles['H1_Custom']))
    story.append(HRFlowable(width="100%", thickness=1, color=c_secondary, spaceBefore=1, spaceAfter=5))

    data_table_data = [
        [Paragraph("Feature / Attribute", styles['TableHeader']), Paragraph("Value / Dimension", styles['TableHeader']), Paragraph("Description & Analytical Role", styles['TableHeader'])],
        [Paragraph("Total Nodes (|V|)", styles['TableText']), Paragraph("169,343", styles['TableText']), Paragraph("Academic computer science papers in corpus", styles['TableText'])],
        [Paragraph("Total Edges (|E|)", styles['TableText']), Paragraph("2,315,598", styles['TableText']), Paragraph("Directed citation links between academic papers", styles['TableText'])],
        [Paragraph("Node Features", styles['TableText']), Paragraph("128-dim", styles['TableText']), Paragraph("Word2Vec skip-gram averaged title/abstract embeddings", styles['TableText'])],
        [Paragraph("Target Classes", styles['TableText']), Paragraph("40 Categories", styles['TableText']), Paragraph("Fine-grained arXiv CS primary categories (cs.CV, cs.LG, etc.)", styles['TableText'])],
        [Paragraph("Graph Density", styles['TableText']), Paragraph("1.615 × 10^-4", styles['TableText']), Paragraph("Sparse scale-free network suited for sparse tensor convolutions", styles['TableText'])],
        [Paragraph("Chronological Split", styles['TableText']), Paragraph("53.7% / 17.6% / 28.7%", styles['TableText']), Paragraph("Train (≤2017: 90.9K), Valid (2018: 29.8K), Test (2019–20: 48.6K)", styles['TableText'])],
    ]
    t_data = Table(data_table_data, colWidths=[4.0*cm, 3.8*cm, 9.2*cm])
    t_data.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [white, c_bg_light]),
        ('GRID', (0,0), (-1,-1), 0.5, HexColor("#CBD5E1")),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_data)
    story.append(Paragraph("Table 1: Technical specifications of the target OGBN-Arxiv benchmark dataset.", styles['Caption']))

    # Section 5: Work Plan & Milestones
    story.append(Paragraph("5. Project Execution Work Plan & Milestones", styles['H1_Custom']))
    story.append(HRFlowable(width="100%", thickness=1, color=c_secondary, spaceBefore=1, spaceAfter=5))

    plan_table_data = [
        [Paragraph("Phase", styles['TableHeader']), Paragraph("Focus Area / Milestone", styles['TableHeader']), Paragraph("Key Output Deliverable", styles['TableHeader'])],
        [Paragraph("Phase 1", styles['TableText']), Paragraph("Tensor Algebra & GPU Acceleration", styles['TableText']), Paragraph("Notebook 01: Tensor operations, broadcasting, memory profile", styles['TableText'])],
        [Paragraph("Phase 2", styles['TableText']), Paragraph("Graph Topology & Temporal Splitting", styles['TableText']), Paragraph("Notebooks 02–03: Degree plots, density, processed data tensors", styles['TableText'])],
        [Paragraph("Phase 3", styles['TableText']), Paragraph("GNN Architecture Implementation", styles['TableText']), Paragraph("Notebooks 04–05: GCN/GAT checkpoints (models/*.pt) & loss history", styles['TableText'])],
        [Paragraph("Phase 4", styles['TableText']), Paragraph("Hyperparameter Optimization", styles['TableText']), Paragraph("Notebook 06: Grid search tuning table (results/training/)", styles['TableText'])],
        [Paragraph("Phase 5", styles['TableText']), Paragraph("Model Benchmarking & Evaluation", styles['TableText']), Paragraph("Notebook 07: Test accuracy/F1 table & 40-class confusion matrices", styles['TableText'])],
        [Paragraph("Phase 6", styles['TableText']), Paragraph("Explainability & Latent Projections", styles['TableText']), Paragraph("Notebook 08: 2D PCA & t-SNE embedding figures, homophily score", styles['TableText'])],
        [Paragraph("Phase 7", styles['TableText']), Paragraph("Interactive Dashboard Engineering", styles['TableText']), Paragraph("dashboard/app.py: Live Streamlit web application on port 8501", styles['TableText'])],
        [Paragraph("Phase 8", styles['TableText']), Paragraph("Reporting & GitHub Repository Setup", styles['TableText']), Paragraph("Technical Report PDF/MD, Proposal Document, GitHub README", styles['TableText'])],
    ]
    t_plan = Table(plan_table_data, colWidths=[2.2*cm, 6.4*cm, 8.4*cm])
    t_plan.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_secondary),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [white, c_bg_light]),
        ('GRID', (0,0), (-1,-1), 0.5, HexColor("#CBD5E1")),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t_plan)
    story.append(Paragraph("Table 2: Proposed 8-phase execution roadmap and deliverables.", styles['Caption']))

    # Section 6: Risk Management & References
    story.append(Paragraph("6. Risk Assessment & Key References", styles['H1_Custom']))
    story.append(HRFlowable(width="100%", thickness=1, color=c_secondary, spaceBefore=1, spaceAfter=5))

    p_risk = (
        "<b>Identified Risks & Mitigations:</b> (1) <i>Memory Bottlenecks:</i> Addressed by utilizing sparse COO edge representations (edge_index). "
        "(2) <i>Oversmoothing:</i> Mitigated by maintaining a 2-layer depth with 50% dropout regularization. "
        "(3) <i>Temporal Data Leakage:</i> Prevented by strictly honoring OGB chronological partitioning.<br/>"
        "<b>Key References:</b> [1] Kipf & Welling, ICLR 2017. [2] Veličković et al., ICLR 2018. [3] Hu et al., NeurIPS 2020. [4] Hamilton et al., NeurIPS 2017."
    )
    story.append(Paragraph(p_risk, styles['Body_Custom']))

    # Build document
    doc = SimpleDocTemplate(
        str(out.resolve()),
        pagesize=A4,
        rightMargin=1.6*cm,
        leftMargin=1.6*cm,
        topMargin=1.6*cm,
        bottomMargin=1.6*cm
    )
    doc.build(story, canvasmaker=NumberedProposalCanvas)
    print(f"Successfully compiled project proposal PDF to {out.resolve()} ({os.path.getsize(out)} bytes)")

if __name__ == '__main__':
    build_proposal_pdf()
