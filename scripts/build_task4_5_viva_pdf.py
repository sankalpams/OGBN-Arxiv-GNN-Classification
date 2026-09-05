"""
Build a clean, simple English Viva Contribution Speech Guide PDF (No math symbols).
Module: CCS4354 - Tensors and Graphs
SLTC Faculty of Computing and Information Technology
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
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
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#4A5568"))
        
        # Header on pages > 1
        if self._pageNumber > 1:
            self.drawString(54, 755, "CCS4354: Tensors and Graphs - Viva Speech Guide (Tasks 4 and 5)")
            self.drawRightString(558, 755, "OGBN-Arxiv Classification")
            self.setStrokeColor(colors.HexColor("#CBD5E0"))
            self.setLineWidth(0.5)
            self.line(54, 748, 558, 748)

        # Footer
        self.setStrokeColor(colors.HexColor("#CBD5E0"))
        self.setLineWidth(0.5)
        self.line(54, 45, 558, 45)
        self.drawString(54, 32, "Sri Lanka Technology Campus (SLTC) - Faculty of Computing and IT")
        self.drawRightString(558, 32, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()


def build_simple_viva_pdf(output_path="report/Task_04_05_Viva_Speech_Guide.pdf"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1A365D"),
        alignment=1,
        spaceAfter=4
    )
    
    sub_style = ParagraphStyle(
        'DocSub',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#4A5568"),
        alignment=1,
        spaceAfter=12
    )

    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1A365D"),
        spaceBefore=10,
        spaceAfter=5
    )

    h2_style = ParagraphStyle(
        'H2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=5,
        spaceAfter=3
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=4
    )

    speech_style = ParagraphStyle(
        'Speech',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9,
        leading=13.5,
        textColor=colors.HexColor("#1A202C")
    )

    qa_q_style = ParagraphStyle(
        'QA_Q',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#C53030")
    )

    qa_a_style = ParagraphStyle(
        'QA_A',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=4
    )

    th_style = ParagraphStyle(
        'TH',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
        alignment=1
    )

    tc_style = ParagraphStyle(
        'TC',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#2D3748")
    )

    tc_bold = ParagraphStyle(
        'TCB',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#1A365D")
    )

    story = []

    # Title
    story.append(Paragraph("CCS4354 - Tensors and Graphs", sub_style))
    story.append(Paragraph("Viva Speech and Defense Guide (Simple English)", title_style))
    story.append(Paragraph("Personal Contribution: Task 4 (GNN Models) and Task 5 (Training Optimization)<br/>Dataset: OGBN-Arxiv (169,343 Papers, 1.16 Million Citations, 40 Categories)", sub_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2B6CB0"), spaceBefore=2, spaceAfter=8))

    # 1. Opening Pitch
    story.append(Paragraph("1. Opening Introduction (Say This First)", h1_style))
    open_speech = (
        "\"Good morning / afternoon professors. In our group project on the OGBN-Arxiv dataset, "
        "my main personal contributions were <b>Task 4</b> and <b>Task 5</b>.<br/><br/>"
        "In <b>Task 4</b>, I built and tested the core Graph Neural Network models: the Graph Convolutional Network (GCN) "
        "and the Graph Attention Network (GAT).<br/>"
        "In <b>Task 5</b>, I performed the training optimization and hyperparameter tuning to stop overfitting and improve our test accuracy.<br/><br/>"
        "I will now explain how I designed the models and how I improved their performance.\""
    )
    story.append(Table([[Paragraph(open_speech, speech_style)]], colWidths=[504], style=[
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#EDF2F7")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E0")),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(Spacer(1, 6))

    # 2. Task 4 Speech
    story.append(KeepTogether([
        Paragraph("2. Task 4 Speech: GNN Model Development (GCN and GAT)", h1_style),
        Paragraph("Notebooks: <b>notebooks/04_gcn_model.ipynb</b> and <b>notebooks/05_gat_model.ipynb</b>", body_style),
        Paragraph("<b>Spoken Speech:</b>", h2_style),
        Table([[Paragraph(
            "\"In Task 4, I designed and trained our baseline Graph Neural Network models using PyTorch Geometric.<br/><br/>"
            "First, I built a <b>2-layer Graph Convolutional Network (GCN)</b>:<br/>"
            "• <b>Why 2 Layers:</b> I selected a 2-layer design because it allows each paper to learn from its 2-hop neighbor papers (papers cited by the papers it cites). If we use too many layers, all paper representations become identical, which is called over-smoothing.<br/>"
            "• <b>Dimensions:</b> The network takes 128 input features, passes them through 256 hidden channels, and outputs predictions for the 40 research categories.<br/>"
            "• <b>Self-Loops and Degree Normalization:</b> I added self-loops so that each paper keeps its own text information while collecting citations. Degree normalization makes sure that popular papers with thousands of citations do not overpower smaller papers.<br/>"
            "• <b>Activations and Loss:</b> I used ReLU activation for non-linearity, Dropout of 0.5 to prevent memorization, and Cross-Entropy loss for classification.<br/><br/>"
            "Second, I implemented a <b>Graph Attention Network (GAT)</b> with 4 attention heads. Instead of treating all citations equally, GAT learns attention weights to give more importance to the most relevant cited papers.<br/><br/>"
            "Our baseline GCN quickly reached about <b>65.2% validation accuracy</b>, proving that using graph citations works much better than regular machine learning.\"",
            speech_style
        )]], colWidths=[504], style=[
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F7FAFC")),
            ('BOX', (0,0), (-1,-1), 0.75, colors.HexColor("#CBD5E0")),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ]),
        Spacer(1, 6)
    ]))

    # 3. Task 5 Speech
    story.append(KeepTogether([
        Paragraph("3. Task 5 Speech: Training Optimization and Tuning", h1_style),
        Paragraph("Notebook: <b>notebooks/06_training_optimization.ipynb</b>", body_style),
        Paragraph("<b>Spoken Speech:</b>", h2_style),
        Table([[Paragraph(
            "\"In Task 5, my goal was to optimize the training process so the model could generalize well to future papers published in 2019 and 2020.<br/><br/>"
            "I conducted a systematic hyperparameter grid search where I tested different configurations:<br/>"
            "1. <b>Hidden Layer Size:</b> Increasing hidden units from 128 to 256 gave us a 1.1% boost in accuracy. It gave the model enough capacity to separate similar categories like Computer Vision and Machine Learning.<br/>"
            "2. <b>Dropout Regularization:</b> Using a Dropout rate of 0.5 successfully stopped the model from overfitting on training papers.<br/>"
            "3. <b>Optimizer:</b> I compared standard Adam with AdamW. AdamW applies weight decay directly to the weights, which gave smoother loss curves and better final accuracy.<br/>"
            "4. <b>Early Stopping:</b> I added early stopping with a patience of 10 epochs to monitor validation loss. This stopped training at the exact moment before validation loss started to increase.<br/><br/>"
            "Thanks to these optimizations, our tuned GCN accuracy jumped from <b>65.2% to 69.62%</b>, and our GAT model achieved <b>70.80%</b>.\"",
            speech_style
        )]], colWidths=[504], style=[
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F7FAFC")),
            ('BOX', (0,0), (-1,-1), 0.75, colors.HexColor("#CBD5E0")),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ]),
        Spacer(1, 6)
    ]))

    story.append(PageBreak())

    # 4. Top 5 Questions
    story.append(Paragraph("4. Top 5 Examiner Questions and Simple Answers", h1_style))
    
    qa_list = [
        (
            "Q1: What is over-smoothing in Graph Neural Networks?",
            "A: Over-smoothing happens when you stack too many graph convolution layers. With every layer, neighbor features are mixed together. If you use 4 or 5 layers, all papers end up with almost the exact same features and the model cannot tell them apart. I prevented this by using only 2 layers and a 0.5 dropout rate."
        ),
        (
            "Q2: Why do we add self-loops to the citation graph?",
            "A: Without self-loops, a paper would only gather features from the papers it cites, completely ignoring its own title and abstract text. Self-loops ensure that a paper keeps its own text embedding while mixing in citations."
        ),
        (
            "Q3: What is the main difference between GCN and GAT?",
            "A: GCN treats all neighbor papers with fixed, equal mathematical weights based on degree. GAT uses an attention mechanism to learn which cited paper is actually the most important, giving higher weights to key reference papers."
        ),
        (
            "Q4: Why did AdamW perform better than standard Adam?",
            "A: AdamW applies weight decay directly to the weights rather than mixing it with gradient momentum. This gives real regularization, prevents large weight explosions, and leads to better generalization on unseen test papers."
        ),
        (
            "Q5: Why is the dataset split by publication year rather than randomly?",
            "A: Random splitting causes data leakage because future citations would be mixed into past papers. Splitting by year tests if the model can learn from historical papers (up to 2017) and accurately predict future research papers (2019 to 2020)."
        )
    ]

    for q, a in qa_list:
        story.append(KeepTogether([
            Paragraph(f"<b>{q}</b>", qa_q_style),
            Paragraph(a, qa_a_style),
            Spacer(1, 3)
        ]))

    story.append(Spacer(1, 4))

    # 5. Summary Table
    story.append(Paragraph("5. Model and Tuning Summary Table", h1_style))
    table_data = [
        [Paragraph("<b>Configuration</b>", th_style), 
         Paragraph("<b>Baseline GCN (Task 4)</b>", th_style), 
         Paragraph("<b>GAT Model (Task 4)</b>", th_style), 
         Paragraph("<b>Optimized GCN (Task 5)</b>", th_style)],
        [Paragraph("Input Features", tc_bold), Paragraph("128 dimensions", tc_style), Paragraph("128 dimensions", tc_style), Paragraph("128 dimensions", tc_style)],
        [Paragraph("Hidden Channels", tc_bold), Paragraph("128", tc_style), Paragraph("64 x 4 heads (256)", tc_style), Paragraph("<b>256 channels</b>", tc_bold)],
        [Paragraph("Output Classes", tc_bold), Paragraph("40 categories", tc_style), Paragraph("40 categories", tc_style), Paragraph("40 categories", tc_style)],
        [Paragraph("Number of Layers", tc_bold), Paragraph("2 layers", tc_style), Paragraph("2 layers", tc_style), Paragraph("2 layers", tc_style)],
        [Paragraph("Attention Type", tc_bold), Paragraph("Fixed degree weights", tc_style), Paragraph("4 attention heads", tc_style), Paragraph("Fixed degree weights", tc_style)],
        [Paragraph("Dropout Rate", tc_bold), Paragraph("0.5", tc_style), Paragraph("0.5", tc_style), Paragraph("<b>0.5</b>", tc_bold)],
        [Paragraph("Optimizer", tc_bold), Paragraph("Adam (lr=0.01)", tc_style), Paragraph("Adam (lr=0.005)", tc_style), Paragraph("<b>AdamW (lr=0.005)</b>", tc_bold)],
        [Paragraph("Validation Accuracy", tc_bold), Paragraph("65.20%", tc_style), Paragraph("68.40%", tc_style), Paragraph("<b>69.62%</b>", tc_bold)],
        [Paragraph("Test Accuracy", tc_bold), Paragraph("64.80%", tc_style), Paragraph("70.80%", tc_style), Paragraph("<b>70.20%</b>", tc_bold)],
    ]
    summary_table = Table(table_data, colWidths=[120, 124, 130, 130])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1A365D")),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F7FAFC")]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 6))

    # 6. Quick Rules
    story.append(KeepTogether([
        Paragraph("6. Simple Tips for the Presentation", h1_style),
        Paragraph(
            "• <b>Speak clearly and simply:</b> Avoid confusing formulas; explain what the layers and optimizers actually do.<br/>"
            "• <b>Remember the key numbers:</b> 169,343 papers, 1.16 million citations, 40 categories, 128 features, 2 layers, and ~70% accuracy.<br/>"
            "• <b>Refer to your notebooks:</b> Mention that your code and plots are in <code>04_gcn_model.ipynb</code>, <code>05_gat_model.ipynb</code>, and <code>06_training_optimization.ipynb</code>.",
            body_style
        )
    ]))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[SUCCESS] Simple English Viva PDF generated at: {output_path}")

if __name__ == "__main__":
    build_simple_viva_pdf()
