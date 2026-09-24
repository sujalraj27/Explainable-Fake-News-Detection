from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    PageBreak
)

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch

from pathlib import Path
from datetime import datetime


BASE_DIR = Path(__file__).resolve().parent.parent



def add_image_if_exists(story, image_path, title, styles):

    if image_path:

        image_path = Path(image_path)

        if image_path.exists():

            story.append(
                Paragraph(
                    title,
                    styles["Heading2"]
                )
            )

            img = Image(
                str(image_path),
                width=5*inch,
                height=3*inch
            )

            story.append(img)

            story.append(
                Spacer(1,20)
            )



def generate_pdf_report(news_text, result):


    output_dir = BASE_DIR / "results"
    output_dir.mkdir(exist_ok=True)


    pdf_path = output_dir / "prediction_report.pdf"


    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter
    )


    styles = getSampleStyleSheet()


    story = []



    # -----------------------------
    # TITLE
    # -----------------------------

    story.append(
        Paragraph(
            "<b>Explainable Fake News Detection using LLM and Knowledge Graph</b>",
            styles["Title"]
        )
    )


    story.append(
        Spacer(1,20)
    )


    story.append(
        Paragraph(
            f"<b>Generated Date:</b> {datetime.now().strftime('%d-%m-%Y %H:%M')}",
            styles["Normal"]
        )
    )


    story.append(
        Spacer(1,20)
    )



    # -----------------------------
    # NEWS CONTENT
    # -----------------------------

    story.append(
        Paragraph(
            "1. News Content",
            styles["Heading2"]
        )
    )


    story.append(
        Paragraph(
            news_text,
            styles["BodyText"]
        )
    )


    story.append(
        Spacer(1,20)
    )



    # -----------------------------
    # PREDICTION SUMMARY TABLE
    # -----------------------------


    story.append(
        Paragraph(
            "2. Model Prediction",
            styles["Heading2"]
        )
    )


    prediction_data = [

        ["Prediction",
         result.get("prediction","N/A")],

        ["Confidence",
         str(result.get("confidence","N/A"))+"%"],

        ["Model",
         "TF-IDF + Logistic Regression"]

    ]


    table = Table(prediction_data)


    table.setStyle(
        TableStyle(
            [
                ("GRID",(0,0),(-1,-1),0.5,None),
                ("VALIGN",(0,0),(-1,-1),"TOP")
            ]
        )
    )


    story.append(table)


    story.append(
        Spacer(1,20)
    )



    # -----------------------------
    # PROBABILITY CHART
    # -----------------------------

    add_image_if_exists(
        story,
        result.get("probability_chart"),
        "3. Probability Distribution",
        styles
    )



    # -----------------------------
    # LIME GRAPH
    # -----------------------------


    add_image_if_exists(
        story,
        result.get("lime_graph"),
        "4. LIME Explainability Graph",
        styles
    )



    # -----------------------------
    # LIME WORD IMPORTANCE
    # -----------------------------


    story.append(
        Paragraph(
            "5. LIME Feature Importance",
            styles["Heading2"]
        )
    )


    lime_data = result.get(
        "lime_explanation",
        []
    )


    for item in lime_data:


        story.append(
            Paragraph(
                f"""
                <b>{item.get('word')}</b>
                : {item.get('weight')}
                """,
                styles["BodyText"]
            )
        )



    story.append(
        Spacer(1,20)
    )



    # -----------------------------
    # KNOWLEDGE GRAPH
    # -----------------------------


    add_image_if_exists(
        story,
        result.get("knowledge_graph"),
        "6. Knowledge Graph Visualization",
        styles
    )



    # -----------------------------
    # AI EXPLANATION
    # -----------------------------


    story.append(
        Paragraph(
            "7. AI Generated Explanation",
            styles["Heading2"]
        )
    )


    story.append(
        Paragraph(
            result.get(
                "ai_explanation",
                "No explanation generated"
            ),
            styles["BodyText"]
        )
    )



    story.append(
        Spacer(1,30)
    )



    # -----------------------------
    # CONCLUSION
    # -----------------------------


    story.append(
        Paragraph(
            """
            <b>Conclusion:</b><br/>
            This report combines Machine Learning prediction,
            LIME explainability, Large Language Model reasoning,
            and Knowledge Graph based evidence analysis.
            """,
            styles["BodyText"]
        )
    )


    doc.build(story)


    return str(pdf_path)