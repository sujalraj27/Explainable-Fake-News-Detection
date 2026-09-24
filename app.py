import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# Backend API
API_URL = "http://127.0.0.1:8000/prediction/predict"

# Page Configuration
st.set_page_config(
    page_title="Explainable Fake News Detection",
    page_icon="📰",
    layout="wide"
)

st.title("📰 Explainable Fake News Detection")
st.write("Enter a news article and detect whether it is REAL or FAKE.")

# News Input
news_text = st.text_area(
    "Enter News Article Text",
    height=220
)

# Analyze Button
if st.button("🔍 Analyze News"):

    if news_text.strip() == "":
        st.warning("Please enter a news article.")
    else:
        payload = {
            "news_text": news_text
        }

        try:

            response = requests.post(API_URL, json=payload)

            if response.status_code == 200:

                result = response.json()

                prediction = result.get("prediction", "Unknown")
                confidence = result.get("confidence", "0%")
                probabilities = result.get("probabilities", {})
                lime_data = result.get("lime_explanation", [])
                graph_path = result.get("knowledge_graph")

                st.divider()

                st.subheader("Prediction Result")

                col1, col2 = st.columns(2)

                with col1:

                    if prediction.upper() == "REAL":
                        st.success("✅ REAL NEWS")
                    else:
                        st.error("❌ FAKE NEWS")

                with col2:
                    st.metric(
                        "Confidence",
                        confidence
                    )

                st.divider()

                st.subheader("Class Probability")

                if probabilities:

                    df = pd.DataFrame(
                        {
                            "Class": list(probabilities.keys()),
                            "Probability": list(probabilities.values())
                        }
                    )

                    st.bar_chart(
                        df.set_index("Class")
                    )

                st.divider()

                
                st.subheader("🧠 LIME Explanation")

                if lime_data:
                    lime_df = pd.DataFrame(lime_data)

                    st.dataframe(lime_df, use_container_width=True)

                    fig, ax = plt.subplots(figsize=(8,5))

                    colors = [
                        "green" if w >= 0 else "red"
                        for w in lime_df["weight"]
                    ]

                    ax.barh(
                        lime_df["word"],
                        lime_df["weight"],
                        color=colors
                    )

                    ax.set_xlabel("Contribution Weight")
                    ax.set_ylabel("Words")

                    st.pyplot(fig)
                else:
                    st.info("No LIME explanation available.")

                st.divider()

                st.subheader("🕸️ Knowledge Graph")

                if graph_path:

                    try:

                        with open(graph_path, "r", encoding="utf-8") as f:

                            html = f.read()

                        components.html(
                            html,
                            height=650,
                            scrolling=True
                        )

                    except Exception as e:

                        st.error(f"Unable to load Knowledge Graph: {e}")

                else:

                    st.info("Knowledge Graph not available.")
            else:
                st.divider()
                st.error(f"Backend Error : {response.status_code}")

                try:
                    st.json(response.json())
                except:
                    st.write(response.text)

            # AI Explanation (if available)
            if response.status_code == 200:
                st.divider()
                st.subheader("🤖 AI Explanation")

                ai_explanation = result.get("ai_explanation", "")

                if ai_explanation:
                    st.success(ai_explanation)
                else:
                    st.info("No AI explanation available.")
                                    # -------------------------------
                # PDF Download
                # -------------------------------

                pdf_path = result.get("pdf_report")

                if pdf_path:

                    try:

                        with open(pdf_path, "rb") as pdf_file:

                            st.download_button(
                                label="📄 Download Prediction Report",
                                data=pdf_file,
                                file_name="prediction_report.pdf",
                                mime="application/pdf"
                            )

                    except Exception as e:

                        st.error(f"Unable to load PDF: {e}")

                else:

                    st.info("PDF Report not available.")

        except Exception as e:

            st.error(f"Connection Error : {e}")