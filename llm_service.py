import os
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Optional: Google Gemini Setup
try:
    import google.generativeai as genai
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
except ImportError:
    genai = None


def generate_explanation(
    prediction: str, 
    confidence: float, 
    lime_output: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Generates a human-readable explanation using LIME word weights.
    This function matches your prediction route pipeline.
    """
    positive_words = []
    negative_words = []

    # Safe percentage formatting (handles both 0.95 and 95.0)
    formatted_confidence = f"{confidence * 100:.1f}" if confidence <= 1.0 else f"{confidence:.1f}"

    if lime_output:
        for item in lime_output:
            word = item.get("word", "").strip()
            weight = item.get("weight", 0)
            if not word:
                continue
            if weight >= 0:
                positive_words.append(word)
            else:
                negative_words.append(word)

    # 1. REAL Article Explanation
    if prediction.upper() == "REAL":
        top_words = positive_words[:5]
        words_str = f"'{', '.join(top_words)}'" if top_words else "factual linguistic structures"
        
        return (
            f"The article is classified as REAL with {formatted_confidence}% confidence. "
            f"The prediction is mainly influenced by terms such as {words_str}. "
            f"The language appears factual and informative."
        )

    # 2. FAKE Article Explanation
    else:
        top_words = negative_words[:5] if negative_words else positive_words[:5]
        words_str = f"'{', '.join(top_words)}'" if top_words else "sensational language cues"

        return (
            f"The article is classified as FAKE with {formatted_confidence}% confidence. "
            f"The terms {words_str} contributed significantly towards the fake prediction. "
            f"The article may contain misleading or sensational content."
        )


def generate_ai_explanation(
    text: str, 
    label: str, 
    confidence: float, 
    lime_output: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Primary wrapper imported by routes.prediction.
    Handles LIME, Gemini AI API, and Rule-based fallbacks seamlessly.
    """
    # Priority 1: LIME output if available
    if lime_output:
        return generate_explanation(label, confidence, lime_output)

    # Priority 2: Gemini API if key is configured
    if genai and os.getenv("GEMINI_API_KEY"):
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = f"""
            Act as an expert fact-checker. 
            A news text was classified as '{label}' with {confidence * 100 if confidence <= 1.0 else confidence:.1f}% confidence.
            
            Article Text: "{text}"
            
            Provide a clear 2-sentence explanation of why this text was classified as {label}.
            """
            response = model.generate_content(prompt)
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            logger.warning(f"Gemini API explanation failed: {str(e)}")

    # Priority 3: Fallback rule-based logic
    return generate_explanation(label, confidence, lime_output=[])