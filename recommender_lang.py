
# recommender_lang.py
from __future__ import annotations

import json
from typing import Dict, Any
import pandas as pd

from llm_clients_lang import chat_completion_json

BODY_TYPES = ["hourglass", "pear", "apple", "rectangle", "inverted_triangle"]
SKIN_TONES = ["light", "wheatish", "dark"]
STYLE_TYPES = ["boho", "preppy", "minimalist", "streetwear", "classic", "romantic"]

SYSTEM_STYLIST = "You are a professional stylist, selecting outfits based on the user preferences provided."

OUTPUT_SCHEMA_HINT = """
Return STRICT JSON with this schema (no markdown):
{
  "outfits":[
    {
      "title":"string",
      "why_it_works":"string",
      "items":[
        {"category":"top|bottom|dress|outerwear|shoes|accessory", "name":"string", "sku":"optional string or null", "color":"string"}
      ],
      "styling_tips":["string", "..."],
      "image_prompt":"string (a single prompt for an image generation model)"
    }
  ]
}
"""

def load_catalog(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in ["style_tags", "occasion_tags", "season_tags", "body_type_fit", "skin_tone_palette"]:
        if col in df.columns:
            df[col] = df[col].fillna("")
    return df

def simple_retrieve(df: pd.DataFrame, prefs: Dict[str, Any], k: int = 12) -> pd.DataFrame:
    """
    Lightweight retrieval without embeddings:
    score by overlaps on style/occasion/body_type/skin_tone.
    """
    style = prefs.get("style_type", "")
    occasion = prefs.get("occasion", "")
    body = prefs.get("body_type", "")
    tone = prefs.get("skin_tone", "")

    def score_row(r) -> int:
        s = 0
        if style and style in str(r.get("style_tags", "")):
            s += 3
        if occasion and occasion in str(r.get("occasion_tags", "")):
            s += 3
        if body and (str(r.get("body_type_fit", "")) == "all" or body in str(r.get("body_type_fit", ""))):
            s += 2
        if tone and (str(r.get("skin_tone_palette", "")) == "all" or tone in str(r.get("skin_tone_palette", ""))):
            s += 2
        return s

    scored = df.copy()
    scored["score"] = scored.apply(score_row, axis=1)
    return scored.sort_values("score", ascending=False).head(k)

def build_user_prompt(prefs: Dict[str, Any], retrieved_df: pd.DataFrame) -> str:
    cols = [c for c in ["sku", "name", "category", "color", "style_tags", "occasion_tags", "price_eur"] if c in retrieved_df.columns]
    wardrobe_context = retrieved_df[cols].to_dict(orient="records")

    return f"""
User preferences (fixed dropdowns):
- body_type: {prefs.get('body_type')}
- skin_tone: {prefs.get('skin_tone')}
- style_type: {prefs.get('style_type')}
- occasion: {prefs.get('occasion')}
- budget_eur: {prefs.get('budget_eur')}
- notes: {prefs.get('notes')}

Candidate items from catalog (you can reference sku + name; do NOT invent SKUs):
{json.dumps(wardrobe_context, ensure_ascii=False)}

Rules:
- The dropdown occasion is a HARD constraint.
- Propose 3 outfits (different vibes but consistent with style_type).
- Prefer catalog items; you MAY include 1-2 non-catalog generic items if needed (set sku to null).
- Make sure the silhouette flatters body_type.
- Colors should suit skin_tone.
- Keep within budget if possible.
- Ensure image_prompt is fashion-photo friendly: full-body lookbook, neutral background, no text.

{OUTPUT_SCHEMA_HINT}
""".strip()

def recommend_outfits(prefs: Dict[str, Any], catalog_df: pd.DataFrame) -> Dict[str, Any]:
    retrieved = simple_retrieve(catalog_df, prefs, k=14)
    user_prompt = build_user_prompt(prefs, retrieved)
    temperature = float(prefs.get("temperature", 0.6))
    return chat_completion_json(SYSTEM_STYLIST, user_prompt, temperature=temperature)
