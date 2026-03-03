# streamlit_app_lang.py
import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# ✅ Always load .env from same directory as this file
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)

import streamlit as st
import pandas as pd

from recommender_lang import BODY_TYPES, SKIN_TONES, STYLE_TYPES, load_catalog, recommend_outfits
from llm_clients_lang import generate_image_bytes

# --------- Page config ----------
st.set_page_config(
    page_title="Attiria – GenAI Stylist Demo (LangChain)",
    page_icon="👗",
    layout="wide",
)

# --------- Brand styling ----------
BG = "#FAF1E8"
PRIMARY = "#D00F47"
ACCENT = "#AC2156"
TEXT = "#1F1A1C"  # black-ish

st.markdown(
    f"""
    <style>
      /* App background + base text */
      .stApp {{
        background-color: {BG};
        color: {TEXT};
      }}

      /* ---------- FIX: make label text readable (black) ---------- */
      /* Streamlit widget labels */
      div[data-testid="stWidgetLabel"] > label {{
        color: {TEXT} !important;
        font-weight: 650 !important;
      }}

      /* Headings */
      h1, h2, h3, h4, h5, h6 {{
        color: {TEXT} !important;
      }}

      /* Regular text */
      p, span, div {{
        color: {TEXT};
      }}

      /* Captions / help text slightly muted but still visible */
      .attiria-muted {{
        color: rgba(31,26,28,0.68) !important;
      }}

      /* ---------- Cards ---------- */
      .attiria-card {{
        padding: 18px 18px;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.78);
        border: 1px solid rgba(0,0,0,0.06);
        box-shadow: 0 10px 30px rgba(0,0,0,0.04);
      }}

      /* ---------- Sidebar ---------- */
      section[data-testid="stSidebar"] {{
        background-color: rgba(255,255,255,0.65);
        border-right: 1px solid rgba(0,0,0,0.06);
      }}
      section[data-testid="stSidebar"] * {{
        color: {TEXT} !important;
      }}

      /* ---------- Buttons ---------- */
      .stButton > button {{
        background-color: {PRIMARY};
        color: white !important;
        border: 0;
        border-radius: 14px;
        padding: 0.7rem 1rem;
        font-weight: 800;
      }}
      .stButton > button:hover {{
        background-color: {ACCENT};
        color: white !important;
      }}

      /* ---------- Inputs (keep modern rounded look) ---------- */
      div[data-baseweb="select"] > div {{
        border-radius: 14px !important;
      }}
      .stNumberInput input {{
        border-radius: 14px !important;
      }}
      textarea {{
        border-radius: 14px !important;
      }}

      /* Make text inside dark inputs readable */
      input, textarea {{
        color: #ffffff !important;
      }}

      /* Remove Streamlit chrome */
      #MainMenu {{visibility: hidden;}}
      footer {{visibility: hidden;}}
      header {{visibility: hidden;}}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------- Sidebar: intro + provider ----------
with st.sidebar:
    st.image(os.path.join(ASSETS_DIR, "attiria_logo.jpeg"), width=140)

    st.markdown(
        """
        <div class="attiria-card">
          <div style="font-size: 16px; font-weight: 900; margin-bottom: 6px;">Welcome to Attiria 👗</div>
          <div class="attiria-muted" style="font-size: 13.5px; line-height: 1.45;">
            Attiria is your AI-powered personal stylist. Select your body type, skin tone, style, occasion, and budget —
            and we’ll generate 3 outfit ideas grounded in your catalog, plus optional outfit images.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    st.subheader("Provider status")
    provider = (os.getenv("PROVIDER") or "gemini").lower()
    st.write(f"Using PROVIDER = `{provider}`")
    st.caption(f"Loaded env from: {ENV_PATH}")
    st.write("GEMINI_API_KEY detected:", "✅" if os.getenv("GEMINI_API_KEY") else "❌")
    st.caption("Text model: " + (os.getenv("GEMINI_MODEL") or "models/gemini-1.5-flash"))
    st.caption("Image model: " + (os.getenv("GEMINI_IMAGE_MODEL") or "(disabled / quota-limited)"))

# --------- Hero banner + header ----------
st.image(os.path.join(ASSETS_DIR, "hero_banner.jpeg"), width="stretch")

top_l, top_r = st.columns([1, 5], vertical_alignment="center")
with top_l:
    st.image(os.path.join(ASSETS_DIR, "attiria_logo.jpeg"), width=88)
with top_r:
    st.markdown(
        """
        <div style="margin-top: 6px;">
          <div style="font-size: 44px; font-weight: 900; line-height: 1.0;">Attiria</div>
          <div class="attiria-muted" style="font-size: 16px; margin-top: 6px;">
            GenAI Stylist Demo (LangChain + Gemini) — dropdown-first styling with optional image generation.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

# --------- Preferences ----------
with st.container():
    st.markdown("<div class='attiria-card'>", unsafe_allow_html=True)
    st.markdown("### Your preferences")

    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
    with c1:
        body_type = st.selectbox("Body type", BODY_TYPES, index=0)
    with c2:
        skin_tone = st.selectbox("Skin tone", SKIN_TONES, index=1)
    with c3:
        style_type = st.selectbox("Preferred style", STYLE_TYPES, index=0)
    with c4:
        occasion = st.selectbox(
            "Occasion",
            ["work", "casual", "date", "brunch", "evening", "wedding_guest", "travel", "formal"],
            index=0,
        )
    with c5:
        budget_eur = st.number_input("Budget (€)", min_value=30, max_value=500, value=150, step=10)

    notes = st.text_area(
        "Notes (optional)",
        placeholder="e.g., I hate high heels. Prefer modest outfits. Weather is cold.",
    )
    temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.6, 0.1)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

# --------- Load catalog ----------
CATALOG_PATH = os.path.join(BASE_DIR, "catalog.csv")
try:
    df = load_catalog(CATALOG_PATH)
except Exception as e:
    st.error(f"Could not load catalog '{CATALOG_PATH}'. Put it next to this file. Error: {e}")
    df = None

# --------- Generate recommendations ----------
with st.container():
    st.markdown("<div class='attiria-card'>", unsafe_allow_html=True)
    st.markdown("### Generate recommendations")
    st.markdown(
        "<div class='attiria-muted'>Click to generate 3 outfits. You can optionally generate images for each outfit.</div>",
        unsafe_allow_html=True,
    )

    gen_disabled = df is None
    if st.button("✨ Generate outfits", width="stretch", disabled=gen_disabled):
        prefs = dict(
            body_type=body_type,
            skin_tone=skin_tone,
            style_type=style_type,
            occasion=occasion,
            budget_eur=budget_eur,
            notes=notes,
            temperature=temperature,
        )

        with st.spinner("Thinking like a stylist…"):
            try:
                result = recommend_outfits(prefs, df)
            except Exception as e:
                st.exception(e)
                st.stop()

        st.session_state["outfits_result_lang"] = result

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

# --------- Render results ----------
result = st.session_state.get("outfits_result_lang")
if result:
    outfits = result.get("outfits", [])
    if not outfits:
        st.warning("No outfits returned. Try again or loosen constraints.")
    else:
        st.markdown("## Results")
        for i, o in enumerate(outfits, start=1):
            st.markdown("<div class='attiria-card'>", unsafe_allow_html=True)

            st.markdown(f"### Outfit {i}: {o.get('title','')}")
            st.write(o.get("why_it_works", ""))

            tips = o.get("styling_tips", [])
            if tips:
                st.markdown("**Styling tips**")
                st.write("\n".join([f"- {t}" for t in tips]))

            image_prompt = o.get("image_prompt", "")

            # Optional image generation (may hit quota)
            if st.button(f"🖼️ Generate image for Outfit {i}", key=f"gen_img_lang_{i}", width="stretch"):
                with st.spinner("Generating image…"):
                    try:
                        img_bytes = generate_image_bytes(image_prompt)
                        st.image(img_bytes, caption=f"Outfit {i} preview", width="stretch")
                    except Exception as e:
                        st.error(f"Image generation failed: {e}")

            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)