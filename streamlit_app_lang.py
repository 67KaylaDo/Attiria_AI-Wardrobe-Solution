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
    initial_sidebar_state="expanded",  # ✅ sidebar starts open and can be reopened
)

# --------- UI mode state (DO NOT rely on browser theme) ----------
if "ui_mode" not in st.session_state:
    st.session_state["ui_mode"] = "light"  # default
MODE = st.session_state["ui_mode"]

# --------- Brand styling ----------
BG = "#FAF1E8"
PRIMARY = "#D00F47"
ACCENT = "#AC2156"
TEXT = "#1F1A1C"  # black-ish

# ---- Mode-specific tokens (colors only; content/layout unchanged) ----
if MODE == "dark":
    APP_BG = "#0E0B14"
    PANEL_BG = "rgba(24, 18, 34, 0.78)"
    PANEL_BORDER = "rgba(255,255,255,0.10)"
    TEXT_MAIN = "#F6F2FF"
    TEXT_MUTED = "rgba(246,242,255,0.70)"

    INPUT_BG = "rgba(255,255,255,0.06)"
    INPUT_TEXT = "#FFFFFF"
    PLACEHOLDER = "rgba(255,255,255,0.60)"

    POPOVER_BG = "rgba(18, 14, 26, 0.98)"
    POPOVER_TEXT = "#FFFFFF"

    ACCENT_PINK = "#D00F47"
    ACCENT_PURPLE = "#B56BFF"
else:
    APP_BG = BG
    PANEL_BG = "rgba(255, 255, 255, 0.78)"
    PANEL_BORDER = "rgba(0,0,0,0.06)"
    TEXT_MAIN = TEXT
    TEXT_MUTED = "rgba(31,26,28,0.68)"

    INPUT_BG = "rgba(31,26,28,0.88)"
    INPUT_TEXT = "#FFFFFF"
    PLACEHOLDER = "rgba(255,255,255,0.70)"

    POPOVER_BG = "rgba(18,18,20,0.95)"
    POPOVER_TEXT = "#FFFFFF"

    ACCENT_PINK = PRIMARY
    ACCENT_PURPLE = ACCENT

# --------- CSS ----------
st.markdown(
    f"""
    <style>
      .stApp {{
        background-color: {APP_BG};
        color: {TEXT_MAIN};
      }}

      div[data-testid="stWidgetLabel"] > label {{
        color: {TEXT_MAIN} !important;
        font-weight: 650 !important;
      }}

      h1, h2, h3, h4, h5, h6 {{
        color: {TEXT_MAIN} !important;
      }}

      p, span, div {{
        color: {TEXT_MAIN};
      }}

      .attiria-muted {{
        color: {TEXT_MUTED} !important;
      }}

      .attiria-card {{
        padding: 18px 18px;
        border-radius: 18px;
        background: {PANEL_BG};
        border: 1px solid {PANEL_BORDER};
        box-shadow: 0 10px 30px rgba(0,0,0,0.18);
      }}

      section[data-testid="stSidebar"] {{
        background-color: {PANEL_BG};
        border-right: 1px solid {PANEL_BORDER};
      }}
      section[data-testid="stSidebar"] * {{
        color: {TEXT_MAIN} !important;
      }}

      .stButton > button {{
        background-color: {ACCENT_PINK};
        color: #ffffff !important;
        border: 0;
        border-radius: 14px;
        padding: 0.7rem 1rem;
        font-weight: 800;
      }}
      .stButton > button:hover {{
        background-color: {ACCENT_PURPLE};
        color: #ffffff !important;
      }}
      .stButton > button * {{
        color: #ffffff !important;
      }}

      div[data-baseweb="select"] > div {{
        border-radius: 14px !important;
        background: {INPUT_BG} !important;
        border: 1px solid {PANEL_BORDER} !important;
      }}
      .stNumberInput input {{
        border-radius: 14px !important;
        background: {INPUT_BG} !important;
        color: {INPUT_TEXT} !important;
        border: 1px solid {PANEL_BORDER} !important;
      }}
      textarea {{
        border-radius: 14px !important;
        background: {INPUT_BG} !important;
        color: {INPUT_TEXT} !important;
        border: 1px solid {PANEL_BORDER} !important;
      }}
      textarea::placeholder {{
        color: {PLACEHOLDER} !important;
      }}

      div[data-baseweb="select"] * {{
        color: {INPUT_TEXT} !important;
      }}

      div[data-baseweb="popover"] > div {{
        background: {POPOVER_BG} !important;
        border-radius: 14px !important;
        border: 1px solid {PANEL_BORDER} !important;
        box-shadow: 0 18px 55px rgba(0,0,0,0.35) !important;
      }}
      div[data-baseweb="popover"] * {{
        color: {POPOVER_TEXT} !important;
      }}
      div[role="listbox"] div[role="option"],
      div[role="listbox"] div[role="option"] * {{
        color: {POPOVER_TEXT} !important;
      }}

      div[role="option"]:hover {{
        background: rgba(181,107,255,0.22) !important;
      }}
      div[aria-selected="true"] {{
        background: rgba(208,15,71,0.25) !important;
      }}

      div[data-testid="stSlider"] [role="slider"] {{
        box-shadow: 0 10px 22px rgba(208,15,71,0.28);
      }}

      /* =========================================
         SMALL "Activate" TOGGLE (ON = Dark mode)
         ========================================= */

      #attiria_theme_activate {{
        display: flex;
        justify-content: flex-end;
      }}

      /* keep it compact */
      #attiria_theme_activate [data-testid="stToggle"] {{
        transform: scale(0.95);
        transform-origin: right center;
      }}

      /* track */
      #attiria_theme_activate div[role="switch"] {{
        width: 44px !important;
        height: 22px !important;
        border-radius: 999px !important;
        background: rgba(255,255,255,0.14) !important;
        border: 1px solid {PANEL_BORDER} !important;
        transition: all 180ms ease !important;
        position: relative !important;
      }}

      /* knob */
      #attiria_theme_activate div[role="switch"]::before {{
        content: "";
        position: absolute;
        top: 2px;
        left: 2px;
        width: 18px;
        height: 18px;
        border-radius: 999px;
        background: rgba(255,255,255,0.92);
        box-shadow: 0 8px 18px rgba(0,0,0,0.22);
        transition: all 180ms ease !important;
      }}

      /* ON state (pink) */
      #attiria_theme_activate input:checked + div[role="switch"] {{
        background: {ACCENT_PINK} !important;
        border-color: rgba(255,255,255,0.10) !important;
        box-shadow: 0 10px 24px rgba(208,15,71,0.18) !important;
      }}

      #attiria_theme_activate input:checked + div[role="switch"]::before {{
        transform: translateX(22px);
        background: #ffffff;
      }}

      /* remove chrome */
      #MainMenu {{visibility: hidden;}}
      footer {{visibility: hidden;}}
      /* ✅ DO NOT hide header */
    </style>
    """,
    unsafe_allow_html=True,
)

# --------- Emoji + Title Case formatting (UI only) ----------
BODY_EMOJI = {
    "hourglass": "⏳",
    "pear": "🍐",
    "apple": "🍎",
    "rectangle": "▭",
    "inverted_triangle": "🔻",
}
SKIN_EMOJI = {
    "fair": "🌸",
    "light": "🌸",
    "wheatish": "🌾",
    "medium": "🌾",
    "tan": "🌞",
    "dark": "🌙",
    "deep": "🌙",
}
STYLE_EMOJI = {
    "boho": "✨",
    "minimal": "🖤",
    "classic": "👑",
    "chic": "💅",
    "street": "🧢",
    "workwear": "💼",
    "romantic": "🎀",
}
OCCASION_EMOJI = {
    "work": "💼",
    "casual": "👟",
    "date": "💘",
    "brunch": "🥐",
    "evening": "🌙",
    "wedding_guest": "💒",
    "travel": "✈️",
    "formal": "🎩",
}

def pretty_label(x: str, emoji_map: dict) -> str:
    s = str(x)
    emoji = emoji_map.get(s, "✨")
    text = s.replace("_", " ").title()
    return f"{emoji} {text}"

# --------- Sidebar: intro + provider (UNCHANGED content) ----------
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

# --------- Hero banner ----------
st.image(os.path.join(ASSETS_DIR, "hero_banner.jpeg"), width="stretch")

# --------- Theme Switch (TOP right): Activate (ON = dark mode) ----------
spacer, toggle_col = st.columns([8, 2])
with spacer:
    pass
with toggle_col:
    st.markdown('<div id="attiria_theme_activate">', unsafe_allow_html=True)

    is_dark = st.toggle(
        "Dark Mode",
        value=(MODE == "dark"),
        key="__attiria_activate_dark__",
    )

    st.markdown("</div>", unsafe_allow_html=True)

    new_mode = "dark" if is_dark else "light"
    if new_mode != MODE:
        st.session_state["ui_mode"] = new_mode
        st.rerun()

# --------- Hero header + title ----------
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
        body_type = st.selectbox(
            "Body type",
            BODY_TYPES,
            index=0,
            format_func=lambda x: pretty_label(x, BODY_EMOJI),
        )
    with c2:
        skin_tone = st.selectbox(
            "Skin tone",
            SKIN_TONES,
            index=1,
            format_func=lambda x: pretty_label(x, SKIN_EMOJI),
        )
    with c3:
        style_type = st.selectbox(
            "Preferred style",
            STYLE_TYPES,
            index=0,
            format_func=lambda x: pretty_label(x, STYLE_EMOJI),
        )
    with c4:
        occasion = st.selectbox(
            "Occasion",
            ["work", "casual", "date", "brunch", "evening", "wedding_guest", "travel", "formal"],
            index=0,
            format_func=lambda x: pretty_label(x, OCCASION_EMOJI),
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
                st.write("\\n".join([f"- {t}" for t in tips]))

            image_prompt = o.get("image_prompt", "")

            if st.button(f"🖼️ Generate image for Outfit {i}", key=f"gen_img_lang_{i}", width="stretch"):
                with st.spinner("Generating image…"):
                    try:
                        img_bytes = generate_image_bytes(image_prompt)
                        st.image(img_bytes, caption=f"Outfit {i} preview", width="stretch")
                    except Exception as e:
                        st.error(f"Image generation failed: {e}")

            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)