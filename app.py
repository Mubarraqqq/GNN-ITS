# app.py
from __future__ import annotations

import datetime as dt
from typing import Dict, Any, List
import os
import re
from PIL import Image
import json

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from ontology_engine import OntologyEngine
from question_bank import QUESTIONS

# ==================== AI API Configuration ====================
def get_ai_api_config():
    """Get AI API configuration from secrets or environment."""
    # Always use OpenAI as provider and get key from .env or environment
    import dotenv
    dotenv.load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "")
    config = {
        "provider": "openai",
        "api_key": api_key,
        "enabled": bool(api_key)
    }
    return config


def call_ai_api(prompt: str, max_tokens: int = 500) -> str | None:
    """Call AI API for insights and question generation."""
    config = get_ai_api_config()
    
    if not config["enabled"]:
        return None
    
    try:
        if config["provider"] == "openai":
            import openai
            client = openai.OpenAI(api_key=config["api_key"])
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content
        
        elif config["provider"] == "github":
            import openai
            client = openai.OpenAI(
                api_key=config["api_key"],
                base_url="https://models.inference.ai.azure.com"
            )
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content
        
        elif config["provider"] == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=config["api_key"])
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        
        elif config["provider"] == "google":
            import google.generativeai as genai
            genai.configure(api_key=config["api_key"])
            model = genai.GenerativeModel("gemini-pro")
            response = model.generate_content(prompt)
            return response.text
    
    except Exception as e:
        st.warning(f"AI API error: {str(e)}")
        return None
    
    return None


def generate_ai_question(concept_name: str, difficulty: str = "medium") -> Dict[str, Any] | None:
    """Generate an AI-powered practice question."""
    config = get_ai_api_config()
    if not config["enabled"]:
        return None
    
    prompt = f"""Generate a {difficulty} difficulty multiple choice question about {concept_name}.
    
    Response format (JSON):
    {{
        "question": "Your question here?",
        "options": ["Option A", "Option B", "Option C", "Option D"],
        "correct_idx": 0,
        "explanation": "Why this is correct..."
    }}
    
    Return only valid JSON, no additional text."""
    
    try:
        response = call_ai_api(prompt, max_tokens=300)
        if response:
            # Extract JSON from response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
    except Exception as e:
        st.warning(f"Failed to generate question: {str(e)}")
    
    return None


def generate_ai_insights(history: List[Dict], performance_data: Dict) -> str | None:
    """Generate AI-powered learning insights."""
    config = get_ai_api_config()
    if not config["enabled"]:
        return None
    
    # Prepare summary data
    total_questions = len(history)
    correct = sum(1 for h in history if h.get("correct"))
    accuracy = (correct / total_questions * 100) if total_questions > 0 else 0
    
    prompt = f"""As an expert learning coach, provide personalized learning insights based on this student data:
    
    - Total questions attempted: {total_questions}
    - Correct answers: {correct}
    - Accuracy: {accuracy:.1f}%
    - Study sessions: {performance_data.get('study_days', 0)}
    - Hints used: {performance_data.get('total_hints', 0)}
    
    Provide:
    1. One key strength to celebrate
    2. One area for improvement
    3. One specific action to take next
    
    Keep response concise (3-4 sentences) and motivating."""
    
    return call_ai_api(prompt, max_tokens=200)


# ==================== End AI Configuration ====================


@st.cache_resource
def get_engine() -> OntologyEngine:
    return OntologyEngine()


def init_session_state():
    ss = st.session_state
    ss.setdefault("student_iri", "http://www.co-ode.org/ontologies/ont.owl#StudentAdvanced01")
    ss.setdefault("current_objective_iri", None)
    ss.setdefault("current_task_iri", None)
    ss.setdefault("current_question_id", None)
    ss.setdefault("hint_level", 0)
    ss.setdefault("history", [])


def questions_for_objective(obj_iri: str) -> List[str]:
    return [qid for qid, q in QUESTIONS.items() if q["objective_iri"] == obj_iri]


def pick_next_question(obj_iri: str, current_id: str | None) -> str | None:
    ids = questions_for_objective(obj_iri)
    if not ids:
        return None
    if current_id not in ids:
        return ids[0]
    idx = ids.index(current_id)
    return ids[(idx + 1) % len(ids)]


def check_answer(q: Dict[str, Any], user_answer: Any) -> bool | None:
    if q["type"] == "MC":
        # user_answer is choice id
        for c in q["mc_choices"]:
            if c["id"] == user_answer:
                return c["correct"]
        return False
    elif q["type"] == "NUMERIC":
        try:
            user_val = float(user_answer)
        except (TypeError, ValueError):
            return False
        target = float(q["numeric_answer"])
        tol = float(q.get("numeric_tolerance", 0.0))
        return abs(user_val - target) <= tol
    else:
        # REFLECTION: we don't auto-mark; return None
        return None


def display_banner_image(title: str, emoji: str = "🧠"):
    """Display a colorful banner with title and emoji"""
    st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #3366ff 0%, #667eea 100%);
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        ">
            <h1 style="color: white; margin: 0; font-size: 2.5em;">{emoji} {title}</h1>
        </div>
    """, unsafe_allow_html=True)


def get_performance_icon(accuracy: float) -> str:
    """Get emoji based on accuracy level"""
    if accuracy >= 90:
        return "🌟"
    elif accuracy >= 75:
        return "⭐"
    elif accuracy >= 60:
        return "👍"
    elif accuracy >= 40:
        return "📚"
    else:
        return "🚀"


def create_progress_ring(value: float, max_value: float = 100) -> str:
    """Create an HTML progress ring visualization"""
    percentage = (value / max_value) * 100
    color = "#10b981" if percentage >= 75 else "#f59e0b" if percentage >= 50 else "#ef4444"
    
    return f"""
    <div style="text-align: center; padding: 20px;">
        <svg width="150" height="150" style="transform: rotate(-90deg);">
            <circle cx="75" cy="75" r="70" stroke="#e5e7eb" stroke-width="8" fill="none"/>
            <circle cx="75" cy="75" r="70" stroke="{color}" stroke-width="8" fill="none"
                    stroke-dasharray="{percentage * 4.398}" stroke-dashoffset="0"
                    style="transition: stroke-dasharray 0.5s ease;"/>
        </svg>
        <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); 
                    font-size: 2em; font-weight: bold; color: {color};">
            {percentage:.1f}%
        </div>
    </div>
    """


def display_concept_image(concept_name: str, width: int = 600):
    """Display a concept image if it exists"""
    image_path = f"images/concepts/{concept_name}.png"
    if os.path.exists(image_path):
        try:
            img = Image.open(image_path)
            st.image(img, use_column_width=True, caption=f"{concept_name} Illustration")
            return True
        except Exception as e:
            st.warning(f"Could not load image: {e}")
            return False
    return False


def display_task_image(task_name: str):
    """Display a task-related image if it exists"""
    # Try different naming conventions
    for filename in [f"{task_name}.png", f"{task_name.lower()}.png", f"{task_name.replace(' ', '_')}.png"]:
        image_path = f"images/tasks/{filename}"
        if os.path.exists(image_path):
            try:
                img = Image.open(image_path)
                st.image(img, use_column_width=True, caption=f"{task_name}")
                return True
            except Exception as e:
                st.warning(f"Could not load image: {e}")
    return False


def display_mastery_badge(mastery_level: float) -> str:
    """Create a visual mastery badge based on level"""
    if mastery_level >= 90:
        badge = "🏆 Master"
        color = "#fbbf24"
    elif mastery_level >= 75:
        badge = "⭐ Expert"
        color = "#3b82f6"
    elif mastery_level >= 60:
        badge = "👍 Proficient"
        color = "#10b981"
    elif mastery_level >= 40:
        badge = "📚 Learning"
        color = "#f59e0b"
    else:
        badge = "🌱 Beginner"
        color = "#ef4444"
    
    return f'<span style="background: {color}; color: white; padding: 5px 12px; border-radius: 20px; font-weight: bold; font-size: 0.85em;">{badge}</span>'


def clean_objective_name(name: str) -> str:
    """Remove technical prefixes like 'Obj...' and return a clean concept name"""
    if re.match(r'Obj[A-Z][a-zA-Z0-9]+', name):
        return "Graph Neural Networks"
    return name


def main():
    st.set_page_config(
        page_title="GNN Intelligent Tutoring System",
        layout="wide",
        initial_sidebar_state="collapsed",
        menu_items={
            "Get Help": "https://github.com",
            "Report a bug": "https://github.com",
            "About": "GNN Intelligent Tutoring System - Powered by Ontology"
        }
    )
    
    # Custom styling

    st.markdown("""
        <style>
        :root {
            --primary-color: #ff9800;
            --secondary-color: #fff;
            --success-color: #ff9800;
            --warning-color: #ff9800;
            --error-color: #ef4444;
        }
        body, .main {
            background: linear-gradient(135deg, #fff 0%, #ffe0b2 100%);
        }
        button {
            border-radius: 8px !important;
            font-weight: 600 !important;
            background: #ff9800 !important;
            color: #fff !important;
            transition: all 0.3s ease !important;
        }
        .stMetric {
            background: #fff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(255,152,0,0.15);
        }
        .stMetricValue {
            color: #ff9800 !important;
            font-size: 32px !important;
            font-weight: 700 !important;
        }
        h1, h2, h3 {
            color: #ff9800 !important;
            font-weight: 700 !important;
            margin-top: 20px !important;
            margin-bottom: 15px !important;
        }
        h1 {
            background: linear-gradient(135deg, #ff9800 0%, #fff3e0 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 30px !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 15px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 12px 24px !important;
            border-radius: 8px !important;
            border: none !important;
            background: rgba(255,255,255,0.8);
            font-weight: 600;
            color: #000 !important;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #ff9800 0%, #fff3e0 100%) !important;
            color: #000 !important;
        }
        .streamlit-expanderHeader {
            background-color: #fff3e0 !important;
            border-radius: 8px !important;
        }
        .stSuccess {
            background: rgba(255,152,0,0.08) !important;
            border-left: 4px solid #ff9800 !important;
            border-radius: 8px !important;
        }
        .stError {
            background: rgba(239, 68, 68, 0.1) !important;
            border-left: 4px solid #ef4444 !important;
            border-radius: 8px !important;
        }
        .stWarning {
            background: rgba(255,152,0,0.08) !important;
            border-left: 4px solid #ff9800 !important;
            border-radius: 8px !important;
        }
        .stInfo {
            background: rgba(255,152,0,0.08) !important;
            border-left: 4px solid #ff9800 !important;
            border-radius: 8px !important;
        }
        .stSelectbox, .stTextInput, .stNumberInput {
            border-radius: 8px !important;
            border: 2px solid #ff9800 !important;
        }
        .stRadio > label {
            background: #fff;
            padding: 12px 16px;
            border-radius: 8px;
            margin-bottom: 10px;
            border: 2px solid #ffe0b2;
            transition: all 0.3s ease;
        }
        .stRadio > label:hover {
            border-color: #ff9800;
            background: #fff3e0;
        }
        .stDataFrame {
            border-radius: 8px !important;
        }
        hr {
            margin: 30px 0 !important;
            border: none !important;
            height: 2px;
            background: linear-gradient(90deg, transparent, #ffe0b2, transparent);
        }
        </style>
    """, unsafe_allow_html=True)
    
    init_session_state()
    engine = get_engine()
    O, G = engine.O, engine.G  # not currently used directly, but handy if you expand

    # Display main banner
    st.markdown("""
        <div style="
            background: linear-gradient(135deg, #ff9800 0%, #fff3e0 100%);
            border-radius: 16px;
            box-shadow: 0 2px 12px rgba(255,152,0,0.10);
            padding: 48px 0;
            margin-bottom: 24px;
            text-align: center;
        ">
            <span style="font-size:2.5em; font-weight:700; color:#fff; letter-spacing:1px;">
                🕸️ GNN Intelligent Tutoring System
            </span>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("*<div style='text-align: center;'>Master Graph Neural Networks through adaptive, ontology-driven learning</div>*", unsafe_allow_html=True)

    tab_labels = ["📚 Overview", "📖 Learn", "✍️ Practice", "📊 Progress", "💡 Insights"]
    tab_overview, tab_learn, tab_practice, tab_progress, tab_insights = st.tabs(tab_labels)
    st.markdown("""
        <style>
        .golden-rule-tip {
            background: #fff3e0;
            border-left: 4px solid #ff9800;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 10px;
            font-size: 0.98em;
        }
        .sticky-header {
            position: sticky;
            top: 0;
            z-index: 100;
            background: linear-gradient(90deg, #ff9800 0%, #fff3e0 100%);
            color: #fff;
            padding: 16px 0 8px 0;
            font-size: 2em;
            text-align: center;
            box-shadow: 0 2px 8px rgba(255,152,0,0.15);
        }
        .animated-btn {
            transition: box-shadow 0.2s, transform 0.2s;
        }
        .animated-btn:hover {
            box-shadow: 0 4px 16px rgba(255,152,0,0.25);
            transform: scale(1.04);
        }
        .ontology-panel {
            background: #fff;
            border-radius: 12px;
            box-shadow: 0 2px 12px rgba(51,102,255,0.08);
            padding: 18px;
            margin-bottom: 18px;
        }
        .search-bar {
            border-radius: 8px;
            border: 2px solid #ff9800;
            padding: 8px 12px;
            font-size: 1em;
            width: 100%;
            margin-bottom: 10px;
        }
        .progress-indicator {
            background: linear-gradient(90deg, #ff9800 0%, #fff3e0 100%);
            color: #fff;
            border-radius: 8px;
            padding: 8px 16px;
            font-weight: 600;
            margin-bottom: 10px;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)
    st.markdown("<div class='sticky-header'>GNN Intelligent Tutoring System</div>", unsafe_allow_html=True)

    # ---------- OVERVIEW TAB ----------
    with tab_overview:
        #st.markdown("<div class='ontology-panel'>", unsafe_allow_html=True)
        st.subheader("🎯 Choose a learning objective")
        #st.markdown("<div class='golden-rule-tip'>Consistent controls and terminology. Select your learning objective below. You can always switch objectives.</div>", unsafe_allow_html=True)


        # Simple dropdown for objectives (no search)
        objectives = engine.list_objectives()
        if not objectives:
            st.error("❌ No LearningObjective instances found in the ontology.")
        else:
            names = [o.name for o in objectives]
            iri_by_name = {o.name: o.iri for o in objectives}

            current_name = None
            if st.session_state.current_objective_iri:
                for o in objectives:
                    if o.iri == st.session_state.current_objective_iri:
                        current_name = o.name
                        break

            # Always show all objectives in dropdown

            # If no current objective, don't set index (forces user to select)
            if current_name in names:
                choice = st.selectbox(
                    "Learning objective:",
                    names,
                    index=names.index(current_name),
                    help="Select your learning objective. ",
                    key="overview_objective_selector"
                )
            else:
                choice = st.selectbox(
                    "Learning objective:",
                    names,
                    help="Select your learning objective. ",
                    key="overview_objective_selector"
                )

            selected_iri = iri_by_name[choice]
            selected_obj = engine.get_objective_by_iri(selected_iri)
            info = engine.objective_info(selected_obj)

            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"#### {info.name}")
                if info.description:
                    st.write(info.description)
            with col2:
                if info.level:
                    st.metric("Difficulty Level", info.level)

            if st.button("🚀 Start / switch to this objective", key="btn_select_objective", use_container_width=True):
                st.session_state.current_objective_iri = info.iri
                st.session_state.current_task_iri = None
                st.session_state.current_question_id = None
                st.session_state.hint_level = 0
                st.success("✅ Objective updated! Now, please click the 'Learn' tab to continue with your selected objective.")

            st.markdown("---")
            st.markdown("### 📋 Related assessments in the ontology")
            #st.markdown("<div class='golden-rule-tip'>Get feedback on your progress and requirements.</div>", unsafe_allow_html=True)

            assessments = engine.assessments_for_objective(selected_obj)
            if not assessments:
                st.info("ℹ️ No explicit Assessment individuals linked to this objective.")
            else:
                for a in assessments:
                    with st.expander(f"Assessment: {a.name}", expanded=False):
                        st.markdown(f"""
                            <div style="
                                background: white;
                                padding: 20px;
                                border-radius: 10px;
                                border-left: 5px solid #3366ff;
                                margin-bottom: 15px;
                                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                            ">
                                <h3 style="margin-top: 0; color: #1f2937;">{a.name}</h3>
                            </div>
                        """, unsafe_allow_html=True)
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            if a.description:
                                st.write(a.description)
                        with col2:
                            if a.current_score is not None and a.max_score is not None:
                                score_pct = (float(a.current_score) / float(a.max_score)) * 100
                                icon = get_performance_icon(score_pct)
                                st.metric(f"{icon} Score", f"{score_pct:.0f}%")
                        if a.required_concepts:
                            st.markdown("**🔗 Requires concepts:**")
                            cols = st.columns(min(3, len(a.required_concepts)))
                            for idx, c_iri in enumerate(a.required_concepts):
                                c = engine.describe_concept(c_iri)
                                with cols[idx % len(cols)]:
                                    st.caption(f"• {c['name']} ({c['kind']})")
                        st.divider()
            #st.markdown("<div class='golden-rule-tip'>You can always restart or change your objective. </div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- LEARN TAB ----------
    with tab_learn:
        st.success("✅ Learn tab opened! Once you're ready, Select **Search tasks** or **Available tasks**, then click the **Practice** tab to start answering questions.")
        st.markdown("<div class='ontology-panel'>", unsafe_allow_html=True)
        st.subheader("📚 Ontology-driven learning tasks")
        st.markdown("<div class='golden-rule-tip'>Consistent navigation and feedback. Select a task and see all related info.</div>", unsafe_allow_html=True)

        obj_iri = st.session_state.current_objective_iri
        if not obj_iri:
            st.info("👈 Choose an objective in the **Overview** tab first.")
        else:
            obj = engine.get_objective_by_iri(obj_iri)
            tasks = engine.tasks_for_objective(obj)

            if not tasks:
                st.warning("⚠️ No LearningTask instances linked to this objective.")
            else:
                task_names = [t.name for t in tasks]
                iri_by_name = {t.name: t.iri for t in tasks}

                search_task = st.text_input("🔍 Search tasks:", "", key="task_search", help="Type to filter tasks.", args={"class": "search-bar"})
                filtered_task_names = [n for n in task_names if search_task.lower() in n.lower()] if search_task else task_names

                current_task_name = None
                if st.session_state.current_task_iri:
                    for t in tasks:
                        if t.iri == st.session_state.current_task_iri:
                            current_task_name = t.name
                            break

                selected_task_name = st.selectbox(
                    "Available tasks:",
                    filtered_task_names,
                    index=filtered_task_names.index(current_task_name) if current_task_name in filtered_task_names else 0,
                    help="Consistent selection control."
                )
                t_iri = iri_by_name[selected_task_name]
                st.session_state.current_task_iri = t_iri

                selected_task_info = next(t for t in tasks if t.iri == t_iri)

                st.markdown(f"#### {selected_task_info.name}")
                if selected_task_info.description:
                    st.write(selected_task_info.description)
                display_task_image(selected_task_info.name)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📈 Difficulty", selected_task_info.difficulty or "n/a")
                with col2:
                    st.metric("⏱️ Est. time", selected_task_info.estimated_time or "n/a")
                with col3:
                    st.metric("💻 Requires coding?", "Yes" if selected_task_info.requires_coding else "No")

                st.divider()

                if selected_task_info.concept_iris:
                    st.markdown("##### 🔗 Linked GNN concepts")
                    concept_cols = st.columns(min(3, len(selected_task_info.concept_iris)))
                    for idx, c_iri in enumerate(selected_task_info.concept_iris):
                        c_info = engine.describe_concept(c_iri)
                        with concept_cols[idx % len(concept_cols)]:
                            st.info(f"**{c_info['name']}** ({c_info['kind']})")

                if selected_task_info.dataset_iris:
                    st.markdown("##### 📊 Graph datasets used")
                    for d_iri in selected_task_info.dataset_iris:
                        d = engine.describe_concept(d_iri)
                        details = d["details"]
                        extra = []
                        if details.get("datasetName"):
                            extra.append(f"*{details['datasetName']}*")
                        if details.get("numGraphs") is not None:
                            extra.append(f"**{details['numGraphs']}** graphs")
                        line = f"📁 **{d['name']}**"
                        if extra:
                            line += " – " + ", ".join(extra)
                        st.write(line)

                if selected_task_info.graph_iris:
                    st.markdown("##### 🕸️ Example graph instances")
                    for g_iri in selected_task_info.graph_iris:
                        g = engine.describe_concept(g_iri)
                        details = g["details"]
                        nodes = details.get('numNodes', '?')
                        edges = details.get('numEdges', '?')
                        st.write(f"🔷 **{g['name']}** – {nodes} nodes, {edges} edges")

                st.info("💡 This panel is *directly driven* by your OWL ontology: tasks, concepts, graphs, and datasets are not hard-coded.")
                st.markdown("<div class='golden-rule-tip'>You can always restart or change your task.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- PRACTICE TAB ----------
    with tab_practice:
        st.success("✅ Practice tab opened! Answer questions here, then click the **Progress** tab to review your performance.")
        st.markdown("<div class='ontology-panel'>", unsafe_allow_html=True)
        st.subheader("✍️ Practice Session")
        st.markdown("<div class='golden-rule-tip'>Consistent controls, clear instructions, and feedback. Set your session below.</div>", unsafe_allow_html=True)

        obj_iri = st.session_state.current_objective_iri
        if not obj_iri:
            st.info("👈 Choose an objective in the **Overview** tab first.")
        else:
            st.markdown("#### Practice: AI-Generated Questions (Progressive)")
            st.markdown("**How many questions do you want to practice?**")
            num_questions = st.number_input(
                "Enter number of questions to generate:",
                min_value=1,
                max_value=50,
                value=10,
                step=1,
                key="practice_num_questions",
                help="Set the number of questions before starting your session."
            )
            st.caption("Set the number of questions before starting your session.")

            # Get the selected objective name to use as default concept
            obj = engine.get_objective_by_iri(obj_iri)
            objective_name = obj.name if obj else "Graph Neural Networks"
            default_concept = clean_objective_name(objective_name)
            concept = st.text_input("Concept to practice:", value=default_concept, key="ai_practice_concept", help="Based on your selected learning objective.")
            difficulty = st.selectbox("Difficulty:", ["Easy", "Medium", "Hard"], key="ai_practice_difficulty", help="Select difficulty level. ")

            # Session state for progressive practice
            ss = st.session_state
            ss.setdefault("practice_questions", [])
            ss.setdefault("practice_current_idx", 0)
            ss.setdefault("practice_answers", [])
            ss.setdefault("practice_started", False)
            ss.setdefault("practice_complete", False)

            import random
            def generate_practice_questions(concept, difficulty, n):
                # Use selected learning objective for context
                obj_iri = st.session_state.get("current_objective_iri")
                obj_name = None
                if obj_iri:
                    obj = engine.get_objective_by_iri(obj_iri)
                    obj_name = getattr(obj, "name", None)
                # Clean up topic for plausibility
                import re
                def plausible_topic(name):
                    # Remove technical or code-like names
                    if re.match(r'Obj[A-Z][a-zA-Z0-9]+', name):
                        return "Graph Convolutional Network Training"
                    return name
                topic = plausible_topic(obj_name) if obj_name else concept
                questions = []
                prompt = (
                    f"Generate {n} unique and diverse questions (mix of MC and Theory) covering all key concepts, applications, challenges, and recent advances in the topic: '{topic}'. "
                    f"Questions must be plausible, clear, and suitable for a human learner. Avoid technical names like 'ObjTrainGCNModel' and use conceptual language. Each question should be different and not repeated. Format: For MC, provide question, 4 options, and correct index. For Theory, provide question only. Difficulty: {difficulty}."
                )
                ai_response = call_ai_api(prompt, max_tokens=1200)
                q_blocks = re.split(r'\n\d+\. ', '\n' + ai_response)
                for block in q_blocks:
                    block = block.strip()
                    if not block:
                        continue
                    # Filter out implausible questions
                    if re.search(r'Obj[A-Z][a-zA-Z0-9]+', block):
                        continue
                    if 'Options:' in block:
                        q_match = re.match(r'(.*)Options:(.*)Correct Index:(\d+)', block, re.DOTALL)
                        if q_match:
                            q_text = q_match.group(1).strip()
                            # Remove any "Correct:" information from question text
                            q_text = re.sub(r'\nCorrect[:\-].*', '', q_text, flags=re.IGNORECASE | re.DOTALL)
                            opts = [o.strip() for o in q_match.group(2).split('\n') if o.strip()]
                            # Remove any "Correct:" labels from options
                            opts = [re.sub(r'Correct[:\-].*', '', o, flags=re.IGNORECASE).strip() for o in opts if o.strip()]
                            correct_idx = int(q_match.group(3).strip())
                            questions.append({
                                "question": q_text,
                                "options": opts,
                                "correct_idx": correct_idx,
                                "type": "MC"
                            })
                    else:
                        # Remove any "Correct:" information from theory questions
                        block_clean = re.sub(r'\nCorrect[:\-].*', '', block, flags=re.IGNORECASE | re.DOTALL).strip()
                        if block_clean:
                            questions.append({
                                "question": block_clean,
                                "type": "THEORY"
                            })
                if len(questions) < n:
                    for i in range(n - len(questions)):
                        qtype = "MC" if i % 2 == 0 else "THEORY"
                        sub_prompt = (
                            f"Generate a {qtype} question about the topic '{topic}' at {difficulty} difficulty. Make it unique, plausible, and not repeated. Use conceptual language, not technical names."
                        )
                        q = generate_ai_question(sub_prompt, difficulty.lower())
                        if qtype == "THEORY":
                            q = {
                                "question": f"(Theory) {q.get('question', 'Describe the concept: ' + topic)}",
                                "type": "THEORY"
                            }
                        else:
                            q["type"] = "MC"
                        # Filter again
                        if re.search(r'Obj[A-Z][a-zA-Z0-9]+', q.get('question', '')):
                            continue
                        questions.append(q)
                return questions[:n]

            if not ss["practice_started"]:
                if st.button("Start Practice Session", key="start_practice_btn", args={"class": "animated-btn"}):
                    with st.spinner(f"Generating {num_questions} questions on {concept}..."):
                        ss["practice_questions"] = generate_practice_questions(concept, difficulty, num_questions)
                        ss["practice_current_idx"] = 0
                        ss["practice_answers"] = []
                        ss["practice_started"] = True
                        ss["practice_complete"] = False
                        st.success("✅ Practice session started! ")
            else:
                questions = ss["practice_questions"]
                idx = ss["practice_current_idx"]
                answers = ss["practice_answers"]
                total = len(questions)

                st.markdown(f"<div class='progress-indicator'>Question {idx+1} of {total}</div>", unsafe_allow_html=True)

                if idx < total:
                    q = questions[idx]
                    st.markdown(f"**Question {idx+1} of {total}:** {q.get('question', 'N/A')}")
                    st.markdown("<div class='golden-rule-tip'>Progress is shown above. </div>", unsafe_allow_html=True)
                    if q.get("type") == "MC":
                        options = q.get('options', [])
                        if options:
                            user_choice = st.radio("Select an answer:", [f"{i+1}: {opt}" for i, opt in enumerate(options)], key=f"practice_choice_{idx}", help="Choose your answer.")
                            chosen_idx = int(user_choice.split(":")[0]) - 1 if user_choice else None
                            if st.button("Submit Answer", key=f"practice_submit_{idx}", args={"class": "animated-btn"}):
                                correct = (chosen_idx == q.get('correct_idx', -1))
                                answers.append({"type": "MC", "correct": correct, "user_choice": chosen_idx, "question": q.get('question', ''), "options": options})
                                ss["practice_answers"] = answers
                                ss["practice_current_idx"] += 1
                                st.success("Answer submitted! ")
                                st.rerun()
                    elif q.get("type") == "THEORY":
                        user_answer = st.text_area("Your answer:", key=f"practice_theory_{idx}", help="Type your answer.")
                        if st.button("Submit Answer", key=f"practice_submit_{idx}", args={"class": "animated-btn"}):
                            # Use OpenAI to check similarity and assign 1 mark if close
                            ref_answer_prompt = f"Provide a model answer for: {q.get('question', '')}"
                            ref_answer = call_ai_api(ref_answer_prompt, max_tokens=80)
                            sim_prompt = (
                                f"Compare the following student answer to the reference answer. If the student answer is close in meaning, assign 1 mark. Otherwise, assign 0.\n"
                                f"Reference: {ref_answer}\nStudent: {user_answer}\nOutput: Mark (0 or 1) and brief feedback."
                            )
                            assessment = call_ai_api(sim_prompt, max_tokens=60)
                            # Parse mark from assessment
                            import re
                            mark_match = re.search(r'Mark\s*[:\-]?\s*(\d)', assessment)
                            mark = int(mark_match.group(1)) if mark_match else 0
                            answers.append({
                                "type": "THEORY",
                                "user_answer": user_answer,
                                "assessment": assessment,
                                "question": q.get('question', ''),
                                "mark": mark
                            })
                            ss["practice_answers"] = answers
                            ss["practice_current_idx"] += 1
                            st.success("Answer submitted! ")
                            st.rerun()
                else:
                    ss["practice_complete"] = True
                    st.success("🎉 Practice session complete! ")
                    st.markdown("### Results Summary")
                    mc_results = [a for a in answers if a["type"] == "MC"]
                    theory_results = [a for a in answers if a["type"] == "THEORY"]
                    if mc_results:
                        correct_count = sum(1 for a in mc_results if a["correct"])
                        st.write(f"**MC Questions:** {correct_count}/{len(mc_results)} correct")
                    if theory_results:
                        total_marks = sum(a.get("mark", 0) for a in theory_results)
                        st.write(f"**Theory Questions:** {total_marks}/{len(theory_results)} marks")
                        for i, a in enumerate(theory_results, 1):
                            st.markdown(f"**Q{i}:** {a['question']}")
                            st.markdown(f"Your answer: {a['user_answer']}")
                            st.markdown(f"Assessment: {a['assessment']}")
                    # Save results to session history for analytics
                    if "history" not in st.session_state:
                        st.session_state["history"] = []
                    for a in answers:
                        entry = {
                            "question": a.get("question", ""),
                            "type": a.get("type", ""),
                            "correct": a.get("correct", a.get("mark", 0)),
                            "user_answer": a.get("user_answer", ""),
                            "options": a.get("options", []),
                            "evaluated": True,
                            "timestamp": pd.Timestamp.now(),
                            "concept_iri": st.session_state.get("current_objective_iri", ""),
                            "objective_iri": st.session_state.get("current_objective_iri", ""),
                            "question_type": a.get("type", "")
                        }
                        st.session_state["history"].append(entry)
                    st.markdown("<div class='golden-rule-tip'>You can restart your session below. </div>", unsafe_allow_html=True)
                    if st.button("Restart Practice Session", key="restart_practice_btn", args={"class": "animated-btn"}):
                        ss["practice_questions"] = []
                        ss["practice_current_idx"] = 0
                        ss["practice_answers"] = []
                        ss["practice_started"] = False
                        ss["practice_complete"] = False
                        st.success("Session reset!")
                        st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- PROGRESS TAB ----------
    with tab_progress:
        st.success("✅ Progress tab opened! Review your analytics here, then click the **Insights** tab for AI-powered recommendations.")
        st.subheader("📊 Learner progress & analytics")

        if not st.session_state.history:
            st.info("👈 No attempts recorded yet. Answer some questions in the **Practice** tab.")
        else:
            df = pd.DataFrame(st.session_state.history)

            # Overall stats with visual cards
            st.markdown("### 📈 Overall performance")
            evaluated = df[df["evaluated"]]
            if not evaluated.empty:
                overall_acc = evaluated["correct"].mean()
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    icon = get_performance_icon(overall_acc * 100)
                    st.metric(f"{icon} Overall accuracy", f"{overall_acc * 100:.1f}%")
                with col2:
                    st.metric("📝 Total attempts", len(df))
                with col3:
                    st.metric("🎯 Correct answers", int(evaluated["correct"].sum()))
                with col4:
                    if len(df) > 0:
                        perfect_streak = 0
                        for val in reversed(evaluated["correct"].values):
                            if val:
                                perfect_streak += 1
                            else:
                                break
                        st.metric("🔥 Current streak", perfect_streak)
            else:
                st.write("No auto-graded questions yet (only reflection items).")

            st.divider()

            # Accuracy by objective
            st.markdown("### 📊 Accuracy by learning objective")
            if not evaluated.empty:
                acc_by_obj = (
                    evaluated.groupby("objective_iri")["correct"].mean().rename("accuracy")
                )
                if not acc_by_obj.empty:
                    # map IRIs to names for display
                    obj_labels = {}
                    for o in engine.list_objectives():
                        obj_labels[o.iri] = o.name

                    acc_display = acc_by_obj.reset_index()
                    acc_display["objective"] = acc_display["objective_iri"].map(
                        lambda iri: obj_labels.get(iri, iri)
                    )
                    
                    # Create a bar chart
                    fig = px.bar(
                        acc_display,
                        x="objective",
                        y="accuracy",
                        title="Accuracy by Objective",
                        labels={"accuracy": "Accuracy (%)", "objective": "Objective"},
                        color="accuracy",
                        color_continuous_scale="RdYlGn",
                        range_color=[0, 1]
                    )
                    fig.update_yaxes(tickformat=".0%")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.dataframe(
                        acc_display[["objective", "accuracy"]].style.format(
                            {"accuracy": "{:.1%}"}
                        ),
                        use_container_width=True
                    )
                else:
                    st.write("No auto-graded questions yet.")
            else:
                st.write("No auto-graded questions yet.")

            st.divider()

            # Attempts by concept
            st.markdown("### 🔬 Attempts by concept")
            attempts_by_concept = df.groupby("concept_iri").size().rename("attempts")
            concept_rows = []
            for iri, n in attempts_by_concept.items():
                c = engine.describe_concept(iri)
                concept_rows.append(
                    {"concept": c["name"], "kind": c["kind"], "attempts": n}
                )
            concept_df = pd.DataFrame(concept_rows)
            
            if not concept_df.empty:
                # Create a bar chart for attempts by concept
                fig = px.bar(
                    concept_df,
                    x="concept",
                    y="attempts",
                    title="Practice Attempts by Concept",
                    labels={"attempts": "Number of Attempts", "concept": "Concept"},
                    color="attempts",
                    color_continuous_scale="Blues"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(concept_df, use_container_width=True)

            st.markdown("---")
            st.caption(
                "💡 This analytics view mirrors the idea of the `ProgressTracker` and "
                "`Assessment` components in your ontology, using the ontology's "
                "learning objectives and concepts to organise the data."
            )

    # ---------- INSIGHTS TAB ----------
    with tab_insights:
        st.subheader("💡 AI-Powered Learning Insights & Analysis")

        if not st.session_state.history:
            st.info("👈 No attempts recorded yet. Start practicing to get personalized insights!")
        else:
            df = pd.DataFrame(st.session_state.history)
            evaluated = df[df["evaluated"]]


            # --- Amplified Key Metrics & Visuals ---
            st.markdown("### 📊 Key Metrics & Dynamic Visuals")
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric("Total Questions", len(df), help="Number of questions attempted")
            with col2:
                if not evaluated.empty:
                    acc = evaluated["correct"].mean() * 100
                    st.metric("Accuracy (%)", f"{acc:.1f}", help="Overall accuracy across all attempts")
                else:
                    st.metric("Accuracy (%)", "N/A")
            with col3:
                if not evaluated.empty and "hints_used" in df.columns:
                    avg_hints = df["hints_used"].mean()
                    st.metric("Avg Hints", f"{avg_hints:.2f}", help="Average hints used per question")
                else:
                    st.metric("Avg Hints", "N/A")
            with col4:
                if len(df) > 0:
                    study_days = len(pd.to_datetime(df["timestamp"]).dt.date.unique())
                    st.metric("Study Days", study_days, help="Unique days you practiced")
                else:
                    st.metric("Study Days", "0")
            with col5:
                if not evaluated.empty:
                    streak = 0
                    for val in reversed(evaluated["correct"].values):
                        if val:
                            streak += 1
                        else:
                            break
                    st.metric("Current Streak", streak, help="Consecutive correct answers")
                else:
                    st.metric("Current Streak", "0")

            st.divider()

            # Concept Mastery Analysis
            st.markdown("### 🧠 Concept Mastery Analysis")
            if not evaluated.empty:
                # Build aggregation dict dynamically based on available columns
                agg_dict = {
                    "correct": ["sum", "count", "mean"]
                }
                if "hints_used" in evaluated.columns:
                    agg_dict["hints_used"] = "mean"
                
                concept_perf = evaluated.groupby("concept_iri").agg(agg_dict)
                
                # Flatten multi-level columns
                concept_perf.columns = ['_'.join(col).strip() if col[1] else col[0] for col in concept_perf.columns.values]
                
                # Rename columns to readable names
                if "hints_used_mean" in concept_perf.columns:
                    concept_perf = concept_perf.rename(columns={
                        "correct_sum": "Correct",
                        "correct_count": "Total",
                        "correct_mean": "Accuracy",
                        "hints_used_mean": "Avg Hints"
                    })
                else:
                    concept_perf = concept_perf.rename(columns={
                        "correct_sum": "Correct",
                        "correct_count": "Total",
                        "correct_mean": "Accuracy"
                    })
                
                concept_perf = concept_perf.reset_index()
                
                # Map IRIs to concept names
                concept_perf["Concept"] = concept_perf["concept_iri"].apply(
                    lambda iri: engine.describe_concept(iri)["name"]
                )
                # Convert Accuracy to numeric and calculate Mastery %
                concept_perf["Accuracy"] = pd.to_numeric(concept_perf["Accuracy"], errors='coerce')
                concept_perf["Mastery %"] = (concept_perf["Accuracy"] * 100).round(1)
                
                # Add mastery badges
                concept_perf["Badge"] = concept_perf["Mastery %"].apply(display_mastery_badge)
                
                # Display only available columns
                if "Avg Hints" in concept_perf.columns:
                    concept_perf_display = concept_perf[["Concept", "Correct", "Total", "Mastery %", "Avg Hints", "Badge"]]
                else:
                    concept_perf_display = concept_perf[["Concept", "Correct", "Total", "Mastery %", "Badge"]]
                
                st.dataframe(concept_perf_display, use_container_width=True)

            st.divider()

            st.divider()

            # Study Efficiency Metrics
            st.markdown("### ⚡ Study Efficiency Metrics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if not evaluated.empty:
                    efficiency = (evaluated["correct"].sum() / len(evaluated)) * 100
                    st.metric(
                        "Learning Efficiency",
                        f"{efficiency:.1f}%",
                        help="Percentage of first-attempt correct answers"
                    )
            
            with col2:
                if "hints_used" in df.columns:
                    hint_usage = df["hints_used"].sum()
                    st.metric(
                        "Total Hints Used",
                        int(hint_usage),
                        help="Total number of hints across all attempts"
                    )
                else:
                    st.metric(
                        "Total Hints Used",
                        "N/A",
                        help="Total number of hints across all attempts"
                    )
            
            with col3:
                if not evaluated.empty and len(evaluated) > 1:
                    first_half_acc = evaluated.iloc[:len(evaluated)//2]["correct"].mean() * 100
                    second_half_acc = evaluated.iloc[len(evaluated)//2:]["correct"].mean() * 100
                    improvement = second_half_acc - first_half_acc
                    st.metric(
                        "Learning Improvement",
                        f"{improvement:+.1f}%",
                        help="Accuracy improvement from first half to second half"
                    )

            st.divider()

            # Actionable Insights
            st.markdown("### 💭 AI-Generated Insights")
            
            # Show AI status
            config = get_ai_api_config()
            if config["enabled"]:
                st.success("🤖 AI features enabled ✅")
            
            with st.expander("🔍 Detailed Analysis & Recommendations", expanded=True):
                if not evaluated.empty:
                    # Try to get AI insights first
                    ai_insights = None
                    if config["enabled"]:
                        with st.spinner("Generating AI insights..."):
                            performance_data = {
                                "study_days": len(pd.to_datetime(df["timestamp"]).dt.date.unique()) if len(df) > 0 else 0,
                                "total_hints": int(df["hints_used"].sum()) if "hints_used" in df.columns and len(df) > 0 else 0,
                                "accuracy": evaluated["correct"].mean() * 100 if len(evaluated) > 0 else 0
                            }
                            ai_insights = generate_ai_insights(st.session_state.history, performance_data)
                    
                    if ai_insights:
                        st.markdown("**🤖 AI Coach Says:**")
                        st.markdown(f"> {ai_insights}")
                        st.divider()
                    
                    insights_text = ""
                    
                    overall_acc = evaluated["correct"].mean() * 100
                    avg_hints = df["hints_used"].mean() if "hints_used" in df.columns else 0
                    
                    # Generate insights based on performance
                    if overall_acc >= 80:
                        insights_text += f"🌟 **Excellent Performance!** Your {overall_acc:.1f}% accuracy shows strong mastery of the material.\n\n"
                    elif overall_acc >= 60:
                        insights_text += f"📈 **Good Progress!** You're at {overall_acc:.1f}% accuracy. Keep practicing to reach 80%+ mastery.\n\n"
                    else:
                        insights_text += f"🚀 **Keep Going!** You're at {overall_acc:.1f}% accuracy. Review weak concepts and practice more.\n\n"
                    
                    # Hint usage insights
                    if avg_hints < 0.5:
                        insights_text += "✨ **Efficient Learner:** You rarely need hints - great self-assessment skills!\n\n"
                    elif avg_hints < 2:
                        insights_text += "👍 **Balanced Approach:** You use hints strategically to support your learning.\n\n"
                    else:
                        insights_text += "💡 **Hint-Dependent Learning:** Consider attempting questions without hints first to build confidence.\n\n"
                    
                    # Question type analysis
                    if not df[df["question_type"] == "MC"].empty:
                        mc_acc = evaluated[evaluated["question_type"] == "MC"]["correct"].mean() * 100
                        insights_text += f"- **Multiple Choice:** {mc_acc:.1f}% - "
                        insights_text += "Strong!" if mc_acc > 75 else "Needs work - Try to eliminate distractors more carefully.\n"
                    
                    if not df[df["question_type"] == "NUMERIC"].empty:
                        num_acc = evaluated[evaluated["question_type"] == "NUMERIC"]["correct"].mean() * 100
                        insights_text += f"\n- **Numeric:** {num_acc:.1f}% - "
                        insights_text += "Excellent computational skills!" if num_acc > 75 else "Practice calculation accuracy.\n"
                    
                    if not df[df["question_type"] == "REFLECTION"].empty:
                        insights_text += f"\n- **Reflection:** Think deeply about the conceptual understanding behind each topic.\n"
                    
                    st.markdown(insights_text)
                else:
                    st.info("Complete more questions to generate personalized insights")
            
            # AI Question Generator
            if config["enabled"]:
                st.divider()
                st.markdown("### 🚀 Generate AI Questions")
                

                # --- Adaptive Difficulty Suggestion ---
                def suggest_difficulty():
                    df = pd.DataFrame(st.session_state.history)
                    evaluated = df[df.get("evaluated", False)] if not df.empty else pd.DataFrame()
                    if evaluated.empty or len(evaluated) < 5:
                        return "Medium"  # Not enough data, default to Medium
                    acc = evaluated["correct"].mean()
                    avg_hints = evaluated["hints_used"].mean() if "hints_used" in evaluated else 0
                    # If accuracy is high and hints are low, suggest Hard
                    if acc >= 0.85 and avg_hints < 1:
                        return "Hard"
                    # If accuracy is moderate, suggest Medium
                    if acc >= 0.6:
                        return "Medium"
                    # If accuracy is low or hints are high, suggest Easy
                    return "Easy"

                adaptive_difficulty = suggest_difficulty()
                col1, col2, col3 = st.columns(3)
                with col1:
                    concept = st.text_input("Concept to practice:", value="Graph Neural Networks")
                with col2:
                    difficulty = st.selectbox(
                        "Difficulty:", ["Easy", "Medium", "Hard"],
                        index=["Easy", "Medium", "Hard"].index(adaptive_difficulty)
                    )
                    st.caption(f"Suggested: {adaptive_difficulty} (based on your progress)")
                with col3:
                    generate_btn = st.button("Generate Question", key="ai_gen_btn")

                if generate_btn and concept:
                    with st.spinner(f"Creating {difficulty} question on {concept}..."):
                        ai_question = generate_ai_question(concept, difficulty.lower())
                        
                        if ai_question:
                            st.success("✅ Question generated!")
                            st.markdown(f"**Question:** {ai_question.get('question', 'N/A')}")
                            
                            # Display options
                            st.write("**Options:**")
                            for i, opt in enumerate(ai_question.get('options', []), 1):
                                st.write(f"{i}. {opt}")
                            
                            if st.checkbox("Show explanation"):
                                st.info(f"**Explanation:** {ai_question.get('explanation', 'N/A')}")
                        else:
                            st.warning("Could not generate question. Check API configuration.")


if __name__ == "__main__":
    main()
