# 🧠 GNN Intelligent Tutoring System - Complete Documentation

> **Last Updated**: December 13, 2025
> **Status**: Production Ready
> **Version**: 1.0

---

## 📚 Table of Contents

1. [Project Overview](#project-overview)
2. [Quick Start](#quick-start)
3. [Architecture & Components](#architecture--components)
4. [Core Features](#core-features)
5. [AI Integration](#ai-integration)
6. [User Interface](#user-interface)
7. [Data Flow](#data-flow)
8. [Installation & Setup](#installation--setup)
9. [API Reference](#api-reference)
10. [Troubleshooting](#troubleshooting)


---

## Project Overview

### What is This?

The **GNN Intelligent Tutoring System (ITS)** is an advanced, ontology-driven educational platform designed to teach Graph Neural Networks through:

- **Adaptive Learning Paths**: Learning objectives structured in OWL ontology
- **AI-Generated Questions**: Personalized practice questions generated via OpenAI/Claude/Gemini
- **Real-Time Analytics**: Progress tracking with concept mastery analysis
- **Intelligent Feedback**: AI-powered insights and difficulty adaptation
- **Modern Web UI**: Beautiful, responsive Streamlit interface with orange/white theme

### Target Audience

- 👨‍🎓 Computer Science / ML students learning GNNs
- 🤖 Self-paced learners seeking personalized tutoring
- 👨‍🏫 Educators building ontology-based curricula

### Key Statistics

| Metric | Value |
|--------|-------|
| **Total Lines of Code** | ~1,662 |
| **Main App** | 1,362 lines (app.py) |
| **Ontology Engine** | ~180 lines |
| **Question Bank** | ~120 lines |
| **Supported AI Providers** | 4 (OpenAI, GitHub, Anthropic, Google) |
| **Question Types** | 4 (MC, Numeric, Theory, Reflection) |
| **UI Tabs** | 5 (Overview, Learn, Practice, Progress, Insights) |

---

## Quick Start

### For the Impatient (5 minutes)

```bash
# 1. Clone/navigate to project
cd /Users/mubaraq/Documents/AIC

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API key
echo "OPENAI_API_KEY=sk-..." > .env

# 5. Run it!
streamlit run app.py

# 6. Open browser
# Local: http://localhost:8501
```

### First-Time User Flow

```
1️⃣  Overview Tab → Select a learning objective
2️⃣  Learn Tab → Explore tasks for that objective
3️⃣  Practice Tab → Generate & answer AI questions
4️⃣  Progress Tab → View your analytics
5️⃣  Insights Tab → Get AI-powered recommendations
6️⃣  Loop back to step 1 for next objective
```

---

## Architecture & Components

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────┐
│               Streamlit Web Interface               │
│        (5 Tabs: Overview, Learn, Practice, etc.)    │
└────────────────┬────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────┐
│        Business Logic Layer (app.py)                │
│  ┌──────────────────────────────────────────────┐   │
│  │ • AI API Integration (4 providers)           │   │
│  │ • Question Generation & Grading              │   │
│  │ • Session State Management                   │   │
│  │ • Analytics Calculations                     │   │
│  └──────────────────────────────────────────────┘   │
└────────────────┬────────────────────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
┌───────▼──────┐   ┌─────▼──────────────┐
│ Ontology     │   │ Question Bank      │
│ Engine       │   │ (question_bank.py) │
│ (ontology_   │   │                    │
│  engine.py)  │   │ • Static Qs        │
│              │   │ • Q Meta-data      │
│ • List Objs  │   │ • Q Hints          │
│ • Get Tasks  │   └────────────────────┘
│ • Describe   │
│   Concepts   │
└───────┬──────┘
        │
┌───────▼──────────────────┐
│   Knowledge Base         │
│  • OWL Ontology (RDF)    │
│  • Learning Objectives   │
│  • Tasks & Concepts      │
│  • Assessments           │
└──────────────────────────┘
```

### Core Components

#### 1. **app.py** (1,362 lines) - Main Application
```
┌─ AI Configuration (Lines 20-90)
├─ AI API Wrapper (Lines 33-87)
├─ Question Generation (Lines 91-120)
├─ Session Helpers (Lines 156-201)
├─ UI Helpers (Lines 205-304)
└─ Main Function (Lines 307-1,362)
   ├─ Page Config
   ├─ Custom Styling
   ├─ Tab 1: Overview
   ├─ Tab 2: Learn
   ├─ Tab 3: Practice
   ├─ Tab 4: Progress
   └─ Tab 5: Insights
```

## Core Features

### 1. **Ontology-Driven Learning Structure**

The system uses an OWL ontology to define:

```
Learning Objective (e.g., "Understand Graph Representation")
  ├─ Has Learning Tasks (e.g., "Explain Adjacency Matrix")
  │   ├─ Teaches Concepts (e.g., "BasicGraphRepresentation")
  │   └─ Uses Resources (Datasets, Graph Instances)
  └─ Has Assessments (e.g., "Graph Fundamentals Quiz")
      └─ Requires Concepts
```

**Benefits**:
- Structured, maintainable learning paths
- No hardcoded content
- Easy to extend with new domains
- Semantic relationships preserved

### 2. **AI-Generated Practice Questions**

**Generation Pipeline**:

```
User Input
  ├─ Concept (auto-filled from objective)
  ├─ Difficulty (Easy/Medium/Hard)
  └─ Number of questions (1-50)
  
         ↓

Prompt Engineering
  ├─ Context from learning objective
  ├─ Difficulty level specified
  ├─ Request diverse question types
  └─ Ask for plausible content

         ↓

OpenAI API Call
  ├─ Model: GPT-3.5-Turbo
  ├─ Max tokens: 1,200
  └─ Temperature: 0.7

         ↓

Question Parsing
  ├─ Extract MC questions (question + 4 options)
  ├─ Extract Theory questions
  └─ Remove implausible content (technical names)

         ↓

Storage in Session State
  ├─ practice_questions: List[Dict]
  └─ practice_current_idx: int
```

**Example Generated Question**:

```json
{
  "question": "What does aggregation in message passing do?",
  "type": "MC",
  "options": [
    "Combines neighbor node information",
    "Trains the neural network",
    "Updates graph labels",
    "Converts graphs to images"
  ],
  "correct_idx": 0
}
```

**Plausibility Filtering**:

```python
# Filter out technical names
if re.match(r'Obj[A-Z][a-zA-Z0-9]+', block):
    continue  # Skip this question

# Example rejection:
# "How does ObjTrainGCNModel..." → REJECTED
# Replacement: "How do GCN models..." → ACCEPTED
```

### 3. **Intelligent Auto-Grading**

**For Multiple Choice**:
```python
user_choice_idx = 2
correct_idx = 0
correct = (user_choice_idx == correct_idx)  # False
```

**For Numeric**:
```python
user_answer = 92
expected_answer = 100
tolerance = 5
correct = abs(user_answer - expected_answer) <= tolerance  # True
```

**For Theory (Open-Ended)**:
```
1. Student submits answer
   ↓
2. AI generates reference answer
   ↓
3. AI compares for semantic similarity
   ↓
4. AI assigns 0 or 1 mark
   ↓
5. Grade stored with explanation
```

### 4. **Progress Analytics**

**Metrics Tracked Per Question**:
```python
{
    "question": str,
    "type": "MC" | "NUMERIC" | "THEORY",
    "correct": bool | int (mark),
    "user_answer": str,
    "options": list,
    "evaluated": True,
    "timestamp": pd.Timestamp,
    "concept_iri": str,
    "objective_iri": str,
    "question_type": str
}
```

**Aggregate Metrics Calculated**:
| Metric | Formula | Display |
|--------|---------|---------|
| Overall Accuracy | correct / total * 100 | % |
| Current Streak | Count of consecutive correct from end | # |
| Study Days | Count of unique dates with practice | # |
| Concept Mastery | Accuracy per concept | % with badge |
| Learning Efficiency | First-attempt correct / total * 100 | % |
| Learning Improvement | 2nd_half_acc - 1st_half_acc | ± % |
| Avg Hints Used | Sum of hints / total questions | # |

**Visualizations**:
- Bar chart: Accuracy by objective (RdYlGn color scale)
- Bar chart: Attempts by concept
- Mastery badges: 🏆 Master, ⭐ Expert, 👍 Proficient, 📚 Learning, 🌱 Beginner
- Line chart: Accuracy trend over time

### 5. **Adaptive Difficulty System**

**Logic**:
```python
def suggest_difficulty():
    if accuracy >= 0.85 AND avg_hints < 1:
        return "Hard"      # User is excelling
    elif accuracy >= 0.60:
        return "Medium"    # User is progressing
    else:
        return "Easy"      # User needs support
```

**Behavior**:
- System analyzes last 5+ questions
- Suggests appropriate difficulty
- User can override suggestion
- Suggestion updates dynamically

### 6. **Session State Management**

**Persistent Variables**:
```python
st.session_state = {
    "student_iri": "StudentAdvanced01",
    "current_objective_iri": None,  # Set in Overview tab
    "current_task_iri": None,       # Set in Learn tab
    "history": [],                  # Populated by Practice tab
    "practice_questions": [],       # Generated questions
    "practice_current_idx": 0,      # Current question position
    "practice_answers": [],         # User responses
    "practice_started": False,      # Session flag
    "practice_complete": False,     # Completion flag
}
```

**Persistence Across**:
- Tab switches
- Browser refreshes (within session)
- Button clicks

**Reset When**:
- User clicks "Restart Practice Session"
- User selects new objective

---

## AI Integration

### Multi-Provider Architecture

```
        ┌─ OpenAI (gpt-3.5-turbo) ← DEFAULT
        │   ├─ API: api.openai.com
        │   └─ Use: Question generation, grading
        │
User Code ─┼─ GitHub/Azure (gpt-4o)
        │   ├─ API: models.inference.ai.azure.com
        │   └─ Use: High-quality alternative
        │
        ├─ Anthropic (claude-3.5-sonnet)
        │   ├─ API: api.anthropic.com
        │   └─ Use: Advanced reasoning
        │
        └─ Google (gemini-pro)
            ├─ API: googleapis.com
            └─ Use: Multimodal (future)
```

### API Configuration Flow

```python
# 1. Load configuration
config = get_ai_api_config()
# Returns: {
#     "provider": "openai",
#     "api_key": "sk-...",
#     "enabled": True
# }

# 2. Make API call
response = call_ai_api(prompt, max_tokens=500)
# Returns: Generated text or None

# 3. Parse response
questions = parse_questions(response)
```



---

## User Interface




#### 1. App Won't Start

**Error**: `StreamlitAPIException: ...`

**Solutions**:
```bash
# Clear Streamlit cache
rm -rf ~/.streamlit

# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Run with verbose output
streamlit run app.py --logger.level=debug
```

#### 2. API Key Not Working

**Error**: `AI API error: Invalid API key...`

**Solutions**:
```bash
# Verify key format
echo $OPENAI_API_KEY  # Should start with sk-

# Regenerate key at https://platform.openai.com/api-keys

# Update .env
OPENAI_API_KEY=sk-new-key-here
```

#### 3. Questions Not Generating

**Error**: `Could not generate question...`

**Causes & Solutions**:
| Cause | Solution |
|-------|----------|
| No API key | Add `OPENAI_API_KEY` to `.env` |
| API limit exceeded | Wait 60 seconds, try again |
| Invalid concept | Use default or select objective first |
| Network issue | Check internet connection |

#### 4. Ontology Loading Error

**Error**: `ontology_engine.py: FileNotFoundError: ont.rdf`

**Solution**:
```bash
# Verify file exists
ls -lh ont.rdf

# If missing, restore from backup
cp ont.rdf.backup ont.rdf
```

#### 5. Browser Access Issues

**Error**: `Connection refused` or `localhost:8501 unreachable`

**Solution**:
```bash
# Verify Streamlit is running
ps aux | grep streamlit

# Kill and restart
pkill streamlit
streamlit run app.py

# Try different port
streamlit run app.py --server.port 8502
```

---



## Summary

The **GNN Intelligent Tutoring System** is a modern, AI-powered educational platform that:

✅ **Structures Learning**: Uses OWL ontology for scalable curriculum design  
✅ **Personalizes Practice**: Generates unique questions adapted to each student  
✅ **Tracks Progress**: Real-time analytics with concept mastery analysis  
✅ **Adapts Difficulty**: Intelligent suggestions based on performance  
✅ **Provides Insights**: AI coaching and personalized recommendations  
✅ **Ensures Accessibility**: Beautiful, responsive UI for any device  

**Next Steps**:
1. 🚀 Run the app: `streamlit run app.py`
2. 📖 Select a learning objective in the Overview tab
3. 🎓 Explore tasks in the Learn tab
4. ✍️ Practice with AI-generated questions
5. 📊 Track your progress and get insights

**Questions?** Check the documentation files or review the source code!

---

**Last Updated**: December 13, 2025  
**Version**: 1.0 Production  
**Status**: ✅ Ready for Use
