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

**Key Classes & Functions**:
- `get_ai_api_config()` - Load API configuration
- `call_ai_api()` - Universal AI API wrapper
- `generate_ai_question()` - Generate single question
- `generate_ai_insights()` - Generate coaching feedback
- `init_session_state()` - Initialize Streamlit session
- `get_performance_icon()` - Performance emoji
- `display_mastery_badge()` - Mastery visualization
- `main()` - Main Streamlit app

#### 2. **ontology_engine.py** (~180 lines) - Ontology Wrapper
```
┌─ Class: OntologyEngine
│  ├─ __init__(path) - Load OWL file
│  ├─ list_objectives() - Get all learning objectives
│  ├─ objective_info() - Get objective details
│  ├─ tasks_for_objective() - Get tasks for objective
│  ├─ task_info() - Get task details
│  ├─ assessments_for_objective() - Get assessments
│  └─ describe_concept() - Get concept metadata
│
└─ Data Classes
   ├─ ObjectiveInfo
   ├─ TaskInfo
   └─ AssessmentInfo
```

#### 3. **question_bank.py** (~120 lines) - Static Questions
```
QUESTIONS = {
  "Q1_adj_matrix_mc": {
    "objective_iri": "...",
    "type": "MC",
    "prompt": "...",
    "mc_choices": [...],
    "hints": [...]
  },
  "Q2_adj_matrix_numeric": {...},
  "Q3_gcn_dims_numeric": {...},
  "Q4_message_passing_mc": {...},
  "Q5_eval_reflection": {...}
}
```

#### 4. **ont.rdf** (~1,000+ lines) - OWL Ontology
Contains:
- Learning Objectives
- Learning Tasks
- Concepts (GNNConcept, GraphDataset, etc.)
- Assessments
- Property definitions

#### 5. **requirements.txt** - Dependencies
```
streamlit==latest          # Web UI
owlready2==latest          # OWL loading
pandas==latest             # Data processing
plotly==latest             # Visualizations
Pillow==latest             # Image handling
openai==latest             # OpenAI API
anthropic==latest          # Anthropic API
google-generativeai==latest # Google API
python-dotenv==latest      # .env loading
```

#### 6. **.env** - Configuration
```
OPENAI_API_KEY=sk-proj-...
```

---

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

### Error Handling

```python
try:
    response = client.chat.completions.create(...)
    return response.choices[0].message.content
except Exception as e:
    st.warning(f"AI API error: {str(e)}")
    return None
```

**Graceful Degradation**:
- ❌ No API key → Shows "AI features disabled"
- ❌ API error → Shows warning, continues with fallback
- ❌ Invalid response → Returns None, uses static questions

### Prompt Engineering

**Question Generation Prompt**:
```
Generate {n} unique and diverse questions (mix of MC and Theory) 
covering all key concepts, applications, challenges, and recent 
advances in the topic: '{topic}'. 

Questions must be plausible, clear, and suitable for a human learner. 
Avoid technical names like 'ObjTrainGCNModel' and use conceptual language. 
Each question should be different and not repeated. 

Format: For MC, provide question, 4 options, and correct index. 
For Theory, provide question only. 

Difficulty: {difficulty}.
```

**Insight Generation Prompt**:
```
As an expert learning coach, provide personalized learning insights 
based on this student data:

- Total questions attempted: {total_questions}
- Correct answers: {correct}
- Accuracy: {accuracy:.1f}%
- Study sessions: {study_days}
- Hints used: {total_hints}

Provide:
1. One key strength to celebrate
2. One area for improvement
3. One specific action to take next

Keep response concise (3-4 sentences) and motivating.
```

---

## User Interface

### Layout & Design

**Color Palette**:
```
Primary:      #ff9800 (Deep Orange)
Secondary:    #fff    (White)
Light:        #fff3e0 (Light Peach)
Accent:       #ffe0b2 (Pale Orange)
Success:      #10b981 (Green)
Error:        #ef4444 (Red)
```

**Theme**:
- Orange gradient headers
- White cards with orange shadows
- Smooth animations on buttons
- Responsive 12px grid gaps

### Five-Tab Interface

#### Tab 1: 📚 Overview (Objective Selection)

```
┌─────────────────────────────────────┐
│  🎯 Choose a learning objective     │
├─────────────────────────────────────┤
│                                     │
│  Dropdown: [Learning Objective ▼]   │
│                                     │
│  #### Graph Representation Basics   │
│  Learn how graphs are structured    │
│  and represented computationally.   │
│                                     │
│  Difficulty Level: Beginner         │
│                                     │
│  [🚀 Start / switch to this...]     │
│                                     │
│  ─────────────────────────────────  │
│                                     │
│  Related assessments in the...      │
│  ▶ Assessment: Graph Fundamentals   │
│                                     │
└─────────────────────────────────────┘
```

**Functionality**:
- Dropdown to select objective (no auto-selection)
- Displays objective details (name, description, level)
- Shows related assessments and concepts
- "Start/switch" button updates `current_objective_iri`
- Success message: "✅ Objective updated! Now, click Learn tab..."

#### Tab 2: 📖 Learn (Task Exploration)

```
┌─────────────────────────────────────┐
│  ✅ Learn tab opened!               │
│                                     │
│  📚 Ontology-driven learning tasks  │
├─────────────────────────────────────┤
│                                     │
│  Search: [____________________________] 
│                                     │
│  Available tasks: [Select task ▼]   │
│                                     │
│  #### Explain Adjacency Matrix      │
│  Learn to represent graphs as...    │
│                                     │
│  📈 Difficulty: Beginner            │
│  ⏱️ Est. time: 15 minutes          │
│  💻 Requires coding?: No            │
│                                     │
│  🔗 Linked GNN concepts             │
│  [BasicGraphRepresentation]         │
│                                     │
│  📊 Graph datasets used             │
│  📁 **Cora** – 2,708 graphs         │
│                                     │
│  🕸️ Example graph instances         │
│  🔷 **Citeseer** – 3,327 nodes...   │
│                                     │
└─────────────────────────────────────┘
```

**Functionality**:
- Search bar filters tasks by name
- Dropdown shows all tasks for selected objective
- Task metadata (difficulty, time, coding)
- Linked concepts (info boxes)
- Datasets and graph instances
- Success message: "✅ Learn tab opened! Click Practice tab..."

#### Tab 3: ✍️ Practice (Question Generation & Answering)

```
┌─────────────────────────────────────┐
│  ✅ Practice tab opened!            │
│                                     │
│  ✍️ Practice Session                │
├─────────────────────────────────────┤
│                                     │
│  How many questions do you want?    │
│  [10] (1-50)                        │
│                                     │
│  Concept to practice:               │
│  [Graph Neural Networks          ]  │
│  (Based on your learning objective) │
│                                     │
│  Difficulty: [Medium          ▼]    │
│                                     │
│  [Start Practice Session]           │
│                                     │
│  ─────────────────────────────────  │
│                                     │
│  Question 1 of 10                   │
│                                     │
│  What is an adjacency matrix?       │
│                                     │
│  ◉ A matrix where each entry...    │
│  ○ A matrix that stores labels...  │
│  ○ A method for training models...  │
│  ○ A type of neural network...      │
│                                     │
│  [Submit Answer]                    │
│                                     │
│  ─────────────────────────────────  │
│                                     │
│  🎉 Practice session complete!      │
│                                     │
│  MC Questions: 8/10 correct         │
│  Theory Questions: 2/2 marks        │
│                                     │
│  [Restart Practice Session]         │
│                                     │
└─────────────────────────────────────┘
```

**Workflow**:
1. User selects # questions (1-50)
2. Concept auto-filled from objective (user can override)
3. Difficulty selected (Easy/Medium/Hard)
4. Click "Start Practice Session"
5. Questions generated via AI
6. User answers each question
7. Results summary displayed
8. History saved for analytics

**Question Types**:
- **MC**: 4 options, radio buttons, auto-graded
- **THEORY**: Text area, AI auto-graded, 0-1 mark

#### Tab 4: 📊 Progress (Analytics)

```
┌─────────────────────────────────────┐
│  ✅ Progress tab opened!            │
│                                     │
│  📊 Learner progress & analytics    │
├─────────────────────────────────────┤
│                                     │
│  📈 Overall performance             │
│                                     │
│  🌟 Overall accuracy      │  78.5%  │
│  📝 Total attempts        │   26    │
│  🎯 Correct answers       │   20    │
│  🔥 Current streak        │   3     │
│                                     │
│  ─────────────────────────────────  │
│                                     │
│  📊 Accuracy by objective           │
│                                     │
│  [Bar Chart - RdYlGn scale]         │
│                                     │
│  Graph Rep        85.0%             │
│  GCN Training     72.3%             │
│  Message Passing  68.5%             │
│                                     │
│  ─────────────────────────────────  │
│                                     │
│  🔬 Attempts by concept             │
│                                     │
│  [Bar Chart - Blue scale]           │
│                                     │
│  BasicRepresentation    12          │
│  GCNFundamentals         8          │
│  TrainingWorkflow        6          │
│                                     │
└─────────────────────────────────────┘
```

**Metrics Displayed**:
- Overall stats (accuracy, attempts, streak)
- Accuracy by objective (Plotly bar chart)
- Attempts by concept (Plotly bar chart)
- Concept mastery with badges

#### Tab 5: 💡 Insights (AI Analysis)

```
┌─────────────────────────────────────┐
│  💡 AI-Powered Insights             │
├─────────────────────────────────────┤
│                                     │
│  📊 Key Metrics & Dynamic Visuals   │
│                                     │
│  Total Questions    │  26  Avg Hints   │  0.62     │
│  Accuracy (%)       │  78.5%  Study Days       │  4        │
│                                     │
│  ─────────────────────────────────  │
│                                     │
│  🧠 Concept Mastery Analysis        │
│                                     │
│  BasicGraphRep   10/12  83.3% 🏆   │
│  GCNFundamentals  6/8   75.0% ⭐   │
│  MessagePassing   4/6   66.7% 👍   │
│                                     │
│  ─────────────────────────────────  │
│                                     │
│  ⚡ Study Efficiency Metrics        │
│                                     │
│  Learning Efficiency    │  65.4%     │
│  Total Hints Used       │  16        │
│  Learning Improvement   │  +12.3%    │
│                                     │
│  ─────────────────────────────────  │
│                                     │
│  💭 AI-Generated Insights           │
│                                     │
│  > 🌟 Excellent Performance!        │
│    Your 78.5% accuracy shows strong │
│    mastery of the material...       │
│                                     │
│  > 💡 Hint-Dependent Learning:     │
│    Consider attempting questions    │
│    without hints first...           │
│                                     │
│  ─────────────────────────────────  │
│                                     │
│  🚀 Generate AI Questions           │
│                                     │
│  Concept: [Graph Neural Networks ]  │
│  Difficulty: [Medium        ▼]      │
│  (Suggested: Medium - based on...)  │
│  [Generate Question]                │
│                                     │
│  ✅ Question generated!             │
│  **Question**: What is backprop...  │
│  **Options**: ...                   │
│                                     │
└─────────────────────────────────────┘
```

**Features**:
- Key metrics: total questions, accuracy, hints, study days, streak
- Concept mastery analysis with colored badges
- Study efficiency metrics (efficiency %, hints, improvement)
- AI coach insights with performance-based messages
- Adaptive difficulty suggestion
- Single question generator for on-demand practice

---

## Data Flow

### Complete User Journey

```
START
  │
  ├─ App initializes
  │  ├─ Load OWL ontology
  │  ├─ Initialize session state
  │  └─ Load configuration
  │
  ├─ User lands on app
  │  └─ Sees 5 tabs
  │
  ├─ Tab 1: Overview
  │  ├─ User selects objective
  │  │  └─ Sets: current_objective_iri
  │  ├─ Displays objective info
  │  └─ Clicks "Start/switch"
  │     └─ Session state updated
  │
  ├─ Tab 2: Learn
  │  ├─ Loads tasks for objective
  │  ├─ User searches/selects task
  │  │  └─ Sets: current_task_iri
  │  └─ Displays task details
  │     ├─ Concepts
  │     ├─ Datasets
  │     └─ Graph instances
  │
  ├─ Tab 3: Practice
  │  ├─ User sets parameters
  │  │  ├─ # questions (1-50)
  │  │  ├─ Concept (auto-filled)
  │  │  └─ Difficulty
  │  ├─ Clicks "Start Session"
  │  │  ├─ Calls generate_practice_questions()
  │  │  ├─ Sends prompt to OpenAI
  │  │  ├─ Parses response
  │  │  ├─ Filters implausible Qs
  │  │  └─ Stores in session_state
  │  ├─ User answers questions
  │  │  ├─ MC: Select option
  │  │  ├─ THEORY: Type answer
  │  │  └─ Submit (auto-grade)
  │  ├─ Progresses through Qs
  │  │  └─ Updates progress_current_idx
  │  ├─ Session complete
  │  │  ├─ Shows results summary
  │  │  ├─ Stores in history
  │  │  └─ Updates session state
  │  └─ Optional: Restart session
  │
  ├─ Tab 4: Progress
  │  ├─ Loads history from session
  │  ├─ Calculates metrics
  │  │  ├─ Accuracy by objective
  │  │  ├─ Attempts by concept
  │  │  └─ Performance breakdown
  │  ├─ Generates Plotly charts
  │  └─ Displays with badges
  │
  ├─ Tab 5: Insights
  │  ├─ Loads history
  │  ├─ Calculates study metrics
  │  ├─ Calls generate_ai_insights()
  │  │  ├─ Sends performance data to AI
  │  │  ├─ Gets coaching feedback
  │  │  └─ Displays AI message
  │  ├─ Shows concept mastery analysis
  │  ├─ Suggests adaptive difficulty
  │  └─ Optional: Generate single question
  │
  └─ Optional: Return to Tab 1 for next objective
     └─ LOOP
```

### Session State Lifecycle

```
┌─────────────────────────────────────────────────┐
│         Session State Initialization            │
│  (Happens once when app starts)                 │
├─────────────────────────────────────────────────┤
│                                                 │
│  student_iri: "StudentAdvanced01"               │
│  current_objective_iri: None                    │
│  current_task_iri: None                         │
│  hint_level: 0                                  │
│  history: []                                    │
│  practice_questions: []                         │
│  practice_current_idx: 0                        │
│  practice_answers: []                           │
│  practice_started: False                        │
│  practice_complete: False                       │
│                                                 │
└─────────────────────────────────────────────────┘
         │
         ├─ User selects objective
         │  └─ current_objective_iri = "http://...ObjUnderstandGraphRep"
         │
         ├─ User selects task
         │  └─ current_task_iri = "http://...ExplainAdjacencyMatrixConcept"
         │
         ├─ User starts practice
         │  ├─ practice_questions = [Q1, Q2, ..., Q10]
         │  ├─ practice_current_idx = 0
         │  └─ practice_started = True
         │
         ├─ User answers Q1
         │  ├─ practice_answers = [{q1_response}]
         │  ├─ history = [{q1_data}]
         │  └─ practice_current_idx = 1
         │
         ├─ ... answers Q2 through Q10 ...
         │
         ├─ User completes session
         │  ├─ practice_complete = True
         │  └─ history = [all_q_data]
         │
         ├─ User clicks Restart
         │  ├─ practice_questions = []
         │  ├─ practice_started = False
         │  └─ practice_complete = False
         │     (history persists for analytics)
         │
         └─ Session persists across:
            ├─ Tab switches
            ├─ Browser refreshes (within session)
            └─ App reruns
```

---

## Installation & Setup

### Prerequisites

```bash
# Check Python version
python3 --version  # Should be 3.8+

# Check pip
pip3 --version
```

### Step 1: Clone/Navigate to Project

```bash
cd /Users/mubaraq/Documents/AIC
ls -la  # Verify files exist
```

### Step 2: Create Virtual Environment

```bash
# Create venv
python3 -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt

# Verify installation
pip list | grep streamlit
```

### Step 4: Configure API Key

```bash
# Option A: Create .env file
echo "OPENAI_API_KEY=sk-proj-..." > .env

# Option B: Edit .env in text editor
nano .env
# Add: OPENAI_API_KEY=sk-proj-...

# Option C: Set environment variable
export OPENAI_API_KEY=sk-proj-...
```

### Step 5: Verify Ontology

```bash
# Check ont.rdf exists
ls -lh ont.rdf

# File should be ~1MB+ in size
```

### Step 6: Run Application

```bash
streamlit run app.py

# Output should show:
# You can now view your Streamlit app in your browser.
# Local URL: http://localhost:8501
```

### Step 7: Access in Browser

```
Open: http://localhost:8501
```

### Troubleshooting Installation

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'streamlit'` | Run `pip install -r requirements.txt` |
| `API key not found` | Create `.env` file with `OPENAI_API_KEY=sk-...` |
| `ontology not found` | Verify `ont.rdf` exists in project directory |
| `Port 8501 already in use` | Use `streamlit run app.py --server.port 8502` |
| `Python version error` | Ensure Python 3.8+ (`python3 --version`) |

---

## API Reference

### Session State API

```python
# Get current objective
obj_iri = st.session_state.current_objective_iri
if obj_iri:
    obj = engine.get_objective_by_iri(obj_iri)

# Get history
history_df = pd.DataFrame(st.session_state.history)
accuracy = history_df[history_df["evaluated"]]["correct"].mean()

# Access session variables
st.session_state.practice_questions  # Generated questions
st.session_state.practice_current_idx  # Current Q position
st.session_state.hint_level  # Escalation level
```

### Ontology Engine API

```python
# Initialize engine
engine = OntologyEngine("ont.rdf")

# List objectives
objectives = engine.list_objectives()
# Returns: List[ObjectiveInfo]

# Get objective details
obj = engine.get_objective_by_iri(iri)
info = engine.objective_info(obj)
# Returns: ObjectiveInfo(iri, name, description, level)

# Get tasks for objective
tasks = engine.tasks_for_objective(obj)
# Returns: List[TaskInfo]

# Get task details
task_info = engine.task_info(task)
# Returns: TaskInfo(...)

# Get assessments for objective
assessments = engine.assessments_for_objective(obj)
# Returns: List[AssessmentInfo]

# Describe concept
concept = engine.describe_concept(concept_iri)
# Returns: {
#     "iri": "...",
#     "name": "BasicGraphRepresentation",
#     "kind": "GNNConcept",
#     "details": {...}
# }
```

### AI API

```python
# Get configuration
config = get_ai_api_config()
# Returns: {"provider": "openai", "api_key": "sk-...", "enabled": True}

# Call AI API
response = call_ai_api(prompt, max_tokens=500)
# Returns: Generated text or None

# Generate question
question = generate_ai_question("Graph Neural Networks", "medium")
# Returns: {
#     "question": "...",
#     "options": [...],
#     "correct_idx": 0,
#     "explanation": "..."
# }

# Generate insights
insights = generate_ai_insights(history, performance_data)
# Returns: "Your 78.5% accuracy shows..."
```

---

## Troubleshooting

### Common Issues

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
