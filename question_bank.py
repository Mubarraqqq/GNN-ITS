# question_bank.py

# -----------------------------
# Define ALL IRIs *before* using them
# -----------------------------

OBJ_UNDERSTAND_GRAPH_REP = "http://www.co-ode.org/ontologies/ont.owl#ObjUnderstandGraphRep"
OBJ_TRAIN_GCN = "http://www.co-ode.org/ontologies/ont.owl#ObjTrainGCNModel"
OBJ_EVAL_GRAPH_ACC = "http://www.co-ode.org/ontologies/ont.owl#ObjEvaluateGraphAccuracy"
OBJ_IMPLEMENT_MESSAGE_PASSING = "http://www.co-ode.org/ontologies/ont.owl#ObjImplementMessagePassing"

TASK_EXPLAIN_ADJ = "http://www.co-ode.org/ontologies/ont.owl#ExplainAdjacencyMatrixConcept"
TASK_INTERACTIVE_AGG = "http://www.co-ode.org/ontologies/ont.owl#InteractiveNodeAggregationExercise"
TASK_WORKED_GCN = "http://www.co-ode.org/ontologies/ont.owl#WorkedExampleGCNForwardPass"

CONCEPT_BASIC_REP = "http://www.co-ode.org/ontologies/ont.owl#BasicGraphRepresentation"
CONCEPT_GCN_FUND = "http://www.co-ode.org/ontologies/ont.owl#GCNLayerFundamentals"
CONCEPT_TRAIN_WORKFLOW = "http://www.co-ode.org/ontologies/ont.owl#TrainingWorkflowConcept"


# -----------------------------
# Question definitions
# -----------------------------

QUESTIONS = {
    "Q1_adj_matrix_mc": {
        "objective_iri": OBJ_UNDERSTAND_GRAPH_REP,
        "task_iri": TASK_EXPLAIN_ADJ,
        "concept_iri": CONCEPT_BASIC_REP,
        "type": "MC",
        "prompt": (
            "In an adjacency matrix for an *unweighted* directed graph, "
            "what does a value of **1** at position (i, j) represent?"
        ),
        "mc_choices": [
            {"id": "A", "text": "There is an edge from node i to node j", "correct": True},
            {"id": "B", "text": "Nodes i and j have the same degree", "correct": False},
            {"id": "C", "text": "There is a path of length 2 between i and j", "correct": False},
            {"id": "D", "text": "The graph has exactly one connected component", "correct": False},
        ],
        "hints": [
            "Think about how we encode the *presence* or *absence* of edges in a matrix.",
            "Look at row i and column j: what relationship between those nodes are we recording?",
            "If an edge exists directly from i to j, what value should we store at (i, j)?",
        ],
    },

    "Q2_adj_matrix_numeric": {
        "objective_iri": OBJ_UNDERSTAND_GRAPH_REP,
        "task_iri": TASK_EXPLAIN_ADJ,
        "concept_iri": CONCEPT_BASIC_REP,
        "type": "NUMERIC",
        "prompt": (
            "A simple undirected graph has **5** nodes with edges forming a chain "
            "(1–2–3–4–5). How many non-zero entries will its adjacency matrix have?"
        ),
        "numeric_answer": 8,
        "numeric_tolerance": 0.0001,
        "hints": [
            "For an undirected graph, each edge appears twice in the adjacency matrix.",
            "Count how many edges a 5-node chain has.",
            "Multiply the number of edges by 2 to get the number of non-zero entries.",
        ],
    },

    "Q3_gcn_dims_numeric": {
        "objective_iri": OBJ_TRAIN_GCN,
        "task_iri": TASK_WORKED_GCN,
        "concept_iri": CONCEPT_GCN_FUND,
        "type": "NUMERIC",
        "prompt": (
            "A GCN layer takes a node feature matrix X of shape (N, 1433) and outputs "
            "a hidden representation of size 64. How many parameters are in W?"
        ),
        "numeric_answer": 1433 * 64,
        "numeric_tolerance": 1e-6,
        "hints": [
            "X has 1433 input features and outputs 64 features.",
            "A linear layer maps 1433 inputs → 64 outputs.",
            "The parameter count is 1433 × 64.",
        ],
    },

    "Q4_message_passing_mc": {
        "objective_iri": OBJ_IMPLEMENT_MESSAGE_PASSING,
        "task_iri": TASK_INTERACTIVE_AGG,
        "concept_iri": CONCEPT_GCN_FUND,
        "type": "MC",
        "prompt": (
            "In a typical message passing step of a GNN, what does the aggregation function do?"
        ),
        "mc_choices": [
            {"id": "A", "text": "It updates the graph labels", "correct": False},
            {"id": "B", "text": "It combines information from a node's neighbors", "correct": True},
            {"id": "C", "text": "It trains the model using gradient descent", "correct": False},
            {"id": "D", "text": "It converts graphs into images", "correct": False},
        ],
        "hints": [
            "Messages come from neighboring nodes.",
            "What must a node do with multiple incoming messages?",
            "Aggregation pools or combines all neighbor messages.",
        ],
    },

    "Q5_eval_reflection": {
        "objective_iri": OBJ_EVAL_GRAPH_ACC,
        "task_iri": "http://www.co-ode.org/ontologies/ont.owl#ReflectionTaskOnModelPerformance",
        "concept_iri": CONCEPT_TRAIN_WORKFLOW,
        "type": "REFLECTION",
        "prompt": (
            "Your GNN achieves 99% accuracy on the training set but 65% on validation.\n"
            "Explain in 2–3 sentences what this suggests and what you should do."
        ),
        "hints": [
            "Training accuracy is high, validation accuracy is low — what does that imply?",
            "Think of overfitting: memorising training data but not generalising.",
            "Consider regularisation, early stopping, reducing model complexity, etc.",
        ],
    },
}
