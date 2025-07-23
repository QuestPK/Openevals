"""
OpenEvals RAG Evaluation Examples with your custom inputs and outputs

Demonstrates evaluation of RAG pipeline outputs using OpenEvals.
Includes:
- Correctness
- Helpfulness
- Retrieval Relevance
- Groundedness

Replace inputs, outputs, and context with your own data.
"""

import json
from openevals.llm import create_llm_as_judge
from openevals.prompts import (
    CORRECTNESS_PROMPT,
    RAG_HELPFULNESS_PROMPT,
    RAG_RETRIEVAL_RELEVANCE_PROMPT,
    RAG_GROUNDEDNESS_PROMPT,
)

# Load standard or reference document
with open("path/to/your-standard.json", "r", encoding="utf-8") as f:
    standard_context = json.load(f)



# ------------------- Correctness -------------------
# Measures how accurately the output matches a reference (ground-truth) answer.
correctness_evaluator = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    feedback_key="correctness",
    model="openai:o3-mini",
)

correctness_result = correctness_evaluator(
    inputs="Does the finding comply with clause 6.2.5 about IATA training for pathology specimen handling?",
    outputs="The laboratory must ensure staff responsible for packaging and transport of pathology specimens receive IATA training.",
    reference_outputs="The laboratory is required to ensure all staff involved in packaging and transporting pathology specimens are trained in IATA regulations.",
)


# ------------------- Helpfulness -------------------
# Evaluates how well the generated answer addresses the original user question.
helpfulness_evaluator = create_llm_as_judge(
    prompt=RAG_HELPFULNESS_PROMPT,
    feedback_key="helpfulness",
    model="openai:o3-mini",
)

helpfulness_result = helpfulness_evaluator(
    inputs={"question": "What is required for staff handling pathology specimen packaging and transport?"},
    outputs={"answer": "The laboratory must ensure staff responsible for packaging and transport of pathology specimens receive IATA training."},
)


# ------------------- Retrieval Relevance -------------------
# Measures how relevant the retrieved documents/context are to the user query.
retrieval_relevance_evaluator = create_llm_as_judge(
    prompt=RAG_RETRIEVAL_RELEVANCE_PROMPT,
    feedback_key="retrieval_relevance",
    model="openai:o3-mini",
)

retrieval_relevance_result = retrieval_relevance_evaluator(
    inputs={"question": "What does clause 6.2.5 say about staff training?"},
    context={"documents": standard_context},
)


# ------------------- Groundedness -------------------
# Assesses whether the answer is supported by the provided retrieved context.
groundedness_evaluator = create_llm_as_judge(
    prompt=RAG_GROUNDEDNESS_PROMPT,
    feedback_key="groundedness",
    model="openai:o3-mini",
)

groundedness_result = groundedness_evaluator(
    context={"documents": standard_context},
    outputs={"answer": "The laboratory must ensure staff responsible for packaging and transport of pathology specimens receive IATA training."},
)
