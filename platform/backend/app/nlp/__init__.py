"""NLP Pipeline — 42-layer micro-NLP architecture for AI Tutor.

Usage:
    from app.nlp import FULL_PIPELINE, NLPContext, run_pipeline
    ctx = NLPContext(user_message="...", week=4, topic="梯度下降")
    ctx = run_pipeline(ctx, FULL_PIPELINE)
"""

from .pipeline import NLPContext, run_pipeline
from .preprocessing import (segment_chinese, tag_pos, split_sentences,
                             detect_language, normalize_text, filter_stopwords)
from .intent import detect_intent
from .understanding import (detect_sub_intent, score_sentiment,
                            escalate_frustration, estimate_confidence, detect_politeness)
from .emotion import detect_emotion
from .difficulty import assess_difficulty
from .student_level import (score_vocabulary, score_technical_fluency,
                           detect_learning_style, detect_misconceptions, detect_knowledge_gaps)
from .topic import extract_topics
from .content_analysis import (recognize_entities, detect_code_blocks,
                               detect_math, score_question_quality, score_readability)
from .context_tracker import track_conversation
from .context_memory import (detect_topic_continuity, summarize_session, track_knowledge_state)
from .reranker import retrieve_and_rerank
from .retrieval import expand_query, link_cross_week
from .response import assemble_response
from .response_gen import (adjust_complexity, inject_citations,
                          generate_follow_up, generate_encouragement, check_completeness)
from .coordinators import aggregate_confidence, resolve_conflicts, route_pipeline

# 42-layer pipeline in execution order (+ 3 coordinator layers)
FULL_PIPELINE = [
    # C3. Route Pipeline (informational, before preprocessing)
    route_pipeline,             # C3
    # A. Text Preprocessing (L1-6)
    segment_chinese,        # L1
    tag_pos,                # L2
    split_sentences,        # L3
    detect_language,        # L4
    normalize_text,         # L5
    filter_stopwords,       # L6
    # B. Student Understanding (L7-14)
    detect_intent,          # L7
    detect_sub_intent,      # L8
    detect_emotion,         # L9
    score_sentiment,        # L10
    escalate_frustration,   # L11
    estimate_confidence,    # L12
    detect_emotion,         # L13 (urgency is inside detect_emotion)
    detect_politeness,      # L14
    # C1. Aggregate Confidence (after all understanding layers)
    aggregate_confidence,   # C1
    # C. Student Level (L15-20)
    assess_difficulty,      # L15
    score_vocabulary,       # L16
    score_technical_fluency,  # L17
    detect_learning_style,  # L18
    detect_misconceptions,  # L19
    detect_knowledge_gaps,  # L20
    # D. Content Analysis (L21-27)
    extract_topics,         # L21-22
    recognize_entities,     # L23
    detect_code_blocks,     # L24
    detect_math,            # L25
    score_question_quality, # L26
    score_readability,      # L27
    # C2. Resolve Conflicts (after all analysis layers)
    resolve_conflicts,      # C2
    # E. Context & Memory (L28-32)
    track_conversation,     # L28+30
    detect_topic_continuity,  # L29
    summarize_session,      # L31
    track_knowledge_state,  # L32
    # F. Retrieval (L33-36)
    expand_query,           # L33
    retrieve_and_rerank,    # L34-35
    link_cross_week,        # L36
    # G. Response (L37-42)
    assemble_response,      # L37
    adjust_complexity,      # L38
    inject_citations,       # L39
    generate_follow_up,     # L40
    generate_encouragement, # L41
    check_completeness,     # L42
]
