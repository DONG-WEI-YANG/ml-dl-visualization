"""NLP Pipeline — 42-layer micro-NLP architecture for AI Tutor.

Architecture (7 groups, 42 layers):
  A. Text Preprocessing (L1-6): segmentation, POS, sentences, language, normalize, stopwords
  B. Student Understanding (L7-14): intent, sub-intent, emotion, sentiment, frustration, confidence, urgency, politeness
  C. Student Level (L15-20): difficulty, vocabulary, fluency, learning-style, misconception, knowledge-gap
  D. Content Analysis (L21-27): keywords, concepts, NER, code-detect, math-detect, question-quality, readability
  E. Context & Memory (L28-32): conversation, topic-continuity, hint-ladder, session-summary, knowledge-state
  F. Retrieval (L33-36): query-expand, RAG, semantic-rerank, cross-week
  G. Response (L37-42): assemble, complexity-adjust, citation, follow-up, encouragement, completeness-check
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class NLPContext:
    """Shared context object passed through all 42 NLP layers."""
    # ── Input ──
    user_message: str = ""
    conversation_history: list[dict] = field(default_factory=list)
    week: int = 1
    topic: str = ""
    is_homework: bool = False
    student_id: str = ""

    # ── A. Text Preprocessing (L1-6) ──
    tokens: list[str] = field(default_factory=list)           # L1 ChineseSegmenter
    pos_tags: list[tuple[str, str]] = field(default_factory=list)  # L2 POSTagger
    sentences: list[str] = field(default_factory=list)         # L3 SentenceSplitter
    language: str = "zh"                                       # L4 LanguageDetector
    language_confidence: float = 0.0
    normalized_text: str = ""                                  # L5 TextNormalizer
    filtered_tokens: list[str] = field(default_factory=list)   # L6 StopwordFilter

    # ── B. Student Understanding (L7-14) ──
    intent: str = "general"                                    # L7 IntentClassifier
    intent_confidence: float = 0.0
    sub_intent: str = ""                                       # L8 SubIntentDetector
    sub_intent_confidence: float = 0.0
    emotion: str = "neutral"                                   # L9 EmotionClassifier
    emotion_confidence: float = 0.0
    sentiment_score: float = 0.5                               # L10 SentimentScorer (0=negative, 1=positive)
    frustration_level: int = 0                                 # L11 FrustrationEscalator (0-5)
    confidence_level: float = 0.5                              # L12 ConfidenceEstimator (0-1)
    urgency: str = "normal"                                    # L13 UrgencyDetector
    politeness_score: float = 0.5                              # L14 PolitenessDetector (0-1)

    # ── C. Student Level (L15-20) ──
    student_level: str = "beginner"                            # L15 DifficultyAssessor
    uses_technical_terms: bool = False
    question_complexity: str = "simple"
    vocabulary_score: float = 0.0                              # L16 VocabularyLevelScorer (0-1)
    technical_fluency: float = 0.0                             # L17 TechnicalFluencyScorer (0-1)
    learning_style: str = "balanced"                           # L18 LearningStyleDetector
    misconceptions: list[str] = field(default_factory=list)    # L19 MisconceptionDetector
    knowledge_gaps: list[str] = field(default_factory=list)    # L20 KnowledgeGapDetector

    # ── D. Content Analysis (L21-27) ──
    keywords: list[str] = field(default_factory=list)          # L21 KeywordExtractor
    keyword_scores: list[tuple[str, float]] = field(default_factory=list)
    domain_concepts: list[str] = field(default_factory=list)   # L22 DomainConceptMatcher
    named_entities: list[dict] = field(default_factory=list)   # L23 NER
    has_code: bool = False                                     # L24 CodeBlockDetector
    code_language: str = ""
    code_blocks: list[str] = field(default_factory=list)
    has_math: bool = False                                     # L25 MathExpressionDetector
    math_expressions: list[str] = field(default_factory=list)
    question_quality: float = 0.5                              # L26 QuestionQualityScorer (0-1)
    quality_feedback: str = ""
    readability_score: float = 0.0                             # L27 ReadabilityScorer

    # ── E. Context & Memory (L28-32) ──
    turn_count: int = 0                                        # L28 ConversationTracker
    is_followup: bool = False
    previous_intent: str = ""
    topic_continuity: float = 0.0                              # L29 TopicContinuityDetector (0-1)
    continued_topic: str = ""
    hint_level: int = 1                                        # L30 HintLadderManager (1-4)
    session_summary: str = ""                                  # L31 SessionSummarizer
    known_concepts: list[str] = field(default_factory=list)    # L32 KnowledgeStateTracker
    unknown_concepts: list[str] = field(default_factory=list)

    # ── F. Retrieval (L33-36) ──
    expanded_query: str = ""                                   # L33 QueryExpander
    rag_context: str = ""                                      # L34 RAGRetriever
    rag_sources: list[str] = field(default_factory=list)
    reranked_results: list[dict] = field(default_factory=list) # L35 SemanticReranker
    cross_week_links: list[dict] = field(default_factory=list) # L36 CrossWeekLinker

    # ── G. Response (L37-42) ──
    response: str = ""                                         # L37 ResponseAssembler
    response_complexity: str = "moderate"                      # L38 ComplexityAdjuster
    citations: list[str] = field(default_factory=list)         # L39 CitationInjector
    follow_up_questions: list[str] = field(default_factory=list)  # L40 FollowUpGenerator
    encouragement: str = ""                                    # L41 EncouragementGenerator
    completeness_score: float = 0.0                            # L42 ResponseCompletenessChecker
    completeness_missing: list[str] = field(default_factory=list)

    # ── Pipeline metadata ──
    layers_executed: list[str] = field(default_factory=list)
    total_processing_ms: float = 0.0


def run_pipeline(ctx: NLPContext, layers: list) -> NLPContext:
    """Execute all NLP layers sequentially, collecting timing and layer names."""
    import time
    start = time.time()
    for layer_fn in layers:
        layer_name = layer_fn.__name__
        try:
            ctx = layer_fn(ctx)
            ctx.layers_executed.append(layer_name)
        except Exception as e:
            logger.warning("NLP layer %s failed: %s", layer_name, e)
            ctx.layers_executed.append(f"{layer_name}:ERROR")
    ctx.total_processing_ms = (time.time() - start) * 1000
    return ctx
