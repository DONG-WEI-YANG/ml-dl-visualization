"""Text correction layers — typo fixing, encoding normalization, pinyin-based correction."""

import logging
from .pipeline import NLPContext

logger = logging.getLogger(__name__)

# ── Lazy imports ──
_pycorrector = None
_zhconv = None
_pypinyin = None


def _get_pycorrector():
    global _pycorrector
    if _pycorrector is None:
        try:
            from pycorrector import ProperCorrector
            _pycorrector = ProperCorrector()
        except Exception as e:
            logger.warning("pycorrector ProperCorrector not available: %s", e)
            _pycorrector = False
    return _pycorrector if _pycorrector is not False else None


def _get_zhconv():
    global _zhconv
    if _zhconv is None:
        try:
            import zhconv
            _zhconv = zhconv
        except ImportError:
            logger.warning("zhconv not available")
            _zhconv = False
    return _zhconv if _zhconv is not False else None


def _get_pypinyin():
    global _pypinyin
    if _pypinyin is None:
        try:
            import pypinyin
            _pypinyin = pypinyin
        except ImportError:
            logger.warning("pypinyin not available")
            _pypinyin = False
    return _pypinyin if _pypinyin is not False else None


# ── Known ML/DL concept terms for pinyin comparison ──
# Imported lazily from topic.py to avoid circular imports at module load time.

_CONCEPT_TERMS = None


def _get_concept_terms():
    """Return known concept term keys from topic.py CONCEPT_MAP (Chinese only)."""
    global _CONCEPT_TERMS
    if _CONCEPT_TERMS is None:
        try:
            from .topic import CONCEPT_MAP
            import re
            _CONCEPT_TERMS = [
                k for k in CONCEPT_MAP.keys()
                if re.search(r'[\u4e00-\u9fff]', k)
            ]
        except Exception:
            _CONCEPT_TERMS = []
    return _CONCEPT_TERMS


# ── Layer: correct_typos ──

def correct_typos(ctx: NLPContext) -> NLPContext:
    """Use pycorrector ProperCorrector to detect and fix common typos in student messages.

    Only processes messages shorter than 200 characters to avoid slow calls.
    ProperCorrector uses pinyin and stroke similarity to correct proper nouns
    and domain terms without needing kenlm.
    Stores corrected text in ctx.corrected_text and logs corrections.
    """
    corrector = _get_pycorrector()
    if corrector is None:
        return ctx

    text = ctx.normalized_text or ctx.user_message
    if len(text) > 200:
        logger.debug("correct_typos: skipping — message too long (%d chars)", len(text))
        return ctx

    try:
        result = corrector.correct(text)
        # ProperCorrector returns {'source': ..., 'target': ..., 'errors': [...]}
        errors = result.get("errors", [])
        target = result.get("target", text)
        if errors:
            ctx.corrected_text = target
            ctx.normalized_text = target
            for detail in errors:
                # detail is (error_word, correct_word, position)
                wrong = detail[0] if len(detail) > 0 else ""
                right = detail[1] if len(detail) > 1 else ""
                ctx.corrections.append({
                    "original": wrong,
                    "corrected": right,
                    "type": "pycorrector",
                })
            logger.info("correct_typos: %d correction(s) applied: %s", len(errors), errors)
        else:
            ctx.corrected_text = text
    except Exception as e:
        logger.warning("correct_typos failed: %s", e)
        ctx.corrected_text = text

    return ctx


# ── Layer: normalize_chinese_variant ──

def normalize_chinese_variant(ctx: NLPContext) -> NLPContext:
    """Normalize all Chinese text to Traditional Chinese (zh-tw) using zhconv.

    The course is taught in Taiwan, so Traditional Chinese is the standard.
    """
    zc = _get_zhconv()
    if zc is None:
        return ctx

    try:
        ctx.user_message = zc.convert(ctx.user_message, 'zh-tw')
        if ctx.normalized_text:
            ctx.normalized_text = zc.convert(ctx.normalized_text, 'zh-tw')
        if ctx.corrected_text:
            ctx.corrected_text = zc.convert(ctx.corrected_text, 'zh-tw')
    except Exception as e:
        logger.warning("normalize_chinese_variant failed: %s", e)

    return ctx


# ── Layer: detect_pinyin_typos ──

def detect_pinyin_typos(ctx: NLPContext) -> NLPContext:
    """Use pypinyin to detect pinyin-based typos in Chinese ML/DL terms.

    Compares the pinyin (with tones) of user tokens against known concept terms.
    If pinyin matches but characters differ, the user likely typed the wrong
    homophone — a common mistake with phonetic input methods.
    """
    pp = _get_pypinyin()
    if pp is None:
        return ctx

    concept_terms = _get_concept_terms()
    if not concept_terms:
        return ctx

    try:
        import re as _re

        # Extract Chinese word segments from the message (2-4 chars)
        text = ctx.normalized_text or ctx.user_message
        user_segments = _re.findall(r'[\u4e00-\u9fff]{2,4}', text)
        if not user_segments:
            return ctx

        # Build pinyin cache for concept terms (both with and without tones)
        concept_pinyin_map = {}
        for term in concept_terms:
            py_tone = tuple(
                p[0] for p in pp.pinyin(term, style=pp.Style.TONE)
            )
            py_normal = tuple(
                p[0] for p in pp.pinyin(term, style=pp.Style.NORMAL)
            )
            concept_pinyin_map[term] = (py_tone, py_normal)

        # Check each user segment
        for seg in user_segments:
            seg_py_tone = tuple(
                p[0] for p in pp.pinyin(seg, style=pp.Style.TONE)
            )
            seg_py_normal = tuple(
                p[0] for p in pp.pinyin(seg, style=pp.Style.NORMAL)
            )
            for term, (term_py_tone, term_py_normal) in concept_pinyin_map.items():
                if len(seg_py_tone) != len(term_py_tone):
                    continue
                if seg == term:
                    continue  # exact match, no typo
                # Match with tones (exact pinyin match) or without tones
                # (same base syllable, different tone — common with phonetic input)
                if seg_py_tone == term_py_tone:
                    match_type = "pinyin_exact"
                elif seg_py_normal == term_py_normal:
                    match_type = "pinyin_tone_diff"
                else:
                    continue
                # Pinyin matches but characters differ — likely typo
                suggestion = f"可能的拼音錯誤：「{seg}」→ 應為「{term}」（拼音相似）"
                if suggestion not in ctx.misconceptions:
                    ctx.misconceptions.append(suggestion)
                # Also store in quality_feedback for downstream persistence
                # (misconceptions list may be reset by L19 misconception_detector)
                if ctx.quality_feedback:
                    ctx.quality_feedback += f"；{suggestion}"
                else:
                    ctx.quality_feedback = suggestion
                ctx.corrections.append({
                    "original": seg,
                    "corrected": term,
                    "type": match_type,
                })
                logger.info("detect_pinyin_typos: '%s' -> '%s' (%s)", seg, term, match_type)

    except Exception as e:
        logger.warning("detect_pinyin_typos failed: %s", e)

    return ctx
