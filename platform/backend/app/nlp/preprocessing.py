"""Group A: Text Preprocessing Layers (L1-6).

L1 ChineseSegmenter — jieba
L2 POSTagger — jieba.posseg
L3 SentenceSplitter — nltk.punkt / regex
L4 LanguageDetector — langdetect
L5 TextNormalizer — custom
L6 StopwordFilter — nltk + custom Chinese
"""

import re
import logging

logger = logging.getLogger(__name__)

# ── Lazy imports for heavy libraries ──
_jieba = None
_posseg = None
_langdetect = None
_sent_tokenize = None
_zhconv = None
_opencc = None
_thulac = None
_emoji_mod = None


def _get_jieba():
    global _jieba
    if _jieba is None:
        import jieba
        jieba.setLogLevel(logging.WARNING)
        _jieba = jieba
    return _jieba


def _get_posseg():
    global _posseg
    if _posseg is None:
        import jieba.posseg as pseg
        _posseg = pseg
    return _posseg


def _get_langdetect():
    global _langdetect
    if _langdetect is None:
        import langdetect
        _langdetect = langdetect
    return _langdetect


def _get_sent_tokenize():
    global _sent_tokenize
    if _sent_tokenize is None:
        try:
            from nltk.tokenize import sent_tokenize
            _sent_tokenize = sent_tokenize
        except Exception:
            _sent_tokenize = None
    return _sent_tokenize


def _get_zhconv():
    global _zhconv
    if _zhconv is None:
        try:
            import zhconv
            _zhconv = zhconv
        except ImportError:
            logger.warning("zhconv not available for text_normalizer")
            _zhconv = False
    return _zhconv if _zhconv is not False else None


def _get_opencc():
    """Lazy-load OpenCC (more accurate than zhconv for Traditional Chinese)."""
    global _opencc
    if _opencc is None:
        try:
            from opencc import OpenCC
            _opencc = OpenCC("s2twp")  # Simplified → Traditional (Taiwan phrases)
            logger.info("OpenCC loaded (s2twp)")
        except ImportError:
            _opencc = False
    return _opencc if _opencc is not False else None


def _get_thulac():
    """Lazy-load THULAC (Tsinghua Chinese segmenter)."""
    global _thulac
    if _thulac is None:
        try:
            import thulac as _thulac_mod
            _thulac = _thulac_mod.thulac(seg_only=True)
            logger.info("THULAC loaded (seg_only mode)")
        except Exception:
            _thulac = False
    return _thulac if _thulac is not False else None


def _get_emoji_mod():
    """Lazy-load emoji package."""
    global _emoji_mod
    if _emoji_mod is None:
        try:
            import emoji
            _emoji_mod = emoji
        except ImportError:
            _emoji_mod = False
    return _emoji_mod if _emoji_mod is not False else None


# ── Chinese + English stopwords ──

STOPWORDS_ZH = {
    "的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一", "這",
    "上", "也", "到", "說", "要", "會", "可以", "請問", "想", "個", "中", "嗎",
    "怎麼", "什麼", "為什麼", "如何", "能", "用", "讓", "吧", "呢", "跟", "從",
    "那", "被", "把", "給", "很", "太", "再", "還", "去", "來", "做", "看",
    "但", "所以", "因為", "如果", "然後", "或", "而", "對", "比", "以", "其",
    "你", "他", "她", "它", "們", "這個", "那個", "哪", "嗯", "喔", "啊",
}

STOPWORDS_EN = set()
try:
    from nltk.corpus import stopwords as _nltk_sw
    STOPWORDS_EN = set(_nltk_sw.words("english"))
except Exception:
    STOPWORDS_EN = {"the", "is", "a", "an", "in", "of", "to", "and", "for", "it", "this",
                    "that", "how", "what", "why", "can", "do", "i", "my", "me", "be", "are"}


# ── L0.5: Emoji Detector (runs before segmentation) ──

def emoji_detector(ctx):
    """L0.5: Detect and extract emoji from message."""
    emo = _get_emoji_mod()
    if emo is None:
        return ctx
    emoji_list = [c["emoji"] for c in emo.emoji_list(ctx.user_message)]
    if emoji_list:
        ctx.has_emoji = True
        ctx.emoji_list = emoji_list
    return ctx


# ── L1: Chinese Segmenter ──

def chinese_segmenter(ctx):
    """L1: Segment text using jieba + optional THULAC merge."""
    jieba = _get_jieba()
    ctx.tokens = list(jieba.cut(ctx.user_message))

    # THULAC parallel segmentation: merge unique tokens from THULAC
    thu = _get_thulac()
    if thu is not None:
        try:
            thulac_result = thu.cut(ctx.user_message)
            thulac_tokens = [w for w, _ in thulac_result if len(w.strip()) > 1]
            jieba_set = set(ctx.tokens)
            # Add THULAC-only tokens that jieba missed (often better compound words)
            for t in thulac_tokens:
                if t not in jieba_set and t in ctx.user_message:
                    ctx.tokens.append(t)
        except Exception as e:
            logger.debug("THULAC merge failed: %s", e)

    return ctx


# ── L2: POS Tagger ──

def pos_tagger(ctx):
    """L2: POS tagging using jieba.posseg."""
    pseg = _get_posseg()
    ctx.pos_tags = [(word, flag) for word, flag in pseg.cut(ctx.user_message)]
    return ctx


# ── L3: Sentence Splitter ──

_SENT_SPLIT_RE = re.compile(r'(?<=[。！？.!?])\s*|(?<=\n)\s*')


def sentence_splitter(ctx):
    """L3: Split text into sentences using nltk (English) and regex (Chinese)."""
    text = ctx.user_message.strip()
    if not text:
        ctx.sentences = []
        return ctx

    # Try nltk for English-heavy text
    sent_tok = _get_sent_tokenize()
    if sent_tok and ctx.language == "en":
        ctx.sentences = sent_tok(text)
    else:
        # Regex-based for Chinese / mixed
        parts = _SENT_SPLIT_RE.split(text)
        ctx.sentences = [s.strip() for s in parts if s.strip()]

    if not ctx.sentences:
        ctx.sentences = [text]

    return ctx


# ── L4: Language Detector ──

def language_detector(ctx):
    """L4: Detect language using langdetect."""
    ld = _get_langdetect()
    try:
        result = ld.detect_langs(ctx.user_message)
        if result:
            ctx.language = result[0].lang
            ctx.language_confidence = result[0].prob
        else:
            ctx.language = "zh"
            ctx.language_confidence = 0.5
    except Exception:
        # Default to Chinese for this course
        ctx.language = "zh"
        ctx.language_confidence = 0.5
    return ctx


# ── L5: Text Normalizer ──

_FULLWIDTH_MAP = str.maketrans(
    "\uff10\uff11\uff12\uff13\uff14\uff15\uff16\uff17\uff18\uff19\uff21\uff22\uff23\uff24\uff25\uff26\uff27\uff28\uff29\uff2a\uff2b\uff2c\uff2d\uff2e\uff2f\uff30\uff31\uff32\uff33\uff34\uff35\uff36\uff37\uff38\uff39\uff3a\uff41\uff42\uff43\uff44\uff45\uff46\uff47\uff48\uff49\uff4a\uff4b\uff4c\uff4d\uff4e\uff4f\uff50\uff51\uff52\uff53\uff54\uff55\uff56\uff57\uff58\uff59\uff5a\uff08\uff09\u3010\u3011",
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz()[]",
)


def text_normalizer(ctx):
    """L5: Normalize text — fullwidth->halfwidth, collapse whitespace, Traditional Chinese.

    Uses OpenCC (s2twp) for more accurate Simplified→Traditional (Taiwan phrases) conversion,
    with zhconv as fallback.
    """
    text = ctx.user_message
    # Fullwidth to halfwidth
    text = text.translate(_FULLWIDTH_MAP)
    # Collapse multiple spaces/newlines
    text = re.sub(r'\s+', ' ', text).strip()
    # Strip emoji text representations (keep original characters intact)
    emo = _get_emoji_mod()
    if emo is not None:
        text = emo.replace_emoji(text, replace="")
        text = re.sub(r'\s+', ' ', text).strip()
    # Convert to Traditional Chinese — prefer OpenCC (more accurate), fallback to zhconv
    occ = _get_opencc()
    if occ is not None:
        try:
            text = occ.convert(text)
        except Exception:
            # Fallback to zhconv
            zc = _get_zhconv()
            if zc is not None:
                try:
                    text = zc.convert(text, 'zh-tw')
                except Exception:
                    pass
    else:
        zc = _get_zhconv()
        if zc is not None:
            try:
                text = zc.convert(text, 'zh-tw')
            except Exception:
                pass
    ctx.normalized_text = text
    return ctx


# ── L6: Stopword Filter ──

def stopword_filter(ctx):
    """L6: Remove stopwords from token list."""
    all_stops = STOPWORDS_ZH | STOPWORDS_EN
    ctx.filtered_tokens = [t for t in ctx.tokens if t.strip() and t not in all_stops and len(t.strip()) > 0]
    return ctx


# ── Public aliases (used by __init__.py FULL_PIPELINE) ──

detect_emoji = emoji_detector
segment_chinese = chinese_segmenter
tag_pos = pos_tagger
split_sentences = sentence_splitter
detect_language = language_detector
normalize_text = text_normalizer
filter_stopwords = stopword_filter
