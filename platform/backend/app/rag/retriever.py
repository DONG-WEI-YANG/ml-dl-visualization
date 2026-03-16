"""High-level retriever: combines FTS search with week-based context."""

import re

from .store import search_fts, get_chunks_by_week

MAX_CONTEXT_CHARS = 4000  # Limit injected context size

# ── Content sanitization ──

try:
    from opencc import OpenCC
    _retriever_occ = OpenCC("s2twp")
except ImportError:
    _retriever_occ = None

# Simplified-only characters (safety net)
_SIMPLIFIED_RE = re.compile(
    r"[么个们仅从仓优伤体佣债倾储儿关兴养减几凤则创办务动劳势单卫发变叹响哑团园圆坏块场壮声处备够头夸奋奖妇妈学宁宝实审宽将对导专岁岗岛帅师帮干并广庆库应废开异弃张弹归当录忆志忧怀态总恋惊惧愿懒戏户扑执扩扫扬扰护报拟拥拨择挡挤挥捞损换据搅携摄撑操收斗断无时显晓暂术机杀杂权条来杨极构枪档梦检样桥欢歼毁毕氢汇汉汤沟济涌温湿滚满灭灯灵烂烧热爱牍状独狭狱猎献环现电疗盐监盖盘矫矿硕础确码祝种稳窃窍竞笔笼简粮纠纤纪纯纸纺线练组细经绘结绝给统继绩绪续维综绿编缘缝缩网罗翘耸职联聪肤肠脑腾艰艺节荣获蓝虏虑虽蛮装观览觉规视证评词译试诗话说请读课调谅谢赁资赋赌赏赔赖赚赛赞趋跃踪轨轩轮软轴轻载辅辆辈辉辑输辩边达迁过运近还这进远连迟适选递通遗邻郁释里钉钟钢钱铁银铜铝链锁错锡键锤长门闭问闲间闷闻阀阅阔阳阴阵阶阻际陆陈险随隐难雾静韩顶顿预领频颜风饭饮饰饱馆驰驱验骗鱼鲜鸟鸡鸣鹰龙龟]"
)

# Wikipedia garbage indicators
_GARBAGE_INDICATORS = [
    "簡繁重定向", "本重定向用來", "請勿使用管道連結",
    "消歧義", "消歧义", "本條目存在以下問題",
    "Template:", "分類:", "|stub",
]

# LaTeX artifact patterns
_LATEX_CLEANUP = re.compile(
    r"\\(?:displaystyle|alpha|beta|gamma|delta|sigma|theta|lambda|nabla|partial|"
    r"hat|vec|frac|sqrt|sum|prod|int|lim|log|exp|sin|cos|tan|"
    r"mathbb|mathrm|mathbf|mathcal|operatorname|left|right|cdot|times|"
    r"leq|geq|neq|approx|equiv|infty|forall|exists|in|subset|cup|cap)(?=\W|$)"
)


def _sanitize_content(text: str) -> str:
    """Clean retrieved content: fix encoding, strip Simplified, remove LaTeX artifacts."""
    # 1. Convert any remaining Simplified → Traditional
    if _retriever_occ is not None:
        try:
            text = _retriever_occ.convert(text)
        except Exception:
            pass

    # 2. Strip remaining Simplified-only characters
    text = _SIMPLIFIED_RE.sub("", text)

    # 3. Clean LaTeX artifacts
    text = _LATEX_CLEANUP.sub("", text)
    text = re.sub(r"[{}]", "", text)  # Stray braces
    text = re.sub(r"\s*\^\s*", "", text)  # Stray carets
    text = re.sub(r"\\[a-z]+", "", text)  # Remaining LaTeX commands

    # 4. Collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"  +", " ", text)

    return text.strip()


def _is_garbage(text: str) -> bool:
    """Check if content is Wikipedia garbage (redirect, disambiguation, etc.)."""
    for indicator in _GARBAGE_INDICATORS:
        if indicator in text:
            return True
    # Too short
    if len(text.strip()) < 30:
        return True
    return False


def retrieve_context(query: str, week: int, top_k: int = 5) -> str:
    """Retrieve relevant curriculum content for a student query.

    Strategy:
    1. Search FTS with week filter first (most relevant)
    2. If not enough results, search across all weeks (but prefer current week)
    3. Sanitize content (fix Simplified Chinese, remove LaTeX, filter garbage)
    4. Build a formatted context string for the LLM
    """
    # First: search within the current week
    results = search_fts(query, week=week, top_k=top_k)

    # If fewer than 2 results from current week, broaden search
    if len(results) < 2:
        broad_results = search_fts(query, week=None, top_k=top_k)
        # Add non-duplicate broad results, but deprioritize off-week content
        seen_ids = {r["id"] for r in results}
        for r in broad_results:
            if r["id"] not in seen_ids:
                results.append(r)
                if len(results) >= top_k:
                    break

    if not results:
        return ""

    # Filter and sanitize results
    clean_results = []
    for r in results:
        content = _sanitize_content(r["content"])
        if _is_garbage(content):
            continue
        if len(content.strip()) < 30:
            continue
        r = dict(r)
        r["content"] = content
        clean_results.append(r)

    if not clean_results:
        return ""

    # Build context string
    parts = []
    total_chars = 0
    for r in clean_results:
        entry = f"[來源：第{r['week']}週 {r['file_type']} — {r['title']}]\n{r['content']}"
        if total_chars + len(entry) > MAX_CONTEXT_CHARS:
            break
        parts.append(entry)
        total_chars += len(entry)

    return "\n\n---\n\n".join(parts)


def get_week_summary(week: int) -> str:
    """Get a summary of all content for a specific week."""
    chunks = get_chunks_by_week(week)
    if not chunks:
        return ""

    # Take first chunk from each file type (usually the heading/intro)
    seen_types = set()
    parts = []
    for c in chunks:
        ft = c["file_type"]
        if ft not in seen_types:
            seen_types.add(ft)
            # Take first 500 chars as summary
            parts.append(f"[{ft}] {c['content'][:500]}")
    return "\n\n".join(parts)
