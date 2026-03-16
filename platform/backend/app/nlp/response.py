"""Layer 7: Response Assembly — adaptive response generation using all NLP layers."""

import jieba.analyse
import textstat
from snownlp import SnowNLP
from .pipeline import NLPContext

# ── Base templates (by intent) ──

INTENT_TEMPLATES = {
    "definition": "關於「{topic}」的定義，以下是課程教材中的相關說明：\n\n{context}",
    "how": "關於你問的操作方法，以下是教材中的步驟與說明：\n\n{context}",
    "why": "這是個好問題！理解「為什麼」比知道「怎麼做」更重要。\n\n以下是教材中的解釋：\n\n{context}",
    "compare": "以下是教材中關於這個比較的說明：\n\n{context}",
    "example": "以下是教材中的相關範例：\n\n{context}",
    "debug": "讓我們一起排除問題。先看看教材中的相關說明：\n\n{context}",
    "formula": "關於「{topic}」的數學公式，以下是教材中的推導與說明：\n\n{context}",
    "code": "關於程式實作，以下是教材中的相關說明與範例程式碼：\n\n{context}",
    "parameter": "關於「{topic}」的參數設定，以下是教材中的說明：\n\n{context}",
    "performance": "關於模型效能評估，以下是教材中的相關內容：\n\n{context}",
    "data": "關於資料處理，以下是教材中的說明：\n\n{context}",
    "visualization": "關於視覺化，以下是教材中的相關說明：\n\n{context}",
    "intuition": "讓我們從直覺的角度來理解這個概念：\n\n{context}",
    "application": "關於「{topic}」的實際應用，以下是教材中的說明：\n\n{context}",
    "prerequisite": "以下是學習這個主題需要的先備知識：\n\n{context}",
    "summary": "以下是教材中的重點整理：\n\n{context}",
    "troubleshoot": "關於環境和安裝問題，以下是教材中的說明：\n\n{context}",
    "deeper": "想要更深入了解「{topic}」，以下是教材中的進階內容：\n\n{context}",
    "general": "以下是課程教材中與你問題相關的內容：\n\n{context}",
}

# ── Emotion-adaptive prefixes ──

EMOTION_PREFIX = {
    "frustrated": "我理解你可能覺得有些挫折，這是學習過程中很正常的。讓我們一步步來解決。\n\n",
    "confused": "沒關係，這個概念確實需要時間消化。讓我試著換個方式說明。\n\n",
    "curious": "很棒的求知精神！",
    "confident": "很高興你對這個有想法！讓我們來驗證一下。\n\n",
    "neutral": "",
}

# ── Difficulty-adaptive guidance ──

LEVEL_GUIDANCE = {
    "beginner": {
        "definition": "💡 **初學者建議：** 先記住這個概念的核心意思，用一句話概括。之後透過實作加深理解。",
        "how": "💡 **初學者建議：** 不要急，一次只做一個步驟。每完成一步就 print 出結果確認。",
        "why": "💡 **初學者建議：** 先接受「這是最佳實踐」，隨著學習深入，你會自然理解背後的道理。",
        "compare": "💡 **初學者建議：** 先學好其中一個方法，等熟練後再比較差異會更有感覺。",
        "formula": "💡 **初學者建議：** 公式看起來複雜，但先理解每個符號的意思。試著代入數字算一遍。",
        "code": "💡 **初學者建議：** 先把範例程式碼完整複製執行一遍，確認能跑。再逐行修改觀察變化。",
        "debug": "💡 **初學者建議：** 錯誤訊息看起來很可怕，但最重要的資訊通常在最後一行。先讀那行。",
        "general": "💡 **初學者建議：** 一次專注一個概念，不要貪多。理解比記憶更重要。",
    },
    "intermediate": {
        "definition": "💡 **進一步思考：** 這個概念和你之前學過的哪些有關？試著建立知識圖譜。",
        "how": "💡 **進一步思考：** 理解步驟後，思考每一步「為什麼」這樣做。有沒有替代方案？",
        "why": "💡 **進一步思考：** 能不能用數學或實驗來驗證這個解釋？",
        "compare": "💡 **進一步思考：** 在什麼情境下，「較差」的方法反而更適合？",
        "formula": "💡 **進一步思考：** 嘗試推導公式。從最簡單的情況開始，再推廣到一般情況。",
        "code": "💡 **進一步思考：** 程式碼的時間/空間複雜度是多少？有更高效的寫法嗎？",
        "debug": "💡 **進一步思考：** 除了修復這個 bug，思考如何預防類似問題。可以加什麼檢查？",
        "general": "💡 **進一步思考：** 試著把學到的概念連結到實際應用場景。",
    },
    "advanced": {
        "definition": "💡 **延伸探索：** 查閱原始論文了解這個概念的演進歷史和最新發展。",
        "how": "💡 **延伸探索：** 考慮邊界情況和實際部署的挑戰。生產環境和實驗環境有什麼差異？",
        "why": "💡 **延伸探索：** 從理論角度分析：是否有嚴格的數學證明支持這個觀點？",
        "compare": "💡 **延伸探索：** 做一個 ablation study — 系統性地測試不同組合的效果。",
        "formula": "💡 **延伸探索：** 思考這個公式在更高維度或不同假設下的推廣。",
        "code": "💡 **延伸探索：** 思考如何將這段程式碼模組化、加上型別提示和單元測試。",
        "debug": "💡 **延伸探索：** 寫一個自動化測試來覆蓋這個 edge case，防止回歸。",
        "general": "💡 **延伸探索：** 查閱最新的 survey paper 了解這個領域的前沿發展。",
    },
}

# ── Hint Ladder suffixes ──

HINT_LADDER = {
    1: (
        "\n\n🔍 **Hint Ladder Level 1 — 釐清問題**\n"
        "在我給更多提示之前，先告訴我：\n"
        "- 你已經嘗試了什麼？\n"
        "- 哪個部分你覺得最不確定？"
    ),
    2: (
        "\n\n📚 **Hint Ladder Level 2 — 概念提示**\n"
        "根據教材內容，這個問題的關鍵概念已列在上方。\n"
        "試著從這些概念出發思考你的問題。"
    ),
    3: (
        "\n\n🗺️ **Hint Ladder Level 3 — 步驟引導**\n"
        "建議的解題步驟：\n"
        "1. 回顧上方教材中提到的方法\n"
        "2. 嘗試用最簡單的資料先做一次\n"
        "3. 逐步增加複雜度\n"
        "4. 如果卡住，告訴我卡在哪一步"
    ),
    4: (
        "\n\n💻 **Hint Ladder Level 4 — 局部範例**\n"
        "你已經嘗試了很多次，讓我提供更具體的指引。\n"
        "請參考上方教材中的程式碼範例，特別注意：\n"
        "- 函數的輸入輸出格式\n"
        "- 關鍵參數的預設值\n"
        "- 教材中標示的常見錯誤"
    ),
}

# ── Intent-specific guidance ──

INTENT_GUIDANCE = {
    "definition": "- 試著用自己的話重新解釋\n- 能想到生活中的類比嗎？",
    "how": "1. 先理解目的，再看步驟\n2. 實際操作一次\n3. 不看教材重做一遍",
    "why": "- 如果不這樣做，會怎樣？\n- 有替代方案嗎？trade-off 是什麼？",
    "compare": "試著列比較表：適用場景 / 複雜度 / 優缺點 / 何時選擇",
    "example": "1. 修改範例中的一個參數\n2. 預測結果\n3. 執行驗證",
    "debug": "1. 讀最後一行錯誤訊息\n2. 定位問題行號\n3. print 出變數檢查\n4. 用最小資料重現",
    "formula": "1. 理解每個符號\n2. 代入小數字手算\n3. 思考各項對結果的影響",
    "code": "1. 先整體讀一遍理解結構\n2. 分段執行\n3. 加 print 觀察中間值",
    "parameter": "1. 先用預設值\n2. 一次改一個參數\n3. 記錄對照表\n4. 最後用 GridSearch",
    "performance": "- 不只看 accuracy\n- 看 confusion matrix\n- 比較 train vs val\n- 用交叉驗證",
    "data": "☐ shape/dtypes ☐ 缺失值 ☐ 分布 ☐ 編碼 ☐ 標準化 ☐ 分割",
    "visualization": "- 類別→柱狀圖 連續→折線圖 分布→直方圖\n- 一定要有標題和軸標籤",
    "intuition": "- 想像極端情況\n- 用生活類比\n- 畫示意圖",
    "application": "- 哪些產業在用？\n- 你的領域能怎麼應用？\n- 有哪些倫理考量？",
    "prerequisite": "- 確認前幾週基礎\n- 不熟的先回去複習\n- 查術語表",
    "summary": "1. 寫一頁摘要\n2. 列 3-5 個核心概念\n3. 畫概念地圖",
    "troubleshoot": "1. 確認 Python 版本\n2. 確認套件版本\n3. 試乾淨虛擬環境",
    "deeper": "- 讀進階段落\n- 做挑戰題\n- 查延伸閱讀\n- 跨資料集實驗",
    "general": "- 你在哪一週的內容？\n- 在做什麼操作？\n- 期望 vs 實際結果？",
}

# ── Domain concept note ──

def _concept_note(ctx: NLPContext) -> str:
    if not ctx.domain_concepts:
        return ""
    concepts = "、".join(ctx.domain_concepts[:3])
    return f"\n\n📌 **偵測到的課程概念：** {concepts}\n"


# ── No context template ──

NO_CONTEXT = (
    "抱歉，我在課程教材中沒有找到直接相關的內容。\n\n"
    "你可以試試：\n"
    "1. 用不同的關鍵詞重新描述問題\n"
    "2. 指定你想問的是哪一週的內容\n"
    "3. 查閱該週的講義或 Notebook\n\n"
    "你的問題關鍵詞：{keywords}\n\n"
    "**提示：** 可以直接問「第 X 週的重點是什麼」。"
)

HOMEWORK_GUARD = (
    "這看起來是作業相關的問題。根據課程規定，我不能直接給你答案，但可以引導你思考。\n\n"
    "以下是相關的教材內容，請先閱讀後嘗試自己解題：\n\n"
    "{context}\n\n"
    "---\n"
    "💡 **解題引導：**\n"
    "1. 你目前的思路是什麼？\n"
    "2. 你已經嘗試了什麼方法？結果如何？\n"
    "3. 哪個步驟卡住了？\n"
    "4. 你能把問題拆解成更小的子問題嗎？\n\n"
    "先回答上面的問題，我再進一步引導你。"
)


def _extract_source_blocks(rag_context: str) -> list[dict]:
    """Parse RAG context into individual source blocks with metadata."""
    import re
    blocks = rag_context.split("\n\n---\n\n")
    result = []
    source_re = re.compile(r"^\[來源：第(\d+)週\s+(\S+)\s+—\s+(.+?)\]\n(.+)", re.DOTALL)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        m = source_re.match(block)
        if m:
            result.append({
                "week": int(m.group(1)),
                "file_type": m.group(2),
                "title": m.group(3),
                "content": m.group(4).strip(),
            })
        else:
            result.append({"week": 0, "file_type": "", "title": "", "content": block})
    return result


def _clean_for_display(text: str) -> str:
    """Clean text for user display: remove source markers, math fragments, metadata."""
    import re
    # Remove [來源：...] markers
    text = re.sub(r"\[來源：[^\]]+\]", "", text)
    # Remove markdown section headers like ## 作業五：...
    text = re.sub(r"^#{1,4}\s+", "", text, flags=re.MULTILINE)
    # Remove orphaned math: lines with mostly symbols
    text = re.sub(r"^[\s\d\.\,\;\:\(\)\[\]\|×→←↔≤≥=+\-*/^_\\$]+$", "", text, flags=re.MULTILINE)
    # Remove empty parentheses and brackets
    text = re.sub(r"[\(\)]\s*[\(\)]", "", text)
    # Remove "參考文獻", "另見", "閱讀更多" sections
    text = re.sub(r"\n(?:參考文獻|另見|閱讀更多|References|See also)[\s\S]*$", "", text)
    # Collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _summarize_block(content: str, max_sentences: int = 3) -> str:
    """Extract key sentences from a content block using scoring heuristics."""
    import re
    content = _clean_for_display(content)

    # Split into sentences (Chinese and English)
    sentences = re.split(r'(?<=[。！？\.\!\?])\s*', content)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    if not sentences:
        return ""
    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    # Score sentences: prefer definitions and explanations
    definition_signals = ["是", "為", "指", "定義", "稱為", "means", "is a", "refers", "defined"]
    explanation_signals = ["因為", "所以", "由於", "例如", "比如", "包括", "目的", "用於",
                           "such as", "because", "used for", "purpose"]
    junk_signals = ["頁面存檔", "外部連結", "stub", "小作品", "ISBN", "doi:", "arXiv"]

    scored = []
    for i, sent in enumerate(sentences):
        score = 0
        # Skip junk
        if any(j in sent for j in junk_signals):
            continue
        # First sentence bonus
        if i == 0:
            score += 3
        # Definition signals
        if any(sig in sent for sig in definition_signals):
            score += 2
        # Explanation signals
        if any(sig in sent for sig in explanation_signals):
            score += 1
        # Penalize very short or very long
        if len(sent) < 15:
            score -= 2
        if len(sent) > 300:
            score -= 1
        # Penalize sentences with too many English fragments (likely math/code)
        eng_ratio = len(re.findall(r'[a-zA-Z]', sent)) / max(len(sent), 1)
        if eng_ratio > 0.6:
            score -= 2
        scored.append((score, i, sent))

    if not scored:
        return ""

    # Take top sentences, maintaining original order
    scored.sort(key=lambda x: x[0], reverse=True)
    top_indices = sorted([s[1] for s in scored[:max_sentences]])
    result = " ".join(sentences[i] for i in top_indices)

    # Hard limit: 500 chars max per block
    if len(result) > 500:
        result = result[:497] + "..."
    return result


def _digest_rag_context(rag_context: str, intent: str, rag_keywords: list) -> str:
    """Digest raw RAG context into concise, readable response.

    Extracts key sentences from each source block and combines them.
    """
    if not rag_context:
        return ""

    blocks = _extract_source_blocks(rag_context)
    if not blocks:
        return _clean_for_display(rag_context)[:600]

    sections = []
    total_len = 0
    for block in blocks[:2]:  # Max 2 source blocks for conciseness
        content = block["content"]
        summary = _summarize_block(content, max_sentences=3)
        if summary and len(summary) > 20:
            sections.append(summary)
            total_len += len(summary)
            if total_len > 800:  # Hard total limit
                break

    if not sections:
        return _clean_for_display(rag_context)[:400]

    return "\n\n".join(sections)


def _build_takeaway(ctx: NLPContext, rag_keywords: list, nlp_topics: list) -> str:
    """Build a concise key-takeaway section from NLP analysis results."""
    items = []

    # Add relevant domain concepts
    if ctx.domain_concepts:
        concepts_str = "、".join(ctx.domain_concepts[:4])
        items.append(f"**核心概念：** {concepts_str}")

    # Add RAG-derived keywords as learning focus
    if rag_keywords and len(rag_keywords) >= 2:
        kw_str = "、".join(rag_keywords[:4])
        items.append(f"**學習重點：** {kw_str}")

    # Add cross-week links if available
    if ctx.cross_week_links:
        links = ctx.cross_week_links[:3]
        link_strs = [f"第{lnk.get('week', '?')}週「{lnk.get('title', '')}」"
                     for lnk in links if isinstance(lnk, dict)]
        if link_strs:
            items.append(f"**相關教材：** {'、'.join(link_strs)}")

    if not items:
        return ""

    header = "\n\n📝 **重點整理：**\n"
    return header + "\n".join(f"- {item}" for item in items)


def assemble_response(ctx: NLPContext) -> NLPContext:
    """Assemble the final response using all NLP layer outputs."""
    # --- NLP-enhanced topic extraction via jieba TF-IDF ---
    nlp_topics = []
    try:
        nlp_topics = jieba.analyse.extract_tags(ctx.user_message, topK=5)
    except Exception:
        pass
    topic = "、".join(nlp_topics[:3]) if nlp_topics else (
        "、".join(ctx.keywords[:3]) if ctx.keywords else ctx.user_message[:20]
    )

    # --- Extract RAG context keywords via SnowNLP ---
    rag_keywords = []
    if ctx.rag_context:
        try:
            rag_keywords = SnowNLP(ctx.rag_context).keywords(5)
        except Exception:
            pass

    # No context found
    if not ctx.rag_context:
        ctx.response = NO_CONTEXT.format(
            keywords="、".join(ctx.keywords) if ctx.keywords else ctx.user_message[:30]
        )
        return ctx

    # Homework mode
    if ctx.is_homework:
        ctx.response = HOMEWORK_GUARD.format(context=ctx.rag_context)
        return ctx

    # Build response parts
    parts = []

    # 1. Emotion-adaptive prefix
    prefix = EMOTION_PREFIX.get(ctx.emotion, "")
    if prefix:
        parts.append(prefix)

    # 2. Main content — digest RAG context into structured sections
    template = INTENT_TEMPLATES.get(ctx.intent, INTENT_TEMPLATES["general"])

    # Digest RAG: extract key sentences instead of raw paste
    digested = _digest_rag_context(ctx.rag_context, ctx.intent, rag_keywords)
    parts.append(template.format(topic=topic, context=digested))

    # 3. Key takeaway section (concise summary of RAG content)
    takeaway = _build_takeaway(ctx, rag_keywords, nlp_topics)
    if takeaway:
        parts.append(takeaway)

    # 4. Domain concept note (enhanced with RAG keywords)
    concept_note = _concept_note(ctx)
    if concept_note:
        parts.append(concept_note)
    elif rag_keywords:
        kw_str = "、".join(rag_keywords[:3])
        parts.append(f"\n\n📌 **相關關鍵詞：** {kw_str}\n")

    # 4. Separator
    parts.append("\n---")

    # 5. Level-adaptive guidance
    level_guides = LEVEL_GUIDANCE.get(ctx.student_level, LEVEL_GUIDANCE["beginner"])
    guide = level_guides.get(ctx.intent, level_guides.get("general", ""))
    if guide:
        parts.append(guide)

    # 6. Intent-specific action items
    action = INTENT_GUIDANCE.get(ctx.intent, "")
    if action and action not in guide:
        parts.append(f"\n📋 **行動建議：**\n{action}")

    # 7. Hint Ladder (for homework-adjacent or multi-turn)
    if ctx.turn_count >= 1 or ctx.emotion in ("frustrated", "confused"):
        hint = HINT_LADDER.get(ctx.hint_level, "")
        if hint:
            parts.append(hint)

    # 8. Follow-up acknowledgment
    if ctx.is_followup:
        parts.append("\n\n（我注意到這是你的追問，已根據先前的對話調整回覆。）")

    # 9. Urgency response
    if ctx.urgency == "high":
        parts.append("\n\n⏰ 看起來你比較趕時間，我盡量給重點。如果需要更詳細的說明，完成後再回來問我。")

    ctx.response = "".join(parts)

    # --- NLP: Check readability vs student level using textstat ---
    try:
        reading_ease = textstat.flesch_reading_ease(ctx.response)
        level_thresholds = {"beginner": 60, "intermediate": 40, "advanced": 20}
        target = level_thresholds.get(ctx.student_level, 40)
        if reading_ease < target and ctx.student_level == "beginner":
            ctx.response += "\n\n📝 **提示：** 以上內容較為進階，建議搭配教材中的基礎說明一起閱讀。"
    except Exception:
        pass

    return ctx
