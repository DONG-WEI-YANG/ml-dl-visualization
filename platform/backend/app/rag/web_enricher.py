"""Auto web enrichment for RAG knowledge base.

Fetches ML/DL educational content from Wikipedia (zh + en) daily.
Deduplicates via content hash before ingesting.
"""

import asyncio
import logging
import re
import random
from datetime import datetime, timedelta

import httpx

from .store import append_chunks, log_enrichment, init_rag_tables

try:
    from zhconv import convert as zhconv_convert
except ImportError:
    zhconv_convert = None

logger = logging.getLogger(__name__)

# ── Course topic → Wikipedia article mapping ──
# Each entry: (week, zh_title, en_title) — fetches both languages
TOPIC_ARTICLES = [
    # Week 1-2: Foundations
    (1, "Python", "Python_(programming_language)"),
    (1, "資料科學", "Data_science"),
    (2, "資料視覺化", "Data_visualization"),
    (2, "探索性資料分析", "Exploratory_data_analysis"),
    # Week 3: Supervised learning
    (3, "監督式學習", "Supervised_learning"),
    (3, "交叉驗證", "Cross-validation_(statistics)"),
    (3, "過適", "Overfitting"),
    # Week 4: Linear regression
    (4, "線性回歸", "Linear_regression"),
    (4, "梯度下降法", "Gradient_descent"),
    (4, "損失函數", "Loss_function"),
    # Week 5: Classification
    (5, "邏輯迴歸", "Logistic_regression"),
    (5, "ROC曲線", "Receiver_operating_characteristic"),
    (5, "決策邊界", "Decision_boundary"),
    # Week 6: SVM
    (6, "支持向量機", "Support-vector_machine"),
    (6, "核方法", "Kernel_method"),
    # Week 7: Tree models
    (7, "決策樹", "Decision_tree"),
    (7, "隨機森林", "Random_forest"),
    (7, "梯度提升", "Gradient_boosting"),
    # Week 8: Feature importance
    (8, "特徵選擇", "Feature_selection"),
    (8, "SHAP", "SHAP"),
    # Week 9: Feature engineering
    (9, "特徵工程", "Feature_engineering"),
    (9, "正規化_(機器學習)", "Feature_scaling"),
    # Week 10: Hyperparameter tuning
    (10, "超參數最佳化", "Hyperparameter_optimization"),
    (10, "學習曲線_(機器學習)", "Learning_curve_(machine_learning)"),
    # Week 11: Neural networks
    (11, "人工神經網路", "Artificial_neural_network"),
    (11, "激活函數", "Activation_function"),
    (11, "正則化_(數學)", "Regularization_(mathematics)"),
    (11, "批次正規化", "Batch_normalization"),
    # Week 12: CNN
    (12, "卷積神經網路", "Convolutional_neural_network"),
    (12, "類別啟動映射", "Class_activation_mapping"),
    # Week 13: RNN / Transformers
    (13, "循環神經網路", "Recurrent_neural_network"),
    (13, "長短期記憶", "Long_short-term_memory"),
    (13, "Transformer模型", "Transformer_(deep_learning_architecture)"),
    # Week 14: Training techniques
    (14, "反向傳播算法", "Backpropagation"),
    (14, "學習率", "Learning_rate"),
    (14, "Dropout_(神經網路)", "Dilution_(neural_networks)"),
    # Week 15: Evaluation / Fairness
    (15, "混淆矩陣", "Confusion_matrix"),
    (15, "機器學習中的偏見", "Fairness_(machine_learning)"),
    # Week 16: MLOps
    (16, "MLOps", "MLOps"),
    # Week 17: LLM / RAG
    (17, "大型語言模型", "Large_language_model"),
    (17, "檢索增強生成", "Retrieval-augmented_generation"),
    (17, "提示工程", "Prompt_engineering"),
]

# Rotate through topics — fetch a different subset each day
_ARTICLES_PER_RUN = 8


def _pick_daily_topics() -> list[tuple]:
    """Pick a rotating subset of topics based on the day of year."""
    day = datetime.now().timetuple().tm_yday
    n = len(TOPIC_ARTICLES)
    start = (day * _ARTICLES_PER_RUN) % n
    topics = []
    for i in range(_ARTICLES_PER_RUN):
        topics.append(TOPIC_ARTICLES[(start + i) % n])
    return topics


def _clean_wiki_text(text: str) -> str:
    """Remove Wikipedia markup artifacts from plain-text extracts."""
    # Remove {\displaystyle ...} LaTeX blocks
    text = re.sub(r"\{\\displaystyle\s[^}]*\}", "", text)
    # Remove remaining \displaystyle fragments
    text = re.sub(r"\\displaystyle\b", "", text)
    # Remove wiki template leftovers like {\\hat {...}}
    text = re.sub(r"\{\\[a-z]+\s*\{[^}]*\}\}", "", text)
    # Remove isolated LaTeX commands like \alpha, \beta, \sigma, \nabla etc.
    text = re.sub(r"\\(?:alpha|beta|gamma|delta|epsilon|sigma|theta|lambda|mu|nu|eta|nabla|partial|hat|vec|bar|tilde|dot|frac|sqrt|sum|prod|int|lim|log|exp|sin|cos|tan|mathbb|mathrm|mathbf|mathcal|operatorname|left|right|cdot|cdots|ldots|times|leq|geq|neq|approx|equiv|sim|infty|forall|exists|in|notin|subset|supset|cup|cap)\b", "", text)
    # Remove stray braces, carets, underscores from LaTeX
    text = re.sub(r"[\{\}]", "", text)
    text = re.sub(r"\s*\^\s*", "", text)
    text = re.sub(r"\s*_\s*", "", text)
    # Remove "参考文献", "外部連結", "扩展阅读", "參見" sections and everything after
    text = re.sub(r"\n==\s*(?:参考文献|參考文獻|外部連結|外部链接|扩展阅读|延伸閱讀|參見|参见|References|External links|See also|Further reading)\s*==[\s\S]*$", "", text)
    # Remove "(页面存档备份，存于互联网档案馆)" and similar
    text = re.sub(r"（页面存档备份，存于互联网档案馆）", "", text)
    text = re.sub(r"\（[^）]*存档[^）]*\）", "", text)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces
    text = re.sub(r"  +", " ", text)
    return text.strip()


def _to_traditional(text: str) -> str:
    """Convert simplified Chinese to traditional Chinese."""
    if zhconv_convert:
        return zhconv_convert(text, "zh-tw")
    return text


async def fetch_wikipedia_article(title: str, lang: str = "zh") -> str | None:
    """Fetch article extract from Wikipedia REST API."""
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title}"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, headers={"User-Agent": "MLDLVizBot/1.0"})
            if resp.status_code != 200:
                return None
            data = resp.json()
            extract = data.get("extract", "")
            if len(extract) < 50:
                return None
            return extract
    except Exception as e:
        logger.debug("Wikipedia fetch failed for %s/%s: %s", lang, title, e)
        return None


async def fetch_wikipedia_full(title: str, lang: str = "zh") -> str | None:
    """Fetch fuller article content via MediaWiki API (plain text extracts).
    Cleans markup artifacts and converts to traditional Chinese."""
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": 1,
        "exsectionformat": "plain",
        "format": "json",
    }
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(url, params=params, headers={"User-Agent": "MLDLVizBot/1.0"})
            if resp.status_code != 200:
                return None
            data = resp.json()
            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                extract = page.get("extract", "")
                if len(extract) > 100:
                    # Clean Wikipedia markup artifacts
                    extract = _clean_wiki_text(extract)
                    # Convert to traditional Chinese for zh articles
                    if lang == "zh":
                        extract = _to_traditional(extract)
                    return extract
            return None
    except Exception as e:
        logger.debug("Wikipedia full fetch failed for %s/%s: %s", lang, title, e)
        return None


def _chunk_article(text: str, max_chunk: int = 800) -> list[str]:
    """Split article text into chunks by sections or paragraphs."""
    # Split by section headers (== Section ==)
    sections = re.split(r"\n(?===+ .+ ==)", text)
    chunks = []
    for section in sections:
        section = section.strip()
        if not section or len(section) < 30:
            continue
        if len(section) <= max_chunk:
            chunks.append(section)
        else:
            # Split by paragraphs
            paragraphs = section.split("\n\n")
            current = ""
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                if len(current) + len(para) + 2 > max_chunk and current:
                    chunks.append(current.strip())
                    current = para
                else:
                    current = current + "\n\n" + para if current else para
            if current.strip() and len(current.strip()) > 30:
                chunks.append(current.strip())
    return chunks


async def enrich_from_web() -> dict:
    """Run one enrichment cycle: pick topics, fetch Wikipedia, ingest new chunks."""
    init_rag_tables()
    topics = _pick_daily_topics()
    all_chunks = []
    searched_topics = []

    for week, zh_title, en_title in topics:
        searched_topics.append(zh_title)

        # Fetch Chinese article (primary)
        text = await fetch_wikipedia_full(zh_title, "zh")
        if text:
            chunks = _chunk_article(text)
            for i, chunk in enumerate(chunks):
                # Extract first line as title
                first_line = chunk.split("\n")[0].strip().strip("= ")
                title = first_line[:60] if first_line else f"{zh_title} 段落{i+1}"
                all_chunks.append({
                    "id": f"web_zh_{zh_title}_{i:03d}",
                    "content": chunk,
                    "metadata": {
                        "week": week,
                        "file_type": "web-zh",
                        "title": title,
                        "source": f"wikipedia/zh/{zh_title}",
                    },
                })

        # Fetch English article (supplementary)
        text_en = await fetch_wikipedia_full(en_title, "en")
        if text_en:
            chunks_en = _chunk_article(text_en)
            for i, chunk in enumerate(chunks_en):
                first_line = chunk.split("\n")[0].strip().strip("= ")
                title = first_line[:60] if first_line else f"{en_title} section {i+1}"
                all_chunks.append({
                    "id": f"web_en_{en_title}_{i:03d}",
                    "content": chunk,
                    "metadata": {
                        "week": week,
                        "file_type": "web-en",
                        "title": title,
                        "source": f"wikipedia/en/{en_title}",
                    },
                })

        # Small delay between requests to be polite to Wikipedia
        await asyncio.sleep(1)

    # Deduplicated append
    result = append_chunks(all_chunks) if all_chunks else {"added": 0, "skipped": 0}

    topics_str = ", ".join(searched_topics)
    log_enrichment(
        source="wikipedia",
        chunks_found=len(all_chunks),
        chunks_added=result["added"],
        chunks_skipped=result["skipped"],
        topics=topics_str,
    )

    logger.info(
        "Web enrichment done: %d found, %d added, %d skipped (topics: %s)",
        len(all_chunks), result["added"], result["skipped"], topics_str,
    )
    return {
        "chunks_found": len(all_chunks),
        "chunks_added": result["added"],
        "chunks_skipped": result["skipped"],
        "topics_searched": searched_topics,
    }


# ── Background scheduler ──

_enrichment_task: asyncio.Task | None = None


async def _daily_loop():
    """Run enrichment once a day. First run waits a short delay after startup."""
    await asyncio.sleep(30)  # Let the app fully start
    while True:
        try:
            logger.info("Starting daily web enrichment...")
            await enrich_from_web()
        except Exception as e:
            logger.error("Web enrichment failed: %s", e)
        # Sleep 24 hours
        await asyncio.sleep(86400)


def start_daily_enrichment():
    """Start the background daily enrichment task."""
    global _enrichment_task
    if _enrichment_task is None or _enrichment_task.done():
        _enrichment_task = asyncio.create_task(_daily_loop())
        logger.info("Daily web enrichment scheduler started")


def stop_daily_enrichment():
    """Stop the background task."""
    global _enrichment_task
    if _enrichment_task and not _enrichment_task.done():
        _enrichment_task.cancel()
        _enrichment_task = None
