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
    from opencc import OpenCC
    _opencc = OpenCC("s2twp")  # Simplified → Traditional (Taiwan phrases)
except ImportError:
    _opencc = None

try:
    from zhconv import convert as zhconv_convert
except ImportError:
    zhconv_convert = None

logger = logging.getLogger(__name__)

# ── Simplified Chinese detection regex ──
# Common Simplified-only characters NOT shared with Traditional
_SIMPLIFIED_ONLY = re.compile(
    r"[么个们仅从仓仿伙优伤体佣侠侣俩俪债倾偻储催像儿兑关兰兹养减凑几凤刘则创剂剑剧劝办务动劳势勋包协单卤卫却厅厌厍厕发变叠叹吓吕呐响哑哟唤啸嘘团园圆坏块坛坝坞垃垒垦垫场壮声处备够头夸夺奋奖奥妇妈姗娇婴媪嫔学宁宝实审宽将尝对导专属岁岗岛岭帅师帜帧帮干并广庆库应废开异弃张弹归当录彦忆志忧怀态怂怜总恋恻悦惊惧愿慑懒戏户才扑执扩扫扬扰找护报拟拥拧拨择挡挤挥挨捞损换据搅携摄撑撵撸操擞收效斗断旋无既时旷旸昙显晓晕暂术机杀杂权条来杨极构枪柜柠查栏标栈样桥档梦检棂椁榄槛横樱欢款歼毁毕毙氢汇汉汤汹沈沟沪泛泞洒浅济涌涛涩淀渊渔渗温湿溃滚滞满滤潇潜澜灭灯灵灿炉炼烁烂烧热焕煌熏爱爷牍犊状独狭狱猎猪猫献玛环现珐琼瓶璃瓮电疗疯盏盐监盖盘着矫矿硕碍础确码磷祝禀种稳窃窍窑窜竞笔笼筑简箩篓籁籴粗粮纠纤纪纫纱纯纸纹纺纽线绅练组细绊绍经绑绘结绝给统绣继绩绪续绳维绵综绿缅缓编缘缝缠缩缭缰缴网罗罚罢罴翘翠翻耀耸聂职联聪肃肤肠肾胀胁胆脉脑脓脸腊腻腾臭舰艰艺节芦苇苍范茧荆荣荤莱莲获萧萨蒋蒙蓝蔑蔡蔬虏虑虽蚀蛮蜗蝇蝉术袜装裤裹观览觉规视觉览证评诊词译试诗诚话诞误说请诸读课谁调谅谊谋谜谢谣谨谱贝贞负贡财贤败账货质贪贫贱购贯贰资赁赃赋赌赏赔赖赚赛赞赢赢赵趋跃踊踪蹄蹈蹑蹬轧轨轩轮软轴轻载辅辆辈辉辑输辩辫边达迁过运近返还这进远连迟适选递逊通逻遗邮邻郁鄙酝酱酿释里钉钓钙钟钢钥钩钮钱铁铃铅银铜铝铸铺链锁锅锈错锡键锣锤锥锦锰锻镇镊镐镜镰长门闭问闰闲间闷闹闻阀阁阅阉阔阙阳阴阵阶阻际陆陈陕陨险随隐隶难雾霸靓静韩韭顶顿颁颂预领颓颖频颜颠颤风饥饭饮饰饱饲饵饶饿馁馅馆馈馋馐馒驭驰驱驳驴驶驻驼驾骂骄骆骇验骗骚骤髓鬓鬼魂鱼鲁鲜鸟鸡鸣鸭鸿鹅鹊鹏鹤鹰麦麻黄龄龙龟]"
)

# Wikipedia garbage patterns to filter out
_WIKI_GARBAGE_PATTERNS = [
    re.compile(r"簡繁重定向"),
    re.compile(r"本重定向用來避免"),
    re.compile(r"請勿使用管道連結"),
    re.compile(r"並非純簡繁字體差異"),
    re.compile(r"使用了本連結的頁面可能需要更新"),
    re.compile(r"消歧義|消歧义"),
    re.compile(r"本條目存在以下問題"),
    re.compile(r"此條目需要"),
    re.compile(r"stub|小作品|小條目"),
    re.compile(r"^分類[:：]", re.MULTILINE),
    re.compile(r"Template[:：]"),
    re.compile(r"^\s*\|\s*\w+\s*=", re.MULTILINE),  # Wiki table params
]

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
    """Convert simplified Chinese to traditional Chinese.
    Prefers OpenCC (s2twp) for accuracy, falls back to zhconv."""
    if _opencc is not None:
        try:
            return _opencc.convert(text)
        except Exception:
            pass
    if zhconv_convert:
        return zhconv_convert(text, "zh-tw")
    return text


def _is_garbage_chunk(text: str) -> bool:
    """Detect Wikipedia redirect, disambiguation, and metadata garbage."""
    for pat in _WIKI_GARBAGE_PATTERNS:
        if pat.search(text):
            return True
    # Too short to be useful
    if len(text.strip()) < 40:
        return True
    # Mostly LaTeX/math symbols (more than 30% non-CJK non-ASCII non-alpha)
    alpha_cjk = len(re.findall(r'[\w\u4e00-\u9fff]', text))
    if len(text) > 50 and alpha_cjk / len(text) < 0.4:
        return True
    return False


def _strip_remaining_simplified(text: str) -> str:
    """Replace any remaining Simplified-only characters with empty string.
    This is a safety net after OpenCC conversion."""
    return _SIMPLIFIED_ONLY.sub("", text)


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
                # Filter garbage chunks (redirect pages, disambiguation, etc.)
                if _is_garbage_chunk(chunk):
                    continue
                # Safety net: strip any remaining Simplified characters
                chunk = _strip_remaining_simplified(chunk)
                if len(chunk.strip()) < 40:
                    continue
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
