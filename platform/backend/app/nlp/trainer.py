"""Training data generator + model trainer for NLP layers.

Generates synthetic training data from domain knowledge,
trains TF-IDF + classifier models using scikit-learn.
"""

import pickle
import json
import random
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import numpy as np

MODEL_DIR = Path(__file__).parent.parent.parent / "data" / "nlp_models"

# ── Domain terms for synthetic data generation ──

ML_CONCEPTS = [
    "梯度下降", "學習率", "損失函數", "線性回歸", "邏輯迴歸",
    "決策邊界", "SVM", "核方法", "決策樹", "隨機森林",
    "特徵重要度", "SHAP", "交叉驗證", "過擬合", "欠擬合",
    "正則化", "超參數", "激活函數", "CNN", "RNN",
    "LSTM", "Transformer", "注意力機制", "Dropout", "BatchNorm",
    "反向傳播", "嵌入", "遷移學習", "集成學習", "Boosting",
    "Bagging", "特徵工程", "標準化", "PCA", "聚類",
    "gradient descent", "learning rate", "loss function",
    "linear regression", "logistic regression", "decision boundary",
    "random forest", "feature importance", "cross-validation",
    "overfitting", "underfitting", "regularization",
    "activation function", "backpropagation", "attention mechanism",
]

ACTIONS = [
    "實作", "使用", "訓練", "計算", "畫出", "呼叫",
    "設定", "調整", "優化", "執行", "建立", "選擇",
]

ERRORS = [
    "TypeError", "ValueError", "ImportError", "IndexError",
    "RuntimeError", "NameError", "AttributeError", "KeyError",
    "Shape mismatch", "維度不對", "NaN", "infinity",
]

PACKAGES = [
    "scikit-learn", "numpy", "pandas", "matplotlib",
    "pytorch", "tensorflow", "keras", "seaborn",
]


def _generate_intent_data() -> tuple[list[str], list[str]]:
    """Generate synthetic training examples for intent classification."""
    texts, labels = [], []
    r = random.Random(42)

    def add(label: str, templates: list[str], n: int = 1):
        for t in templates:
            for _ in range(n):
                texts.append(t)
                labels.append(label)

    for concept in ML_CONCEPTS:
        # definition
        add("definition", [
            f"什麼是{concept}",
            f"{concept}是什麼",
            f"何謂{concept}",
            f"{concept}的定義",
            f"請解釋{concept}的意思",
            f"what is {concept}",
            f"define {concept}",
        ])
        # how
        add("how", [
            f"如何{r.choice(ACTIONS)}{concept}",
            f"怎麼{r.choice(ACTIONS)}{concept}",
            f"{concept}的步驟",
            f"how to implement {concept}",
            f"how to use {concept}",
        ])
        # why
        add("why", [
            f"為什麼要用{concept}",
            f"{concept}的目的是什麼",
            f"為何需要{concept}",
            f"why use {concept}",
        ])
        # compare (pairs)
        other = r.choice([c for c in ML_CONCEPTS if c != concept])
        add("compare", [
            f"{concept}和{other}的差別",
            f"{concept} vs {other}",
            f"比較{concept}和{other}",
            f"difference between {concept} and {other}",
        ])
        # formula
        add("formula", [
            f"{concept}的公式",
            f"{concept}的數學推導",
            f"推導{concept}的方程式",
            f"{concept} formula derivation",
        ])
        # code
        add("code", [
            f"{concept}的程式碼怎麼寫",
            f"如何用 Python 實作{concept}",
            f"{concept} Python code",
            f"寫{concept}的函數",
        ])
        # parameter
        add("parameter", [
            f"{concept}的參數怎麼調",
            f"{concept}的超參數設定",
            f"how to tune {concept} parameters",
        ])
        # application
        add("application", [
            f"{concept}的應用場景",
            f"{concept}在業界怎麼用",
            f"real world application of {concept}",
        ])
        # intuition
        add("intuition", [
            f"{concept}的直覺理解",
            f"{concept}背後的原理",
            f"intuition behind {concept}",
        ])
        # deeper
        add("deeper", [
            f"想更深入了解{concept}",
            f"{concept}的進階內容",
            f"more details about {concept}",
        ])

    # example
    add("example", [
        "可以舉個例子嗎",
        "給我一個範例",
        "能不能示範一下",
        "show me an example",
        "can you demonstrate",
    ] * 5)

    # debug
    for err in ERRORS:
        add("debug", [
            f"出現 {err} 錯誤",
            f"程式報 {err}",
            f"跑出 {err} 怎麼辦",
            f"getting {err} error",
        ])
    add("debug", [
        "程式跑不動", "一直報錯", "結果不對", "bug 找不到",
        "output is wrong", "code not working", "crash when running",
    ] * 3)

    # performance
    add("performance", [
        "準確率只有 50%", "模型效能太差", "F1 score 很低",
        "如何提升模型表現", "驗證集分數一直很低", "loss 不下降",
        "accuracy too low", "how to improve performance",
    ] * 3)

    # data
    add("data", [
        "資料要怎麼前處理", "缺失值怎麼處理", "特徵要怎麼選",
        "資料集太小怎麼辦", "如何做資料增強", "類別不平衡",
        "how to preprocess data", "handle missing values",
    ] * 3)

    # visualization
    add("visualization", [
        "怎麼畫 loss 曲線", "如何視覺化決策邊界", "畫混淆矩陣",
        "matplotlib 畫圖", "畫出訓練曲線", "視覺化特徵分布",
        "how to plot", "visualize results",
    ] * 3)

    # prerequisite
    add("prerequisite", [
        "學這個需要什麼基礎", "需要先學什麼", "先備知識是什麼",
        "what prerequisites", "what should I learn first",
    ] * 4)

    # summary
    add("summary", [
        "這週的重點是什麼", "幫我整理重點", "複習重點",
        "總結一下", "key takeaways", "summary of this week",
    ] * 4)

    # troubleshoot
    for pkg in PACKAGES:
        add("troubleshoot", [
            f"{pkg} 裝不了",
            f"pip install {pkg} 失敗",
            f"{pkg} 版本衝突",
        ])
    add("troubleshoot", [
        "環境設定問題", "虛擬環境怎麼建", "Python 版本不對",
        "cannot install package", "version conflict",
    ] * 3)

    # general
    add("general", [
        "第四週的內容", "這週在教什麼", "課程大綱",
        "推薦什麼教科書", "有沒有延伸閱讀", "哪裡可以找到更多資源",
        "what is this week about", "course outline",
    ] * 4)

    # ── Additional natural student queries ──
    add("definition", [
        "老師，loss function 到底是什麼意思啊", "可以用比喻解釋 overfitting 嗎",
        "正則化我一直聽不懂它在幹嘛", "那個 attention mechanism 是什麼東西",
        "batch normalization 是做什麼用的啊", "softmax 跟 sigmoid 是一樣的嗎",
        "embedding 是怎麼把文字變成數字的", "什麼叫做模型的泛化能力",
    ] * 2)
    add("how", [
        "老師我不知道 pipeline 怎麼串起來", "sklearn 的 cross_val_score 怎麼用",
        "我想自己從頭寫一個神經網路", "怎麼用 matplotlib 畫 confusion matrix",
        "SHAP force plot 要怎麼生成", "transfer learning 的步驟是什麼",
    ] * 2)
    add("debug", [
        "老師我的 model.fit 跑到一半就當掉了", "為什麼我的 loss 一直是 nan",
        "shape mismatch 這個錯怎麼解", "我 import torch 結果說找不到",
        "跑 CNN 的時候 GPU 記憶體不夠", "predict 出來的結果全部都一樣",
    ] * 2)
    add("compare", [
        "隨機森林跟 XGBoost 我該用哪個", "Adam 和 SGD 哪個比較好",
        "CNN 跟 Transformer 處理影像誰好", "L1 跟 L2 正則化差在哪",
    ] * 2)
    add("visualization", [
        "我想畫一個很漂亮的 ROC 曲線", "怎麼用 seaborn 畫 heatmap",
        "plotly 可以做 3D 散佈圖嗎", "我想視覺化 CNN 的 feature map",
    ] * 2)
    add("performance", [
        "我的模型 val accuracy 一直卡在 60%", "loss 下降很慢怎麼辦",
        "precision 跟 recall 怎麼取捨", "F1 score 太低了有什麼方法",
    ] * 2)

    # ── Colloquial / realistic student phrasings (15-20 per category) ──

    add("debug", [
        "老師我不會用這個...", "我的code跑出來長這樣...", "這邊一直出error",
        "照著講義的code貼進去就噴錯了", "我照著notebook做但跑到第三格就死了",
        "為什麼我跑的結果跟助教不一樣", "model.predict 回傳的東西好奇怪",
        "我 fit 完之後 score 是0", "跑到 epoch 3 就 OOM 了", "kernel 一直重啟",
        "import error 一直出現但我有裝了啊", "為什麼 plot 出來是空白的",
        "我加了 dropout 結果更差了", "資料讀不進來一直報編碼錯誤",
        "這段code我看不懂為什麼要這樣寫", "陣列的 shape 一直不對",
        "老師幫我看一下這段程式碼哪裡有問題",
    ] * 2)

    add("compare", [
        "這個跟那個差在哪裡啊", "RNN 跟 LSTM 什麼時候用哪個",
        "batch normalization 跟 layer normalization 怎麼選",
        "random forest 跟 gradient boosting 哪個準", "precision 跟 accuracy 不一樣嗎",
        "standardization 跟 normalization 我搞混了", "grid search 跟 random search 差在哪",
        "early stopping 跟 regularization 哪個效果好", "softmax 跟 sigmoid 什麼時候用哪個",
        "bagging 跟 boosting 的差異是什麼", "validation set 跟 test set 為什麼要分開",
        "MSE 跟 MAE 怎麼選", "PyTorch 跟 TensorFlow 初學者該學哪個",
        "sklearn 的 Pipeline 跟自己寫 for loop 差在哪", "one-hot 跟 label encoding 什麼時候用",
        "微調跟從頭訓練差多少", "CNN 跟全連接差在哪裡",
    ] * 2)

    add("summary", [
        "可以再說一次嗎", "剛剛講的我沒有很懂可以整理一下嗎",
        "這堂課最重要的三件事是什麼", "考試前要複習哪些重點",
        "可以幫我做個這週的筆記嗎", "重點整理一下好嗎",
        "這單元的核心概念有哪些", "可以用一張表格總結嗎",
        "我需要一個複習大綱", "把這章節的關鍵字列出來",
        "回顧一下今天學了什麼", "能不能簡化一下剛才的內容",
        "用三句話總結這個主題", "幫我列出公式清單",
        "學到這邊應該要會哪些東西", "這週跟上週的關聯是什麼",
    ] * 2)

    add("troubleshoot", [
        "我的 jupyter notebook 開不起來", "Colab 一直斷線怎麼辦",
        "pip install 跑了好久都沒反應", "conda 環境壞了", "虛擬環境切不過去",
        "GPU 偵測不到", "CUDA 版本不對怎麼辦", "requirements.txt 裡面有衝突",
        "Docker 跑不起來", "Git clone 下來但 README 的步驟跑不動",
        "M1 Mac 裝 tensorflow 一直失敗", "我的 pip 是 Python 2 的",
        "怎麼確認我用的是哪個 Python", "PATH 設定好像有問題",
        "套件裝了但 import 找不到", "downgrade 版本之後其他套件也壞了",
    ] * 2)

    add("example", [
        "可以給我看一個完整的範例嗎", "有沒有實際案例", "能不能demo一下",
        "可以用真實資料集做給我看嗎", "幫我寫一個簡單的例子",
        "拜託示範一次怎麼用", "步驟做一遍給我看", "能不能舉一個生活中的例子",
        "用 iris 資料集跑一遍", "有沒有那種很簡單的入門範例",
        "老師可以 live coding 嗎", "可以帶我走一遍流程嗎",
        "有沒有可以直接拿來改的模板", "範例程式碼在哪裡",
        "我想看一個end-to-end的例子", "有沒有從資料到部署的完整示範",
    ] * 2)

    add("code", [
        "這個function的參數是什麼意思", "sklearn的API怎麼查",
        "for loop寫不出來", "list comprehension怎麼用", "lambda函數看不懂",
        "class 要怎麼定義", "怎麼把這段code寫得更簡潔", "哪邊有API文件",
        "我想把pandas的操作串起來", "怎麼用 numpy 做矩陣運算",
        "callback怎麼自己寫一個", "怎麼自己寫一個 DataLoader",
        "裝飾器在ML裡面怎麼用", "type hint要怎麼加",
        "我想把訓練過程包成一個class", "程式碼要怎麼模組化",
    ] * 2)

    add("parameter", [
        "learning rate要設多少比較好", "batch size 32 夠嗎",
        "epoch 要跑幾次才夠", "dropout rate通常設多少", "hidden layer要幾層",
        "C 在 SVM 裡面代表什麼要設多少", "max_depth 設太大會怎樣",
        "n_estimators 要多少才夠", "momentum 有必要調嗎", "weight decay是什麼",
        "kernel size 怎麼選", "stride 和 padding 怎麼算",
        "k 在 KNN 裡怎麼決定", "min_samples_split 要設多少",
        "regularization strength 怎麼調", "怎麼知道參數調得好不好",
    ] * 2)

    add("intuition", [
        "反向傳播用一句白話文解釋是什麼", "可以用比喻來解釋梯度下降嗎",
        "為什麼 CNN 可以辨認圖片", "Transformer 的核心想法是什麼",
        "注意力機制其實在做什麼事", "為什麼深度學習這麼有效",
        "overfitting 用生活例子怎麼解釋", "什麼叫做模型在「學習」",
        "embedding 的幾何意義是什麼", "softmax 為什麼可以當機率",
        "batch normalization 直覺上在幹嘛", "為什麼非線性很重要",
        "kernel trick的核心想法", "skip connection 為什麼有用",
        "feature map 可以想像成什麼", "loss landscape 長什麼樣子",
    ] * 2)

    add("data", [
        "我的資料不平衡怎麼辦", "需要多少資料才夠訓練", "怎麼做data augmentation",
        "anomaly detection 的資料怎麼準備", "label 要自己標嗎",
        "train test split 比例通常怎麼分", "stratified split 是什麼",
        "one-hot encoding太多維了怎麼辦", "時間序列的資料怎麼分割",
        "文字資料要怎麼轉成數字", "圖片資料太大怎麼處理",
        "synthetic data 可以拿來訓練嗎", "怎麼處理多標籤分類的資料",
        "缺失值要填平均值還是中位數", "outlier 要不要移除",
        "怎麼確認資料品質", "feature selection 怎麼做",
    ] * 2)

    add("application", [
        "機器學習在醫療怎麼用", "NLP 在業界有哪些應用", "推薦系統是怎麼做的",
        "自動駕駛用什麼模型", "ChatGPT 背後是什麼技術", "AI 在金融的應用",
        "影像辨識在製造業怎麼用", "這些東西畢業後用得到嗎",
        "面試會問這些嗎", "實際工作中機器學習怎麼用",
        "有沒有學生做過的有趣專案", "這學期學的東西可以拿來做什麼",
        "Kaggle 比賽跟實際工作差很多嗎", "創業可以用 ML 做什麼",
        "哪些產業最需要 ML 人才", "這個技術五年後還會用嗎",
    ] * 2)

    add("why", [
        "為什麼一定要做特徵工程", "batch size 為什麼會影響泛化",
        "為什麼 ReLU 比 sigmoid 好", "dropout 為什麼可以防止過擬合",
        "為什麼 ensemble 通常比單一模型好", "為什麼要用交叉驗證",
        "為什麼 CNN 要用 pooling", "為什麼 Transformer 不需要 RNN",
        "為什麼 batch normalization 能加速訓練", "為什麼不能只看 accuracy",
        "為什麼深度學習需要 GPU", "為什麼 Adam 比 SGD 常用",
        "為什麼要做 data augmentation", "這個假設為什麼很重要",
        "為什麼要用 log loss 而不是 MSE", "為什麼 label smoothing 有用",
    ] * 2)

    add("prerequisite", [
        "學 CNN 之前要先會什麼", "微積分忘光了還學得起來嗎",
        "需要很強的數學底子嗎", "線性代數要會到什麼程度",
        "不會寫程式可以學ML嗎", "統計要先學哪些",
        "高中數學夠嗎", "Python 要學到什麼程度才能開始",
        "我只會 Excel 可以嗎", "需要先學 SQL 嗎",
        "data science 跟 ML 要學的先備一樣嗎", "數學證明需要看嗎",
        "哪些課程是這門課的前置課", "微積分跟線代哪個更重要",
        "我是文組的學得起來嗎", "程式底子不好怎麼辦",
    ] * 2)

    add("deeper", [
        "有推薦的論文可以讀嗎", "這個概念有沒有更數學化的解釋",
        "SOTA 目前做到什麼程度", "有什麼進階的變體",
        "research 方向有哪些", "可以推薦延伸閱讀嗎",
        "這方面最新的發展是什麼", "業界的最佳實踐是什麼",
        "有沒有相關的開源專案可以看", "想要自己做研究要從哪裡開始",
        "想挑戰更難的題目有什麼推薦", "這篇 paper 的 contribution 是什麼",
        "ablation study 怎麼設計", "reproducibility 要注意什麼",
        "怎麼寫一篇好的 ML 論文", "有沒有 survey paper 推薦",
    ] * 2)

    add("formula", [
        "交叉熵的公式可以再推一次嗎", "softmax 的微分怎麼算",
        "backpropagation 的鏈式法則怎麼展開", "batch normalization 的公式",
        "attention score 的計算方式", "Adam optimizer 的更新公式",
        "L2 regularization 在 loss function 裡長什麼樣子",
        "information gain 的公式", "Gini impurity 怎麼算",
        "KL divergence 的公式是什麼", "ROC AUC 的數學定義",
        "bias-variance decomposition 的推導", "convolution 的數學表示",
        "gradient 向量怎麼計算", "Jacobian matrix 在 backprop 裡怎麼用",
        "self-attention 的 Q K V 是怎麼算的",
    ] * 2)

    add("general", [
        "老師好", "有問題想請教", "想問一下", "可以問問題嗎",
        "我是第一次上這門課", "這門課難嗎", "作業可以用ChatGPT嗎",
        "期末專題有什麼建議", "修這門課需要什麼背景", "考試是什麼形式",
        "有錄影可以回看嗎", "助教的office hour是什麼時候",
        "這門課的評分標準是什麼", "可以跨組合作嗎", "上課要帶筆電嗎",
        "有推薦的YouTube頻道嗎",
    ] * 2)

    return texts, labels


def _generate_emotion_data() -> tuple[list[str], list[str]]:
    """Generate synthetic training examples for emotion detection."""
    texts, labels = [], []

    def add(label: str, samples: list[str]):
        for s in samples:
            texts.append(s)
            labels.append(label)

    add("frustrated", [
        "完全不懂", "看不懂", "搞不懂", "到底在說什麼", "好難",
        "太難了", "做不出來", "試了好久都不行", "要放棄了", "崩潰",
        "已經試了很多次了", "一直錯", "超級煩", "頭痛", "無語",
        "我不管了就是不會", "完全沒頭緒", "什麼鬼", "越來越不懂",
        "so frustrated", "I give up", "too hard", "impossible",
        "I've been trying for hours", "nothing works",
    ] * 3)

    add("confused", [
        "不太懂", "不確定", "搞混了", "有點模糊", "分不清",
        "不太理解", "不太清楚", "好像懂又好像不懂", "有疑問",
        "這兩個有什麼不一樣", "哪裡不對", "我哪裡搞錯了",
        "a bit confused", "not sure", "I'm lost", "which one",
        "unclear to me", "hard to tell the difference",
    ] * 3)

    add("curious", [
        "好奇", "有趣", "想了解更多", "想知道", "請問",
        "為什麼會這樣", "怎麼可能", "原來如此", "很有意思",
        "cool", "interesting", "I wonder", "fascinating",
        "that's neat", "tell me more",
    ] * 3)

    add("confident", [
        "我覺得應該是", "我認為", "我理解了", "我知道",
        "確認一下", "這樣對不對", "是否正確", "幫我驗證",
        "I think it's", "I believe", "let me verify", "is this correct",
        "am I right", "I got it",
    ] * 3)

    add("neutral", [
        "請問", "我想問", "可以告訴我", "麻煩", "謝謝",
        "好的", "了解", "知道了", "OK", "嗯",
        "什麼是梯度下降", "如何設定參數", "這週教什麼",
        "hello", "hi", "thanks", "question about",
        "can you explain", "what does this mean",
    ] * 3)

    add("frustrated", [
        "老師我真的要崩潰了這個 bug 找了三個小時", "為什麼照著教材做還是不對",
        "明明跟範例一模一樣為什麼就是跑不動", "我覺得我可能不適合學程式",
        "每次改一個 bug 就冒出十個新 bug", "我已經重做了五次了還是一樣的問題",
    ] * 2)
    add("confused", [
        "所以 bias 到底是偏差還是偏移啊", "我搞不清楚 validation set 跟 test set",
        "這個跟上禮拜教的好像不一樣？", "我覺得這兩個演算法好像", "結果很奇怪不確定對不對",
    ] * 2)
    add("curious", [
        "GAN 可以用來生成什麼有趣的東西", "如果把 Transformer 用在其他領域呢",
        "最近有什麼很酷的 AI 應用", "AlphaFold 是怎麼做到的",
    ] * 2)
    add("confident", [
        "我覺得是因為 learning rate 太大所以 loss 爆掉", "這應該用 ReLU 比 sigmoid 好",
        "我理解了，就是用梯度去更新權重對吧", "老師我做出來了！結果 accuracy 有 92%",
    ] * 2)
    add("neutral", [
        "這週的作業什麼時候交", "Notebook 在哪裡下載", "有辦公室時間嗎",
        "可以推薦一些參考書嗎", "考試範圍包含哪些週", "今天教到哪裡了",
    ] * 2)

    # ── Additional nuanced emotion expressions (15-20 per category) ──

    add("frustrated", [
        "算了我直接看答案好了", "我已經放棄理解了直接背",
        "為什麼我跟同學用一樣的code結果不一樣", "一整天都在debug心累",
        "這個概念我看了三遍還是看不懂", "我覺得我腦子不夠用",
        "好煩每次都是shape不對", "怎麼這麼難啊別人都說很簡單",
        "我按照助教的步驟做但就是不行", "要交了但我連第一題都不會",
        "這什麼鬼東西根本看不懂", "完了完了要交了還沒做完",
        "越改越爛我的code", "我想退選了", "這也太複雜了吧",
        "每個字都看得懂合在一起就不懂", "我真的不知道該怎麼辦了",
        "已經嘗試了好多方法全部都失敗", "我快被這個error逼瘋了",
        "連最簡單的範例我都跑不出來",
    ] * 2)

    add("confused", [
        "嗯...我再想想", "等等...所以到底是哪一個", "我有點迷路了",
        "這邊轉彎太快了我跟不上", "感覺有點矛盾啊",
        "你剛才說的跟教材上寫的不太一樣", "所以結論是什麼",
        "我搞不清楚這是在做什麼", "為什麼答案跟我想的不一樣",
        "這個符號是什麼意思", "loss 跟 cost function 是一樣的嗎",
        "gradient 跟 derivative 差在哪", "我不太確定自己理解的對不對",
        "這段推導我跟丟了", "好像每個教材說的都不太一樣",
        "所以 pooling 到底在做什麼事", "我以為我懂了但好像又不太對",
        "這邊用的notation跟課本不一樣", "我的直覺跟數學結果對不上",
        "等一下，所以這兩個是同一件事嗎",
    ] * 2)

    add("curious", [
        "GAN 可以生成音樂嗎", "如果用在我的研究領域會怎樣",
        "這個技術最早是誰發明的", "有沒有什麼反直覺的結果",
        "那如果把兩個模型合在一起呢", "可以用在遊戲開發嗎",
        "最近的AI新聞有什麼值得關注的", "所以其實人腦也是這樣運作的嗎",
        "我突然想到一個有趣的問題", "如果feature都是noise會怎樣",
        "這方面有什麼open problem嗎", "未來ML會往什麼方向發展",
        "可以用 reinforcement learning 來做嗎", "diffusion model 跟 GAN 比呢",
        "如果資料是多模態的怎麼處理", "能不能做到完全不用label的分類",
        "我想試試看不同的approach", "這個問題有沒有更elegant的解法",
        "好酷喔可以再多講一點嗎", "如果我想自己設計一個新的layer呢",
    ] * 2)

    add("confident", [
        "太棒了我終於懂了", "原來是這樣啊我之前理解錯了",
        "我覺得用 Adam 加上 cosine annealing 應該可以", "對對對就是這個意思",
        "我剛剛自己推導出來了跟教材一樣", "我試了一下效果很好accuracy 95%",
        "我知道問題出在哪了是learning rate太大", "所以重點就是要避免overfitting對吧",
        "我把你說的實作出來了跑得很順利", "我理解了gradient就是坡度的意思",
        "我可以解釋給同學聽了", "這題我會做了可以看下一題嗎",
        "我覺得batch normalization就是把分布拉回來", "OK我get到了",
        "原來cross-validation是這樣用的我之前搞錯了",
        "我成功復現了paper的結果", "我自己寫了一個training loop可以跑了",
        "嗯嗯我確定是這樣沒錯", "我已經掌握了這個概念",
        "我把助教的程式碼改良了效果更好",
    ] * 2)

    add("neutral", [
        "好的謝謝", "我知道了", "了解", "收到", "嗯嗯",
        "那我先試試看", "OK我晚點再來問", "這樣啊",
        "我去翻翻講義", "好的我回去練習看看", "作業什麼時候交",
        "有沒有補充教材", "教材在哪裡下載", "下次上課是什麼時候",
        "可以告訴我這週重點嗎", "老師下課了嗎", "有推薦的線上課程嗎",
        "成績什麼時候出來", "期中考範圍到第幾週",
    ] * 2)

    return texts, labels


def train_models() -> dict:
    """Train all NLP models and save to disk."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    # 1. Intent classifier
    texts, labels = _generate_intent_data()
    intent_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(1, 4),
            max_features=8000,
            sublinear_tf=True,
        )),
        ("clf", LinearSVC(C=1.0, max_iter=2000, class_weight="balanced")),
    ])
    intent_pipeline.fit(texts, labels)
    cv_scores = cross_val_score(intent_pipeline, texts, labels, cv=3, scoring="accuracy")
    results["intent"] = {
        "samples": len(texts),
        "classes": len(set(labels)),
        "cv_accuracy": float(np.mean(cv_scores)),
    }
    with open(MODEL_DIR / "intent_model.pkl", "wb") as f:
        pickle.dump(intent_pipeline, f)

    # 2. Emotion classifier
    texts_e, labels_e = _generate_emotion_data()
    emotion_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(1, 3),
            max_features=5000,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")),
    ])
    emotion_pipeline.fit(texts_e, labels_e)
    cv_scores_e = cross_val_score(emotion_pipeline, texts_e, labels_e, cv=3, scoring="accuracy")
    results["emotion"] = {
        "samples": len(texts_e),
        "classes": len(set(labels_e)),
        "cv_accuracy": float(np.mean(cv_scores_e)),
    }
    with open(MODEL_DIR / "emotion_model.pkl", "wb") as f:
        pickle.dump(emotion_pipeline, f)

    # 3. Curriculum TF-IDF vectorizer (for topic extraction + re-ranking)
    from app.rag.store import get_db
    conn = get_db()
    rows = conn.execute("SELECT id, content, week, title FROM rag_chunks").fetchall()
    conn.close()

    if rows:
        chunk_texts = [r["content"] for r in rows]
        chunk_ids = [r["id"] for r in rows]
        chunk_meta = [{"id": r["id"], "week": r["week"], "title": r["title"]} for r in rows]

        corpus_tfidf = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=15000,
            sublinear_tf=True,
        )
        corpus_matrix = corpus_tfidf.fit_transform(chunk_texts)

        with open(MODEL_DIR / "corpus_tfidf.pkl", "wb") as f:
            pickle.dump({
                "vectorizer": corpus_tfidf,
                "matrix": corpus_matrix,
                "ids": chunk_ids,
                "meta": chunk_meta,
            }, f)

        results["corpus"] = {
            "chunks": len(chunk_texts),
            "features": corpus_tfidf.max_features,
        }
    else:
        results["corpus"] = {"chunks": 0, "note": "No RAG chunks found. Run /api/rag/ingest first."}

    # Save metadata
    with open(MODEL_DIR / "training_meta.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results


def load_model(name: str):
    """Load a trained model from disk."""
    path = MODEL_DIR / f"{name}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)
