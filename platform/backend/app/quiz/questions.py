"""Quiz question bank — 3-5 questions per week, auto-graded."""

QUIZ_BANK: dict[int, list[dict]] = {
    1: [
        {"id": "w01q1", "question": "Python 中用來處理表格資料的主要套件是？",
         "options": ["NumPy", "Pandas", "Matplotlib", "Scikit-learn"], "answer": 1},
        {"id": "w01q2", "question": "Jupyter Notebook 的副檔名是？",
         "options": [".py", ".ipynb", ".jnb", ".nb"], "answer": 1},
        {"id": "w01q3", "question": "NumPy 的核心資料結構是？",
         "options": ["list", "DataFrame", "ndarray", "tensor"], "answer": 2},
    ],
    2: [
        {"id": "w02q1", "question": "EDA 的全稱是？",
         "options": ["Exploratory Data Analysis", "Extended Data Algorithm",
                     "Efficient Data Architecture", "Evaluated Data Approach"], "answer": 0},
        {"id": "w02q2", "question": "觀察兩個連續變數關係時最適合用什麼圖？",
         "options": ["長條圖", "散佈圖", "圓餅圖", "直方圖"], "answer": 1},
        {"id": "w02q3", "question": "相關係數 r = -0.95 代表什麼？",
         "options": ["幾乎無關", "強正相關", "強負相關", "中等正相關"], "answer": 2},
    ],
    3: [
        {"id": "w03q1", "question": "訓練集和測試集最常見的比例是？",
         "options": ["50/50", "70/30 或 80/20", "95/5", "60/40"], "answer": 1},
        {"id": "w03q2", "question": "交叉驗證的目的是？",
         "options": ["加速訓練", "減少模型偏差", "更可靠地評估模型效能", "增加訓練資料"], "answer": 2},
        {"id": "w03q3", "question": "K-fold 交叉驗證中 K=5，表示資料被分成？",
         "options": ["2 份", "5 份", "10 份", "K 由模型決定"], "answer": 1},
    ],
    4: [
        {"id": "w04q1", "question": "梯度下降中，學習率太大會導致？",
         "options": ["收斂太慢", "發散（損失不收斂）", "過擬合", "完美收斂"], "answer": 1},
        {"id": "w04q2", "question": "MSE 損失函數的全稱是？",
         "options": ["Mean Square Error", "Maximum Standard Error",
                     "Minimum Squared Estimation", "Mean Squared Error"], "answer": 3},
        {"id": "w04q3", "question": "線性回歸的假設函數形式是？",
         "options": ["y = wx + b", "y = sigmoid(wx + b)", "y = max(0, wx + b)", "y = e^(wx)"], "answer": 0},
    ],
    5: [
        {"id": "w05q1", "question": "邏輯迴歸的輸出範圍是？",
         "options": ["(-∞, +∞)", "[0, 1]", "[-1, 1]", "[0, ∞)"], "answer": 1},
        {"id": "w05q2", "question": "ROC 曲線的 AUC = 0.5 代表？",
         "options": ["完美分類器", "隨機猜測", "完全錯誤", "過擬合"], "answer": 1},
        {"id": "w05q3", "question": "決策邊界是指？",
         "options": ["損失函數的最低點", "模型區分不同類別的分界線",
                     "訓練集和測試集的分界", "特徵的最大值"], "answer": 1},
    ],
    6: [
        {"id": "w06q1", "question": "SVM 的目標是？",
         "options": ["最小化訓練誤差", "最大化分類間隔 (margin)",
                     "最小化特徵數", "最大化訓練樣本數"], "answer": 1},
        {"id": "w06q2", "question": "RBF 核函數可以處理什麼類型的資料？",
         "options": ["只能線性可分", "非線性可分", "只能二分類", "只能數值型"], "answer": 1},
        {"id": "w06q3", "question": "SVM 中的支撐向量是指？",
         "options": ["所有訓練資料", "距離決策邊界最近的資料點",
                     "誤分類的資料", "離群值"], "answer": 1},
    ],
    7: [
        {"id": "w07q1", "question": "決策樹的分裂準則 Gini Impurity 的最小值是？",
         "options": ["0（純節點）", "0.5", "1", "-1"], "answer": 0},
        {"id": "w07q2", "question": "隨機森林屬於哪種集成方法？",
         "options": ["Boosting", "Bagging", "Stacking", "Blending"], "answer": 1},
        {"id": "w07q3", "question": "GBDT 中的 GB 代表？",
         "options": ["Global Best", "Gradient Boosting", "Greedy Base", "General Balanced"], "answer": 1},
    ],
    8: [
        {"id": "w08q1", "question": "SHAP 值的理論基礎來自？",
         "options": ["微積分", "博弈論 (Shapley Value)", "資訊理論", "統計假設檢定"], "answer": 1},
        {"id": "w08q2", "question": "特徵重要度越高表示？",
         "options": ["該特徵值越大", "該特徵對模型預測的影響越大",
                     "該特徵的方差越大", "該特徵缺失值越少"], "answer": 1},
        {"id": "w08q3", "question": "SHAP 相較於 Permutation Importance 的優勢是？",
         "options": ["計算更快", "可以解釋個別預測", "只適用於樹模型", "不需要訓練"], "answer": 1},
    ],
    9: [
        {"id": "w09q1", "question": "StandardScaler 的轉換公式是？",
         "options": ["(x - min) / (max - min)", "(x - mean) / std",
                     "x / max", "log(x + 1)"], "answer": 1},
        {"id": "w09q2", "question": "One-Hot Encoding 適用於？",
         "options": ["連續型特徵", "類別型特徵", "時序特徵", "影像特徵"], "answer": 1},
        {"id": "w09q3", "question": "Scikit-learn 的 Pipeline 的好處是？",
         "options": ["加速訓練", "避免資料洩漏、程式碼更整潔",
                     "自動選特徵", "不需要調參"], "answer": 1},
    ],
    10: [
        {"id": "w10q1", "question": "GridSearchCV 的缺點是？",
         "options": ["計算量大（窮舉所有組合）", "無法找到最佳參數",
                     "只能用於線性模型", "不支援交叉驗證"], "answer": 0},
        {"id": "w10q2", "question": "學習曲線可以幫助診斷？",
         "options": ["計算速度", "過擬合或欠擬合", "資料品質", "特徵重要度"], "answer": 1},
        {"id": "w10q3", "question": "Train error 很低但 Val error 很高代表？",
         "options": ["欠擬合", "過擬合", "完美擬合", "資料太少"], "answer": 1},
    ],
    11: [
        {"id": "w11q1", "question": "ReLU 激活函數的定義是？",
         "options": ["max(0, x)", "1/(1+e^(-x))", "(e^x - e^(-x))/(e^x + e^(-x))", "x"], "answer": 0},
        {"id": "w11q2", "question": "Dropout 的作用是？",
         "options": ["加速訓練", "防止過擬合", "增加模型複雜度", "處理缺失值"], "answer": 1},
        {"id": "w11q3", "question": "Batch Normalization 標準化的是？",
         "options": ["輸入資料", "每一層的輸出（mini-batch 內）",
                     "損失函數", "學習率"], "answer": 1},
    ],
    12: [
        {"id": "w12q1", "question": "CNN 中卷積核的主要功能是？",
         "options": ["降低維度", "提取局部特徵", "增加參數量", "正規化"], "answer": 1},
        {"id": "w12q2", "question": "Max Pooling 的作用是？",
         "options": ["增加特徵圖大小", "降低空間維度、保留重要特徵",
                     "增加通道數", "初始化權重"], "answer": 1},
        {"id": "w12q3", "question": "Grad-CAM 可以告訴我們什麼？",
         "options": ["模型的準確率", "模型在影像中關注的區域",
                     "最佳超參數", "訓練時間"], "answer": 1},
    ],
    13: [
        {"id": "w13q1", "question": "RNN 的主要問題是？",
         "options": ["計算太快", "梯度消失/爆炸", "無法處理序列", "記憶體不足"], "answer": 1},
        {"id": "w13q2", "question": "LSTM 中的 Cell State 的作用是？",
         "options": ["儲存短期記憶", "長期資訊的傳遞通道",
                     "計算損失", "控制學習率"], "answer": 1},
        {"id": "w13q3", "question": "Transformer 的核心機制是？",
         "options": ["卷積", "自注意力 (Self-Attention)",
                     "池化", "梯度下降"], "answer": 1},
    ],
    14: [
        {"id": "w14q1", "question": "Early Stopping 在什麼時候停止訓練？",
         "options": ["訓練損失為零", "驗證損失不再下降", "到達指定 epoch",
                     "學習率為零"], "answer": 1},
        {"id": "w14q2", "question": "Learning Rate Scheduler 的目的是？",
         "options": ["固定學習率", "在訓練過程中動態調整學習率",
                     "增加模型參數", "加速資料載入"], "answer": 1},
        {"id": "w14q3", "question": "資料增強 (Data Augmentation) 的目的是？",
         "options": ["收集更多資料", "人工增加訓練多樣性以減少過擬合",
                     "清理雜訊", "減少訓練時間"], "answer": 1},
    ],
    15: [
        {"id": "w15q1", "question": "模型的 Accuracy = 95% 但表現很差，最可能的原因是？",
         "options": ["學習率太高", "類別不平衡（95%的樣本是同一類）",
                     "特徵太多", "訓練太久"], "answer": 1},
        {"id": "w15q2", "question": "什麼是模型的公平性 (Fairness)？",
         "options": ["模型準確率高", "對不同群體有相近的預測表現",
                     "模型訓練快", "使用最新演算法"], "answer": 1},
        {"id": "w15q3", "question": "Confusion Matrix 可以看出什麼？",
         "options": ["模型的訓練時間", "各類別的 TP/FP/FN/TN 分布",
                     "特徵的重要度", "學習率的變化"], "answer": 1},
    ],
    16: [
        {"id": "w16q1", "question": "MLOps 的核心理念是？",
         "options": ["只關注模型訓練", "自動化 ML 的完整生命週期",
                     "只使用雲端服務", "不需要監控"], "answer": 1},
        {"id": "w16q2", "question": "模型版本控制的好處是？",
         "options": ["加速訓練", "可以回溯和比較不同版本的模型",
                     "減少記憶體", "自動調參"], "answer": 1},
        {"id": "w16q3", "question": "資料漂移 (Data Drift) 是指？",
         "options": ["訓練資料太少", "生產環境的資料分佈與訓練資料不同",
                     "模型參數變化", "API 速度變慢"], "answer": 1},
    ],
    17: [
        {"id": "w17q1", "question": "RAG 的全稱是？",
         "options": ["Random Access Generation", "Retrieval-Augmented Generation",
                     "Recursive Attention Graph", "Reinforced Agent Guidance"], "answer": 1},
        {"id": "w17q2", "question": "文字嵌入 (Text Embedding) 的作用是？",
         "options": ["壓縮檔案", "將文字轉為固定長度的向量表示",
                     "翻譯文字", "加密文字"], "answer": 1},
        {"id": "w17q3", "question": "Prompt Engineering 中的 Few-shot 是指？",
         "options": ["不給範例", "給少量範例讓 LLM 理解任務格式",
                     "多次呼叫 API", "微調模型"], "answer": 1},
    ],
    18: [
        {"id": "w18q1", "question": "好的 ML 專題報告應包含？",
         "options": ["只有程式碼", "問題定義、方法、結果、反思",
                     "只有結果", "只有理論推導"], "answer": 1},
        {"id": "w18q2", "question": "模型部署後最重要的是？",
         "options": ["刪除原始資料", "持續監控模型效能",
                     "停止更新", "增加參數"], "answer": 1},
        {"id": "w18q3", "question": "反思 (Reflection) 在學習中的重要性是？",
         "options": ["浪費時間", "幫助理解自己的學習過程和不足",
                     "增加考試分數", "減少作業量"], "answer": 1},
    ],
}


def get_quiz(week: int) -> list[dict]:
    """Get quiz questions for a week (without answers)."""
    questions = QUIZ_BANK.get(week, [])
    return [{"id": q["id"], "question": q["question"], "options": q["options"]} for q in questions]


def grade_quiz(week: int, answers: dict[str, int]) -> dict:
    """Grade quiz answers. Returns score and corrections."""
    questions = QUIZ_BANK.get(week, [])
    if not questions:
        return {"score": 0, "total": 0, "results": []}

    results = []
    correct = 0
    for q in questions:
        user_ans = answers.get(q["id"])
        is_correct = user_ans == q["answer"]
        if is_correct:
            correct += 1
        results.append({
            "id": q["id"],
            "correct": is_correct,
            "correct_answer": q["answer"],
            "user_answer": user_ans,
        })

    return {
        "score": correct,
        "total": len(questions),
        "percentage": round(correct / len(questions) * 100, 1),
        "results": results,
    }
