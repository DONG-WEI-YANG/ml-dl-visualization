import { useState } from "react";

interface ConceptCard {
  title: string;
  description: string;
  keyPoints: string[];
}

const WEEK_CONCEPTS: Record<number, ConceptCard[]> = {
  1: [
    {
      title: "人工智慧 vs 機器學習 vs 深度學習",
      description: "AI 是總稱，ML 是 AI 的子集（從資料學習），DL 是 ML 的子集（使用多層神經網路）。",
      keyPoints: [
        "AI > ML > DL，三者是包含關係而非同義詞",
        "ML 的核心是「從資料中自動找出規律」，而非人工撰寫規則",
        "DL 在影像、語音、文字等領域特別強大，但需要大量資料",
      ],
    },
    {
      title: "監督式 vs 非監督式 vs 強化學習",
      description: "ML 的三大學習範式，取決於是否有標記資料和學習方式。",
      keyPoints: [
        "監督式：有標準答案（標籤），學會「預測」，如分類和回歸",
        "非監督式：無標籤，自動發現資料結構，如分群和降維",
        "強化學習：透過環境互動和獎勵信號學習決策，如遊戲 AI",
      ],
    },
    {
      title: "ML 工作流程",
      description: "完整的機器學習專案遵循：問題定義 → 資料收集 → 特徵工程 → 模型訓練 → 評估部署。",
      keyPoints: [
        "80% 的時間花在資料處理和特徵工程，而非模型訓練",
        "模型選擇取決於問題類型：回歸預測數值、分類預測類別",
        "評估指標（準確率、F1 等）要根據實際需求選擇",
      ],
    },
    {
      title: "過擬合與正則化",
      description: "模型在訓練資料上學得太好（記住雜訊），在新資料上反而表現差，需要正則化來防止。",
      keyPoints: [
        "訓練表現好但測試表現差 = 過擬合（像是背考古題但不會新題目）",
        "正則化（L1/L2）給模型加上「懲罰」，防止學得太複雜",
        "適當的資料分割（訓練/驗證/測試）是發現過擬合的關鍵",
      ],
    },
    {
      title: "Python 資料科學環境",
      description: "Python 是 ML/DL 最主流的語言，搭配 NumPy、Pandas、scikit-learn 等套件建構完整的開發環境。",
      keyPoints: [
        "NumPy 負責數值計算、Pandas 負責資料處理、Matplotlib 負責繪圖",
        "scikit-learn 是最常用的 ML 套件，PyTorch 是最常用的 DL 框架",
        "Jupyter Notebook 讓你逐步執行程式碼並即時看到結果",
      ],
    },
  ],
  2: [
    {
      title: "資料概覽與摘要統計",
      description: "使用 describe()、info()、shape 快速了解資料集的基本特性。",
      keyPoints: [
        "describe() 顯示數值欄位的統計摘要",
        "info() 顯示資料型態和缺失值數量",
        "nunique() 可檢查類別型特徵的種類數",
      ],
    },
    {
      title: "缺失值處理",
      description: "缺失值會影響模型訓練，需要根據情況選擇刪除或填補策略。",
      keyPoints: [
        "isnull().sum() 統計各欄位缺失數量",
        "fillna 可用平均值、中位數或眾數填補",
        "大量缺失（>50%）的欄位通常直接刪除",
      ],
    },
    {
      title: "分佈視覺化",
      description: "透過直方圖、箱型圖等了解特徵分佈，發現異常值和偏態。",
      keyPoints: [
        "直方圖（hist）觀察單變數分佈",
        "箱型圖（boxplot）可快速辨識離群值",
        "偏態資料可能需要 log 轉換",
      ],
    },
    {
      title: "相關性分析",
      description: "分析特徵之間的相關性，找出對目標變數影響最大的特徵。",
      keyPoints: [
        "corr() 計算 Pearson 相關係數矩陣",
        "heatmap 視覺化相關矩陣更直觀",
        "高度相關的特徵可能需要去除冗餘",
      ],
    },
    {
      title: "異常值偵測",
      description: "異常值可能是資料錯誤或真實極端值，需要適當處理。",
      keyPoints: [
        "IQR 法：超過 Q1-1.5*IQR 或 Q3+1.5*IQR 為異常",
        "Z-score 法：超過 3 個標準差為異常",
        "處理方式包括刪除、截斷或轉換",
      ],
    },
  ],
  3: [
    {
      title: "訓練集與測試集分割",
      description: "將資料分為訓練集和測試集，訓練集用於建模，測試集用於評估泛化能力。",
      keyPoints: [
        "train_test_split 是最常用的分割方法",
        "通常比例為 70-80% 訓練、20-30% 測試",
        "設定 random_state 確保結果可重現",
      ],
    },
    {
      title: "K-Fold 交叉驗證",
      description: "將資料分成 K 份，輪流用每份作測試集，減少評估結果的變異性。",
      keyPoints: [
        "K=5 或 K=10 是常見的選擇",
        "每個樣本都會被用來訓練和測試各一次",
        "cross_val_score 可以一行完成交叉驗證",
      ],
    },
    {
      title: "分層抽樣",
      description: "確保分割後各子集中目標變數的類別比例與原始資料一致。",
      keyPoints: [
        "stratify 參數確保類別比例平衡",
        "對不平衡資料集特別重要",
        "StratifiedKFold 結合交叉驗證和分層抽樣",
      ],
    },
    {
      title: "資料洩漏",
      description: "訓練時使用了測試集的資訊，會導致模型評估過於樂觀。",
      keyPoints: [
        "特徵工程必須只在訓練集上 fit",
        "時間序列資料不能隨機分割",
        "Pipeline 可以幫助避免資料洩漏",
      ],
    },
    {
      title: "驗證集的角色",
      description: "驗證集用於調參和模型選擇，是訓練集和測試集之間的橋樑。",
      keyPoints: [
        "三段式分割：訓練、驗證、測試",
        "驗證集用來選超參數，測試集只用最後評估",
        "資料量少時用交叉驗證代替固定驗證集",
      ],
    },
  ],
  4: [
    {
      title: "梯度下降法",
      description: "透過反覆調整參數來最小化損失函數的最佳化演算法。",
      keyPoints: [
        "學習率控制每次更新的步伐大小",
        "太大會震盪不收斂，太小會收斂極慢",
        "批次、小批次和隨機梯度下降是三種變體",
      ],
    },
    {
      title: "損失函數",
      description: "衡量模型預測值與真實值之間差距的函數，是優化的目標。",
      keyPoints: [
        "MSE（均方誤差）是回歸問題最常用的損失",
        "損失函數的梯度指出參數調整的方向",
        "凸函數保證梯度下降能找到全域最佳解",
      ],
    },
    {
      title: "線性回歸",
      description: "用線性函數 y = wx + b 擬合資料，是最基礎的監督式學習模型。",
      keyPoints: [
        "最小二乘法可以直接求解最佳參數",
        "R² 分數衡量模型解釋變異的比例",
        "多元線性回歸可處理多個特徵",
      ],
    },
    {
      title: "正規化（Regularization）",
      description: "在損失函數中加入懲罰項，防止模型過擬合。",
      keyPoints: [
        "L1（Lasso）傾向產生稀疏解，可做特徵選擇",
        "L2（Ridge）讓權重平均變小，更穩定",
        "正規化強度 α 需要透過交叉驗證調整",
      ],
    },
    {
      title: "收斂與學習曲線",
      description: "觀察損失隨迭代次數變化的曲線，判斷訓練狀態。",
      keyPoints: [
        "損失穩定不再下降代表已收斂",
        "訓練損失下降但驗證損失上升代表過擬合",
        "Early stopping 可在最佳時機停止訓練",
      ],
    },
  ],
  5: [
    {
      title: "邏輯迴歸",
      description: "使用 Sigmoid 函數將線性輸出轉換為機率，用於二元分類問題。",
      keyPoints: [
        "輸出值介於 0 和 1 之間，代表正類機率",
        "決策閾值通常設為 0.5，可依需求調整",
        "使用交叉熵（Cross-entropy）作為損失函數",
      ],
    },
    {
      title: "決策邊界",
      description: "模型在特徵空間中劃出的分界線，將不同類別分開。",
      keyPoints: [
        "線性模型的決策邊界是直線（或超平面）",
        "非線性模型可以產生曲線或複雜形狀的邊界",
        "視覺化決策邊界有助於理解模型行為",
      ],
    },
    {
      title: "多類別分類",
      description: "當目標變數有三個以上的類別時，需要延伸二元分類方法。",
      keyPoints: [
        "One-vs-Rest（OvR）為每個類別訓練一個分類器",
        "Softmax 回歸可直接處理多類別問題",
        "混淆矩陣可觀察各類別之間的誤判情況",
      ],
    },
    {
      title: "Sigmoid 與 Softmax",
      description: "兩種將實數值轉換為機率的活化函數。",
      keyPoints: [
        "Sigmoid 用於二元分類，輸出單一機率值",
        "Softmax 用於多類別，輸出機率分佈（總和為 1）",
        "Softmax 是 Sigmoid 的多類別推廣",
      ],
    },
  ],
  6: [
    {
      title: "SVM 核心概念",
      description: "支撐向量機找到最大化間距的超平面來分隔資料。",
      keyPoints: [
        "最大間距（Maximum Margin）是 SVM 的核心目標",
        "支撐向量是距離決策邊界最近的資料點",
        "軟間距允許部分資料點越界，C 值控制容錯程度",
      ],
    },
    {
      title: "核函數（Kernel Trick）",
      description: "將資料映射到高維空間，使非線性可分的資料變得可分。",
      keyPoints: [
        "線性核適合高維稀疏資料（如文本）",
        "RBF 核適合大多數非線性問題",
        "核函數避免了顯式計算高維映射的成本",
      ],
    },
    {
      title: "超參數 C 和 Gamma",
      description: "C 控制間距與誤分類的權衡，Gamma 控制核函數的影響範圍。",
      keyPoints: [
        "C 大 → 嚴格分類、可能過擬合",
        "Gamma 大 → 影響範圍小、決策邊界更複雜",
        "Grid Search 可系統性搜尋最佳超參數組合",
      ],
    },
    {
      title: "SVM 的優缺點",
      description: "SVM 在中小型資料集和高維空間表現優秀，但大資料集計算成本高。",
      keyPoints: [
        "高維空間中仍然有效（如基因表達資料）",
        "訓練時間隨資料量快速增長 O(n²~n³)",
        "需要特徵縮放（StandardScaler）才能發揮最佳效果",
      ],
    },
  ],
  7: [
    {
      title: "決策樹",
      description: "透過一系列 if-else 規則將資料空間遞迴分割的分類/回歸模型。",
      keyPoints: [
        "Gini 不純度或資訊增益決定分割準則",
        "樹越深越容易過擬合，需要剪枝",
        "決策樹的可解釋性是最大優勢",
      ],
    },
    {
      title: "隨機森林",
      description: "結合多棵決策樹的投票結果，透過多樣性降低過擬合。",
      keyPoints: [
        "Bagging：每棵樹用不同的隨機子樣本訓練",
        "隨機特徵選擇增加樹之間的差異性",
        "n_estimators 通常設 100-500 棵樹",
      ],
    },
    {
      title: "梯度提升（Gradient Boosting）",
      description: "串聯式地訓練弱學習器，每棵新樹學習前面樹的殘差。",
      keyPoints: [
        "XGBoost、LightGBM 是高效實作",
        "學習率（learning_rate）控制每棵樹的貢獻度",
        "通常比隨機森林準確但更容易過擬合",
      ],
    },
    {
      title: "Bagging vs Boosting",
      description: "兩種主要的集成學習策略，各有優缺點。",
      keyPoints: [
        "Bagging 平行訓練、降低變異度（如隨機森林）",
        "Boosting 串聯訓練、降低偏差（如 XGBoost）",
        "Stacking 結合多種不同模型的預測",
      ],
    },
    {
      title: "過擬合控制",
      description: "集成模型透過控制樹的複雜度和數量來防止過擬合。",
      keyPoints: [
        "max_depth 限制樹的最大深度",
        "min_samples_leaf 設定葉節點最少樣本數",
        "學習曲線幫助判斷是否需要更多資料",
      ],
    },
  ],
  8: [
    {
      title: "特徵重要度",
      description: "量化每個特徵對模型預測的貢獻程度，幫助理解模型決策依據。",
      keyPoints: [
        "樹模型的內建重要度基於分割次數或不純度降低",
        "Permutation Importance 打亂特徵值觀察準確度下降",
        "不同方法可能給出不同排序，需要綜合判斷",
      ],
    },
    {
      title: "SHAP 值",
      description: "基於 Shapley 值的解釋方法，公平地分配每個特徵對預測的貢獻。",
      keyPoints: [
        "SHAP 值有堅實的博弈論數學基礎",
        "正值推高預測，負值拉低預測",
        "Summary plot 全面展示特徵的影響模式",
      ],
    },
    {
      title: "局部解釋 vs 全域解釋",
      description: "局部解釋說明單筆預測，全域解釋概括整個模型行為。",
      keyPoints: [
        "LIME 是另一種流行的局部解釋方法",
        "Force plot 展示單一預測的推力分解",
        "全域解釋幫助發現模型的系統性偏差",
      ],
    },
    {
      title: "可解釋性的重要性",
      description: "在醫療、金融等領域，模型的可解釋性是部署的必要條件。",
      keyPoints: [
        "黑箱模型需要事後解釋工具輔助",
        "可解釋性幫助發現資料洩漏和偏差",
        "歐盟 GDPR 規定人民有權得到演算法決策的解釋",
      ],
    },
  ],
  9: [
    {
      title: "特徵縮放",
      description: "將特徵值縮放到相近範圍，避免數值範圍大的特徵主導模型。",
      keyPoints: [
        "StandardScaler：轉換為均值 0、標準差 1",
        "MinMaxScaler：縮放到 [0, 1] 區間",
        "必須在訓練集上 fit，再 transform 測試集",
      ],
    },
    {
      title: "特徵編碼",
      description: "將類別型特徵轉換為數值型，讓模型能夠處理。",
      keyPoints: [
        "One-Hot Encoding 適合無序類別",
        "Label Encoding 適合有序類別",
        "高基數特徵可用 Target Encoding",
      ],
    },
    {
      title: "特徵選擇",
      description: "從大量特徵中挑選最有用的子集，提升模型效能和可解釋性。",
      keyPoints: [
        "Filter 方法用統計指標篩選（如相關係數）",
        "Wrapper 方法用模型效能來挑選（如 RFE）",
        "Embedded 方法在訓練中同時選擇（如 Lasso）",
      ],
    },
    {
      title: "Sklearn Pipeline",
      description: "將前處理和建模步驟串成流水線，確保一致性並避免資料洩漏。",
      keyPoints: [
        "Pipeline 依序執行 fit/transform 和 predict",
        "ColumnTransformer 對不同欄位應用不同處理",
        "Pipeline 可以直接放入交叉驗證和 Grid Search",
      ],
    },
    {
      title: "特徵工程技巧",
      description: "創造新特徵或轉換現有特徵，幫助模型更好地學習資料中的模式。",
      keyPoints: [
        "多項式特徵捕捉非線性關係",
        "日期時間可拆分為年、月、星期等",
        "領域知識是最好的特徵工程指南",
      ],
    },
  ],
  10: [
    {
      title: "超參數調校",
      description: "系統性地搜尋模型超參數的最佳組合，提升模型效能。",
      keyPoints: [
        "Grid Search 窮舉所有組合，保證找到最佳",
        "Random Search 隨機取樣，效率更高",
        "Bayesian Optimization 用歷史結果指導搜尋",
      ],
    },
    {
      title: "學習曲線",
      description: "觀察模型隨訓練資料量增加的表現變化，診斷過擬合或欠擬合。",
      keyPoints: [
        "訓練分數高、驗證分數低 → 過擬合（高變異度）",
        "兩條曲線都低 → 欠擬合（高偏差）",
        "兩條曲線收斂且接近 → 理想狀態",
      ],
    },
    {
      title: "偏差-變異度權衡",
      description: "模型複雜度增加時，偏差降低但變異度升高，需找到平衡點。",
      keyPoints: [
        "偏差：模型假設與真實函數的差距",
        "變異度：模型對不同訓練集的敏感程度",
        "總誤差 = 偏差² + 變異度 + 不可減誤差",
      ],
    },
    {
      title: "模型選擇策略",
      description: "依據資料特性和目標選擇合適的模型和超參數。",
      keyPoints: [
        "先用簡單模型建立 baseline",
        "逐步嘗試更複雜的模型",
        "用交叉驗證比較模型，避免過度調參",
      ],
    },
  ],
  11: [
    {
      title: "感知器與神經元",
      description: "神經網路的基本單元，接收加權輸入、加偏差、通過活化函數輸出。",
      keyPoints: [
        "y = activation(Σ(wi·xi) + b)",
        "權重 w 決定每個輸入的重要性",
        "偏差 b 讓決策邊界不必通過原點",
      ],
    },
    {
      title: "活化函數",
      description: "引入非線性，讓神經網路能學習複雜的函數映射。",
      keyPoints: [
        "ReLU 是目前最常用的活化函數",
        "Sigmoid 和 Tanh 有梯度消失問題",
        "沒有活化函數，多層網路等同於單層線性模型",
      ],
    },
    {
      title: "反向傳播",
      description: "利用鏈式法則從輸出層向輸入層傳遞梯度，更新所有參數。",
      keyPoints: [
        "前向傳播計算預測值和損失",
        "反向傳播計算每個參數的梯度",
        "梯度下降法根據梯度更新參數",
      ],
    },
    {
      title: "多層感知器（MLP）",
      description: "包含輸入層、一或多個隱藏層和輸出層的全連接神經網路。",
      keyPoints: [
        "隱藏層數和神經元數量決定模型容量",
        "Dropout 隨機關閉神經元防止過擬合",
        "Batch Normalization 加速訓練並穩定梯度",
      ],
    },
    {
      title: "損失函數與最佳化器",
      description: "損失函數定義優化目標，最佳化器決定參數更新策略。",
      keyPoints: [
        "分類用 CrossEntropy，回歸用 MSE",
        "Adam 結合 Momentum 和 RMSProp 的優點",
        "學習率排程可以在訓練過程中動態調整",
      ],
    },
  ],
  12: [
    {
      title: "卷積層",
      description: "使用可學習的濾波器在輸入上滑動，提取局部特徵。",
      keyPoints: [
        "濾波器大小（如 3x3）定義感受野",
        "步長（stride）控制濾波器移動的間隔",
        "填充（padding）保持輸出尺寸不變",
      ],
    },
    {
      title: "池化層",
      description: "縮小特徵圖尺寸，減少計算量並提供平移不變性。",
      keyPoints: [
        "Max Pooling 取區域最大值，保留最強特徵",
        "Average Pooling 取區域平均值，較平滑",
        "常用 2x2 池化，尺寸減半",
      ],
    },
    {
      title: "特徵層次",
      description: "CNN 自動學習從低階到高階的特徵表示。",
      keyPoints: [
        "淺層學邊緣、紋理等基礎特徵",
        "中層組合成物件部件（眼、鼻、耳）",
        "深層識別整體概念和語義",
      ],
    },
    {
      title: "經典 CNN 架構",
      description: "LeNet、AlexNet、VGG、ResNet 等經典架構推動了深度學習革命。",
      keyPoints: [
        "ResNet 的跳躍連接解決了深層網路的梯度消失",
        "VGG 證明了深度的重要性（3x3 小卷積核）",
        "遷移學習可以利用預訓練模型的特徵表示",
      ],
    },
    {
      title: "影像資料增強",
      description: "對訓練影像做隨機變換（旋轉、翻轉等）來增加資料多樣性。",
      keyPoints: [
        "隨機裁剪、翻轉、旋轉是常見的增強方式",
        "Color Jitter 改變亮度、對比度、飽和度",
        "資料增強只在訓練時使用，測試時不使用",
      ],
    },
  ],
  13: [
    {
      title: "RNN 循環神經網路",
      description: "處理序列資料的網路，隱藏狀態在時間步之間傳遞資訊。",
      keyPoints: [
        "h_t = f(W·h_{t-1} + U·x_t + b)",
        "適合處理文字、語音、時間序列等",
        "原始 RNN 有長期依賴的梯度消失問題",
      ],
    },
    {
      title: "LSTM 長短期記憶",
      description: "透過門控機制（遺忘門、輸入門、輸出門）解決長期依賴問題。",
      keyPoints: [
        "Cell State 是長期記憶的傳送帶",
        "遺忘門決定丟棄哪些舊資訊",
        "GRU 是簡化版的 LSTM，效果相近",
      ],
    },
    {
      title: "注意力機制",
      description: "讓模型學習聚焦在輸入序列中最相關的部分。",
      keyPoints: [
        "計算 Query 和 Key 的相似度得到權重",
        "用權重加權 Value 得到上下文向量",
        "Self-Attention 讓序列中每個位置關注其他位置",
      ],
    },
    {
      title: "Transformer 架構",
      description: "完全基於注意力機制的架構，捨棄了循環結構，可以平行計算。",
      keyPoints: [
        "Multi-Head Attention 從多個角度關注資訊",
        "Positional Encoding 注入位置資訊",
        "BERT、GPT 等大語言模型都基於 Transformer",
      ],
    },
    {
      title: "序列到序列模型",
      description: "Encoder-Decoder 架構處理輸入序列和輸出序列長度不同的任務。",
      keyPoints: [
        "機器翻譯是經典的 Seq2Seq 任務",
        "Encoder 壓縮輸入，Decoder 生成輸出",
        "Beam Search 在解碼時搜尋較佳序列",
      ],
    },
  ],
  14: [
    {
      title: "學習率策略",
      description: "動態調整學習率可以加速訓練並提升最終效能。",
      keyPoints: [
        "Warmup 先用小學習率穩定訓練再逐步增大",
        "Cosine Annealing 週期性降低學習率",
        "ReduceLROnPlateau 在損失停滯時降低學習率",
      ],
    },
    {
      title: "正則化技術",
      description: "防止深度學習模型過擬合的多種手段。",
      keyPoints: [
        "Dropout 隨機關閉神經元，等效於模型集成",
        "Weight Decay 等同於 L2 正則化",
        "資料增強也是一種隱式的正則化",
      ],
    },
    {
      title: "Batch Normalization",
      description: "在每一層的輸入做正規化，加速訓練並降低對初始化的敏感度。",
      keyPoints: [
        "對每個 mini-batch 正規化為均值 0、方差 1",
        "γ 和 β 可學習的縮放和偏移參數",
        "推理時使用訓練期間累積的統計量",
      ],
    },
    {
      title: "梯度問題與對策",
      description: "深層網路訓練中常見的梯度消失和梯度爆炸問題。",
      keyPoints: [
        "梯度裁剪（Gradient Clipping）防止爆炸",
        "殘差連接幫助梯度流通",
        "適當的權重初始化（如 He、Xavier）很重要",
      ],
    },
    {
      title: "混合精度訓練",
      description: "使用 FP16 和 FP32 混合計算，加速訓練並節省記憶體。",
      keyPoints: [
        "前向和反向傳播用 FP16 加速",
        "權重更新用 FP32 保持精度",
        "Loss Scaling 防止 FP16 梯度下溢",
      ],
    },
  ],
  15: [
    {
      title: "評估指標",
      description: "根據問題類型選擇合適的評估指標來衡量模型效能。",
      keyPoints: [
        "Accuracy 不適合不平衡資料集",
        "Precision 和 Recall 適合不同的錯誤代價",
        "F1-Score 是 Precision 和 Recall 的調和平均",
      ],
    },
    {
      title: "混淆矩陣",
      description: "以表格形式展示分類結果，清楚呈現各類別的預測情況。",
      keyPoints: [
        "TP、TN、FP、FN 四種分類結果",
        "對角線元素越大代表模型越好",
        "非對角線揭示容易混淆的類別對",
      ],
    },
    {
      title: "ROC 曲線與 AUC",
      description: "繪製不同閾值下的 TPR vs FPR，AUC 面積衡量整體分類能力。",
      keyPoints: [
        "AUC = 0.5 代表隨機猜測",
        "AUC 越接近 1 代表模型越好",
        "ROC 曲線不受類別不平衡的影響",
      ],
    },
    {
      title: "模型公平性",
      description: "確保模型對不同群體（性別、種族等）的預測不存在系統性偏差。",
      keyPoints: [
        "Demographic Parity：各群體正預測比例相同",
        "Equal Opportunity：各群體 TPR 相同",
        "偏差可能來自資料、標籤或模型本身",
      ],
    },
    {
      title: "不平衡資料處理",
      description: "當某類別樣本數遠少於其他類別時，需要特殊處理策略。",
      keyPoints: [
        "SMOTE 合成少數類的新樣本",
        "class_weight='balanced' 調整類別權重",
        "下採樣多數類也是常見做法",
      ],
    },
  ],
  16: [
    {
      title: "MLOps 概念",
      description: "將 DevOps 理念應用到機器學習工作流程，實現可重現、可監控的模型管理。",
      keyPoints: [
        "版本控制不只管程式碼，也管資料和模型",
        "CI/CD 自動化模型訓練、測試和部署",
        "MLOps 彌合開發與維運的鴻溝",
      ],
    },
    {
      title: "實驗追蹤",
      description: "記錄每次實驗的超參數、指標和產出物，方便比較和重現。",
      keyPoints: [
        "MLflow 是流行的實驗追蹤工具",
        "記錄超參數、指標、模型檔案和環境",
        "版本化的實驗讓研究可重現",
      ],
    },
    {
      title: "模型服務化",
      description: "將訓練好的模型封裝成 API 服務，供應用程式呼叫。",
      keyPoints: [
        "FastAPI/Flask 可快速建立預測 API",
        "Docker 容器化確保環境一致",
        "批次推理和即時推理的架構不同",
      ],
    },
    {
      title: "模型監控",
      description: "在生產環境中持續監控模型表現，偵測模型退化。",
      keyPoints: [
        "資料漂移：輸入分佈隨時間改變",
        "概念漂移：輸入和輸出的關係改變",
        "設定警報機制在效能下降時通知團隊",
      ],
    },
    {
      title: "可重現性",
      description: "確保實驗結果可以被完整重現，是科學研究的基本要求。",
      keyPoints: [
        "固定隨機種子（random seed）",
        "記錄完整的套件版本（requirements.txt）",
        "DVC 管理大型資料集和模型檔案的版本",
      ],
    },
  ],
  17: [
    {
      title: "RAG 檢索增強生成",
      description: "結合外部知識檢索和語言模型生成，提供更準確且可溯源的回答。",
      keyPoints: [
        "先檢索相關文件，再將內容注入提示詞",
        "降低幻覺（Hallucination）問題",
        "Embedding 模型將文本轉為向量進行相似度搜尋",
      ],
    },
    {
      title: "嵌入（Embedding）",
      description: "將文字、圖片等轉換為固定維度的向量，語義相近的內容向量也相近。",
      keyPoints: [
        "Word2Vec、BERT 等模型可以產生嵌入",
        "餘弦相似度衡量向量之間的語義相似性",
        "向量資料庫（如 FAISS）支援高效相似度搜尋",
      ],
    },
    {
      title: "Prompt Engineering",
      description: "設計有效的提示詞來引導 LLM 產生期望的輸出。",
      keyPoints: [
        "System Prompt 設定角色和行為規則",
        "Few-Shot 提供範例幫助模型理解任務",
        "Chain-of-Thought 讓模型逐步推理",
      ],
    },
    {
      title: "LLM 應用架構",
      description: "建構 LLM 應用的常見模式，從簡單 API 呼叫到複雜的 Agent 系統。",
      keyPoints: [
        "API 呼叫是最基礎的整合方式",
        "Function Calling 讓 LLM 可以呼叫外部工具",
        "Agent 系統讓 LLM 自主規劃和執行多步任務",
      ],
    },
    {
      title: "LLM 的限制與對策",
      description: "了解 LLM 的局限性，才能設計更可靠的應用。",
      keyPoints: [
        "幻覺問題：LLM 可能生成看似合理但錯誤的內容",
        "上下文長度限制：需要策略處理長文本",
        "評估困難：需要人類評估和自動化指標並用",
      ],
    },
  ],
  18: [
    {
      title: "專題規劃",
      description: "從問題定義、資料收集到模型部署的完整機器學習專案流程。",
      keyPoints: [
        "明確定義問題和成功指標",
        "資料品質決定專案的上限",
        "迭代式開發，先跑通再優化",
      ],
    },
    {
      title: "技術報告撰寫",
      description: "以結構化的方式記錄專案的動機、方法、結果和結論。",
      keyPoints: [
        "摘要讓讀者快速了解專案全貌",
        "方法論要夠詳細讓人能重現",
        "結果用圖表呈現比文字更有說服力",
      ],
    },
    {
      title: "簡報技巧",
      description: "用清晰的邏輯和視覺化有效地傳達技術內容。",
      keyPoints: [
        "每張投影片只傳達一個核心訊息",
        "Demo 比投影片更有說服力",
        "準備好回答「為什麼選這個方法」的問題",
      ],
    },
    {
      title: "學習歷程反思",
      description: "回顧 18 週的學習旅程，整理收穫並規劃未來發展方向。",
      keyPoints: [
        "從理論學習到實作之間的差距和體會",
        "哪些概念最有價值，哪些需要深入",
        "建立個人學習路徑和持續學習計畫",
      ],
    },
  ],
};

export default function ConceptCards({ week }: { week: number }) {
  const [expanded, setExpanded] = useState(false);
  const [openCard, setOpenCard] = useState<number | null>(null);
  const concepts = WEEK_CONCEPTS[week] || [];

  if (concepts.length === 0) return null;

  return (
    <div className="border border-gray-200 rounded-xl p-6" role="region" aria-label="Key concepts">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between text-left"
        aria-expanded={expanded}
      >
        <h2 className="text-lg font-semibold text-gray-900">
          本週核心概念
          <span className="ml-2 text-sm font-normal text-gray-400">({concepts.length} 個概念)</span>
        </h2>
        <span className={`text-gray-400 transition-transform duration-200 ${expanded ? "rotate-180" : ""}`}>
          ▼
        </span>
      </button>

      {expanded && (
        <div className="mt-4 space-y-2">
          {concepts.map((card, idx) => (
            <div key={idx} className="border border-gray-100 rounded-lg overflow-hidden">
              <button
                onClick={() => setOpenCard(openCard === idx ? null : idx)}
                className="w-full flex items-center justify-between px-4 py-3 text-left hover:bg-gray-50 transition-colors"
                aria-expanded={openCard === idx}
              >
                <div className="flex items-center gap-3">
                  <span className="flex items-center justify-center w-6 h-6 rounded-full bg-blue-100 text-blue-700 text-xs font-bold">
                    {idx + 1}
                  </span>
                  <span className="text-sm font-medium text-gray-800">{card.title}</span>
                </div>
                <span className={`text-gray-300 text-xs transition-transform duration-200 ${openCard === idx ? "rotate-180" : ""}`}>
                  ▼
                </span>
              </button>

              {openCard === idx && (
                <div className="px-4 pb-3 pt-0">
                  <p className="text-sm text-gray-600 mb-2 ml-9">{card.description}</p>
                  <ul className="ml-9 space-y-1">
                    {card.keyPoints.map((point, pi) => (
                      <li key={pi} className="text-xs text-gray-500 flex items-start gap-1.5">
                        <span className="text-blue-400 mt-0.5">•</span>
                        {point}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
