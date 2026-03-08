export interface LLMMessage {
  role: "user" | "assistant";
  content: string;
}

export interface WeekInfo {
  id: number;
  title: string;
  topic: string;
  level: "core" | "advanced";
}

export const WEEKS: WeekInfo[] = [
  { id: 1, title: "課程導論、Python 環境", topic: "課程導論、Python 與資料科學環境", level: "core" },
  { id: 2, title: "資料視覺化與 EDA", topic: "資料視覺化與 EDA（互動圖表）", level: "core" },
  { id: 3, title: "監督式學習概念", topic: "監督式學習概念、資料分割與交叉驗證", level: "core" },
  { id: 4, title: "線性回歸與梯度下降", topic: "線性回歸：損失函數、梯度下降視覺化", level: "core" },
  { id: 5, title: "分類與決策邊界", topic: "分類：邏輯迴歸、決策邊界與 ROC/PR 曲線", level: "core" },
  { id: 6, title: "SVM 與核方法", topic: "SVM 與核方法視覺化", level: "core" },
  { id: 7, title: "樹模型與集成", topic: "樹模型與集成（RF、GBDT）", level: "core" },
  { id: 8, title: "特徵重要度與 SHAP", topic: "特徵重要度與 Shapley 示意", level: "core" },
  { id: 9, title: "特徵工程與前處理", topic: "特徵工程與資料前處理管線", level: "core" },
  { id: 10, title: "超參數調校", topic: "超參數調校與學習曲線", level: "core" },
  { id: 11, title: "神經網路基礎", topic: "神經網路基礎（激活/正則化/BatchNorm）", level: "core" },
  { id: 12, title: "CNN 視覺化", topic: "CNN 視覺化（卷積核、特徵圖、CAM/Grad-CAM）", level: "advanced" },
  { id: 13, title: "RNN/Transformers", topic: "RNN/序列建模（LSTM/GRU；Transformers 概念）", level: "advanced" },
  { id: 14, title: "DL 訓練技巧", topic: "深度學習訓練技巧（學習率策略、早停、資料增強）", level: "core" },
  { id: 15, title: "模型評估與公平性", topic: "模型評估與偏誤檢測、公平性與穩健性", level: "advanced" },
  { id: 16, title: "MLOps 入門", topic: "MLOps 入門（模型版本、推論服務、監測）", level: "advanced" },
  { id: 17, title: "LLM 與嵌入應用", topic: "LLM 與嵌入應用（檢索增強、提示工程基礎）", level: "advanced" },
  { id: 18, title: "綜合專題展示", topic: "綜合專題開發與展示、反思與課程回饋", level: "core" },
];
