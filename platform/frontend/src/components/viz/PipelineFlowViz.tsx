import { useState } from "react";

const STEPS = [
  { id: "raw", label: "原始資料", icon: "📦", desc: "CSV/JSON 讀取", color: "#e0e7ff" },
  { id: "clean", label: "資料清理", icon: "🧹", desc: "缺失值/離群值處理", color: "#dbeafe" },
  { id: "encode", label: "特徵編碼", icon: "🔢", desc: "One-Hot / Label Encoding", color: "#bfdbfe" },
  { id: "scale", label: "特徵縮放", icon: "📏", desc: "StandardScaler / MinMaxScaler", color: "#93c5fd" },
  { id: "select", label: "特徵選擇", icon: "🎯", desc: "相關性 / 重要度篩選", color: "#60a5fa" },
  { id: "split", label: "資料分割", icon: "✂️", desc: "Train / Val / Test", color: "#3b82f6" },
  { id: "model", label: "模型訓練", icon: "🤖", desc: "fit(X_train, y_train)", color: "#2563eb" },
  { id: "eval", label: "模型評估", icon: "📊", desc: "accuracy / F1 / AUC", color: "#1d4ed8" },
];

const DETAIL: Record<string, string[]> = {
  raw: ["df = pd.read_csv('data.csv')", "df.shape, df.dtypes, df.head()"],
  clean: ["df.isnull().sum()", "df.fillna(df.median())", "df = df[df['age'] < 100]"],
  encode: ["pd.get_dummies(df, columns=['color'])", "LabelEncoder().fit_transform(df['size'])"],
  scale: ["scaler = StandardScaler()", "X_scaled = scaler.fit_transform(X)"],
  select: ["corr = df.corr()", "SelectKBest(k=5).fit_transform(X, y)"],
  split: ["X_train, X_test, y_train, y_test =", "  train_test_split(X, y, test_size=0.2)"],
  model: ["model = RandomForestClassifier()", "model.fit(X_train, y_train)"],
  eval: ["y_pred = model.predict(X_test)", "accuracy_score(y_test, y_pred)"],
};

export default function PipelineFlowViz() {
  const [active, setActive] = useState<string | null>(null);

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold">資料前處理管線 Pipeline</h3>

      <div className="space-y-1">
        {STEPS.map((step, i) => (
          <div key={step.id}>
            <button
              onClick={() => setActive(active === step.id ? null : step.id)}
              className={`w-full flex items-center gap-3 p-3 rounded-lg border transition-all text-left ${
                active === step.id ? "border-blue-400 shadow-sm" : "border-gray-200 hover:border-gray-300"
              }`}
              style={{ backgroundColor: active === step.id ? step.color + "40" : undefined }}
            >
              <span className="text-xl">{step.icon}</span>
              <div className="flex-1">
                <p className="text-sm font-medium">{step.label}</p>
                <p className="text-xs text-gray-500">{step.desc}</p>
              </div>
              <span className="text-xs text-gray-400">Step {i + 1}</span>
            </button>

            {active === step.id && (
              <div className="ml-8 mt-1 mb-2 bg-gray-900 text-green-400 rounded-lg p-3 text-xs font-mono">
                {DETAIL[step.id]?.map((line, j) => <div key={j}>{line}</div>)}
              </div>
            )}

            {i < STEPS.length - 1 && (
              <div className="flex justify-center">
                <div className="w-0.5 h-3 bg-gray-300" />
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="bg-gray-50 rounded-lg p-3 text-xs text-gray-600">
        <p><strong>Scikit-learn Pipeline：</strong></p>
        <code className="block bg-gray-900 text-green-400 p-2 rounded mt-1">
          Pipeline([('scaler', StandardScaler()), ('clf', SVC())])
        </code>
      </div>
    </div>
  );
}
