import { useState, useMemo, useCallback } from "react";

/* ──────────────────────────────────────────────
   BPE (Byte Pair Encoding) 教學用模擬器
   ────────────────────────────────────────────── */

/** 將單詞拆成字元 + 結尾符 </w>，回傳 pair 頻率表 */
function buildCorpus(words: string[]): Map<string, string[]> {
  const corpus = new Map<string, string[]>();
  for (const w of words) {
    const chars = [...w, "</w>"];
    const key = chars.join(" ");
    corpus.set(key, chars);
  }
  return corpus;
}

/** 統計相鄰 pair 出現頻率 */
function countPairs(corpus: Map<string, string[]>): Map<string, number> {
  const pairs = new Map<string, number>();
  for (const tokens of corpus.values()) {
    for (let i = 0; i < tokens.length - 1; i++) {
      const key = `${tokens[i]}|${tokens[i + 1]}`;
      pairs.set(key, (pairs.get(key) ?? 0) + 1);
    }
  }
  return pairs;
}

/** 執行一次合併 */
function mergePair(
  corpus: Map<string, string[]>,
  bestPair: [string, string],
): Map<string, string[]> {
  const merged = bestPair[0] + bestPair[1];
  const next = new Map<string, string[]>();
  for (const [key, tokens] of corpus) {
    const out: string[] = [];
    let i = 0;
    while (i < tokens.length) {
      if (
        i < tokens.length - 1 &&
        tokens[i] === bestPair[0] &&
        tokens[i + 1] === bestPair[1]
      ) {
        out.push(merged);
        i += 2;
      } else {
        out.push(tokens[i]);
        i++;
      }
    }
    next.set(key, out);
  }
  return next;
}

interface BPEStep {
  corpus: Map<string, string[]>;
  mergedPair: [string, string] | null;
  mergedResult: string | null;
  pairFreqs: Map<string, number>;
}

/** 產生完整 BPE 步驟歷程 */
function runBPE(words: string[], maxSteps: number): BPEStep[] {
  let corpus = buildCorpus(words);
  const steps: BPEStep[] = [];
  const pairFreqs = countPairs(corpus);
  steps.push({ corpus, mergedPair: null, mergedResult: null, pairFreqs });

  for (let s = 0; s < maxSteps; s++) {
    const pairs = countPairs(corpus);
    if (pairs.size === 0) break;
    let best = "";
    let bestCount = 0;
    for (const [k, v] of pairs) {
      if (v > bestCount) {
        best = k;
        bestCount = v;
      }
    }
    if (bestCount < 1) break;
    const [a, b] = best.split("|");
    corpus = mergePair(corpus, [a, b]);
    steps.push({
      corpus,
      mergedPair: [a, b],
      mergedResult: a + b,
      pairFreqs: countPairs(corpus),
    });
  }
  return steps;
}

/* ──────────────────────────────────────────────
   簡易分詞策略（教學展示用）
   ────────────────────────────────────────────── */

function tokenizeByChar(text: string): string[] {
  return [...text].filter((c) => c.trim());
}

function tokenizeByWord(text: string): string[] {
  // 中文：每個字元獨立；英文：按空格
  const result: string[] = [];
  let buf = "";
  for (const ch of text) {
    if (/[\u4e00-\u9fff\u3400-\u4dbf]/.test(ch)) {
      if (buf.trim()) result.push(buf.trim());
      buf = "";
      result.push(ch);
    } else if (/\s/.test(ch)) {
      if (buf.trim()) result.push(buf.trim());
      buf = "";
    } else {
      buf += ch;
    }
  }
  if (buf.trim()) result.push(buf.trim());
  return result;
}

/** 模擬 subword：中文按常見雙字詞合併，英文按常見前後綴拆分 */
const COMMON_ZH_PAIRS = new Set([
  "機器", "學習", "深度", "神經", "網路", "人工", "智慧", "自然",
  "語言", "處理", "模型", "訓練", "資料", "演算", "分類", "預測",
  "嵌入", "向量", "注意", "力量", "卷積", "特徵", "梯度", "下降",
  "損失", "函數", "激活", "迴歸", "聚類", "決策", "隨機", "森林",
  "醫療", "護理", "臨床", "診斷", "病患", "藥物", "檢驗", "影像",
  "有趣", "很好", "非常", "重要", "知道", "學生", "老師", "大家",
]);

function tokenizeSubword(text: string): string[] {
  const chars = tokenizeByWord(text);
  const result: string[] = [];
  let i = 0;
  while (i < chars.length) {
    if (
      i < chars.length - 1 &&
      chars[i].length === 1 &&
      chars[i + 1].length === 1 &&
      /[\u4e00-\u9fff]/.test(chars[i]) &&
      COMMON_ZH_PAIRS.has(chars[i] + chars[i + 1])
    ) {
      result.push(chars[i] + chars[i + 1]);
      i += 2;
    } else if (/^[a-zA-Z]+$/.test(chars[i]) && chars[i].length > 4) {
      // 英文長詞拆 subword
      const w = chars[i];
      const suffixes = ["ing", "tion", "ment", "ness", "able", "ful", "less", "ous", "ize", "ised", "ized", "er", "ed", "ly", "al"];
      const prefixes = ["un", "re", "pre", "dis", "over", "mis"];
      let split = false;
      for (const pre of prefixes) {
        if (w.toLowerCase().startsWith(pre) && w.length > pre.length + 2) {
          result.push(w.slice(0, pre.length));
          result.push(w.slice(pre.length));
          split = true;
          break;
        }
      }
      if (!split) {
        for (const suf of suffixes) {
          if (w.toLowerCase().endsWith(suf) && w.length > suf.length + 2) {
            result.push(w.slice(0, w.length - suf.length));
            result.push(w.slice(w.length - suf.length));
            split = true;
            break;
          }
        }
      }
      if (!split) result.push(w);
      i++;
    } else {
      result.push(chars[i]);
      i++;
    }
  }
  return result;
}

/* ──────────────────────────────────────────────
   Token 彩色方塊
   ────────────────────────────────────────────── */

const COLORS = [
  "bg-blue-100 text-blue-800 border-blue-300",
  "bg-green-100 text-green-800 border-green-300",
  "bg-yellow-100 text-yellow-800 border-yellow-300",
  "bg-purple-100 text-purple-800 border-purple-300",
  "bg-pink-100 text-pink-800 border-pink-300",
  "bg-indigo-100 text-indigo-800 border-indigo-300",
  "bg-orange-100 text-orange-800 border-orange-300",
  "bg-teal-100 text-teal-800 border-teal-300",
];

function TokenBadge({ token, index }: { token: string; index: number }) {
  const color = COLORS[index % COLORS.length];
  return (
    <span
      className={`inline-flex items-center px-2 py-1 rounded border text-xs font-mono ${color}`}
    >
      {token}
    </span>
  );
}

function TokenRow({
  label,
  tokens,
  countLabel,
}: {
  label: string;
  tokens: string[];
  countLabel?: string;
}) {
  return (
    <div className="space-y-1">
      <div className="flex items-center gap-2">
        <span className="text-xs font-medium text-gray-500 w-24 shrink-0">
          {label}
        </span>
        <span className="text-xs text-gray-400">
          {countLabel ?? `${tokens.length} tokens`}
        </span>
      </div>
      <div className="flex flex-wrap gap-1 ml-0">
        {tokens.map((t, i) => (
          <TokenBadge key={`${i}-${t}`} token={t} index={i} />
        ))}
      </div>
    </div>
  );
}

/* ──────────────────────────────────────────────
   Tab 1: 即時分詞比較
   ────────────────────────────────────────────── */

const PRESETS = [
  { label: "中文", text: "機器學習很有趣" },
  { label: "英文", text: "Machine learning is fun" },
  { label: "中英混合", text: "深度學習用的是Transformer模型" },
  { label: "醫護", text: "臨床診斷需要結合影像檢驗資料" },
];

function LiveTokenizer() {
  const [text, setText] = useState(PRESETS[0].text);

  const charTokens = useMemo(() => tokenizeByChar(text), [text]);
  const wordTokens = useMemo(() => tokenizeByWord(text), [text]);
  const subwordTokens = useMemo(() => tokenizeSubword(text), [text]);

  return (
    <div className="space-y-4">
      <div>
        <label className="text-sm font-medium text-gray-700 block mb-1">
          輸入文字（可自行修改）
        </label>
        <input
          type="text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-300 focus:border-blue-400 outline-none"
          placeholder="輸入任意文字..."
        />
      </div>

      <div className="flex gap-1.5 flex-wrap">
        {PRESETS.map((p) => (
          <button
            key={p.label}
            onClick={() => setText(p.text)}
            className={`px-2.5 py-1 rounded text-xs transition-colors ${
              text === p.text
                ? "bg-blue-500 text-white"
                : "bg-gray-100 text-gray-600 hover:bg-gray-200"
            }`}
          >
            {p.label}
          </button>
        ))}
      </div>

      {text.trim() && (
        <div className="space-y-3 bg-gray-50 rounded-lg p-4">
          <TokenRow label="字元級 Character" tokens={charTokens} />
          <div className="border-t border-gray-200" />
          <TokenRow label="詞級 Word" tokens={wordTokens} />
          <div className="border-t border-gray-200" />
          <TokenRow label="子詞級 Subword" tokens={subwordTokens} />
        </div>
      )}

      <div className="bg-blue-50 rounded-lg p-3 text-xs text-blue-800 space-y-1">
        <p>
          <strong>為什麼用 Subword？</strong>
        </p>
        <p>
          字元級粒度太細（序列過長）、詞級遇到生詞就不認識（OOV）。
          Subword 取兩者平衡：常見詞保持完整，罕見詞拆成有意義的子片段。
        </p>
      </div>
    </div>
  );
}

/* ──────────────────────────────────────────────
   Tab 2: BPE 動畫
   ────────────────────────────────────────────── */

const BPE_PRESETS = [
  { label: "基礎", words: ["low", "low", "lower", "lowest", "new", "newer"] },
  {
    label: "學習",
    words: ["learn", "learn", "learning", "learned", "learner", "unlearn"],
  },
];

function BPEDemo() {
  const [presetIdx, setPresetIdx] = useState(0);
  const [step, setStep] = useState(0);

  const steps = useMemo(
    () => runBPE(BPE_PRESETS[presetIdx].words, 12),
    [presetIdx],
  );
  const current = steps[Math.min(step, steps.length - 1)];
  const maxStep = steps.length - 1;

  const handlePresetChange = useCallback(
    (idx: number) => {
      setPresetIdx(idx);
      setStep(0);
    },
    [],
  );

  // 取得所有 unique token 序列（去重展示）
  const uniqueTokenLists = useMemo(() => {
    const seen = new Set<string>();
    const result: string[][] = [];
    for (const tokens of current.corpus.values()) {
      const key = tokens.join("|");
      if (!seen.has(key)) {
        seen.add(key);
        result.push(tokens);
      }
    }
    return result;
  }, [current]);

  // 前三名 pair 頻率
  const topPairs = useMemo(() => {
    const entries = [...current.pairFreqs.entries()];
    entries.sort((a, b) => b[1] - a[1]);
    return entries.slice(0, 5);
  }, [current]);

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-xs text-gray-500">語料：</span>
        {BPE_PRESETS.map((p, i) => (
          <button
            key={p.label}
            onClick={() => handlePresetChange(i)}
            className={`px-2.5 py-1 rounded text-xs transition-colors ${
              i === presetIdx
                ? "bg-blue-500 text-white"
                : "bg-gray-100 text-gray-600 hover:bg-gray-200"
            }`}
          >
            {p.label}
          </button>
        ))}
        <span className="text-xs text-gray-400 ml-2">
          [{BPE_PRESETS[presetIdx].words.join(", ")}]
        </span>
      </div>

      {/* 步驟控制 */}
      <div className="flex items-center gap-3">
        <button
          onClick={() => setStep((s) => Math.max(0, s - 1))}
          disabled={step === 0}
          className="px-2.5 py-1 rounded text-xs bg-gray-200 hover:bg-gray-300 disabled:opacity-40 disabled:cursor-not-allowed"
        >
          ← 上一步
        </button>
        <div className="flex-1">
          <input
            type="range"
            min={0}
            max={maxStep}
            value={step}
            onChange={(e) => setStep(+e.target.value)}
            className="w-full"
          />
        </div>
        <button
          onClick={() => setStep((s) => Math.min(maxStep, s + 1))}
          disabled={step >= maxStep}
          className="px-2.5 py-1 rounded text-xs bg-gray-200 hover:bg-gray-300 disabled:opacity-40 disabled:cursor-not-allowed"
        >
          下一步 →
        </button>
      </div>

      <div className="text-sm font-medium text-gray-700">
        步驟 {step} / {maxStep}
        {current.mergedPair && (
          <span className="ml-2 text-blue-600">
            合併：「{current.mergedPair[0]}」+「{current.mergedPair[1]}」→「
            {current.mergedResult}」
          </span>
        )}
        {step === 0 && (
          <span className="ml-2 text-gray-400">（初始：字元級拆分）</span>
        )}
      </div>

      {/* 目前 token 狀態 */}
      <div className="bg-gray-50 rounded-lg p-4 space-y-2">
        <p className="text-xs font-medium text-gray-500 mb-2">
          各詞的 Token 狀態：
        </p>
        {uniqueTokenLists.map((tokens, i) => (
          <div key={i} className="flex flex-wrap gap-1">
            {tokens.map((t, j) => (
              <span
                key={`${j}-${t}`}
                className={`inline-flex items-center px-2 py-1 rounded border text-xs font-mono ${
                  current.mergedResult && t === current.mergedResult
                    ? "bg-blue-200 text-blue-900 border-blue-400 ring-1 ring-blue-300"
                    : COLORS[j % COLORS.length]
                }`}
              >
                {t}
              </span>
            ))}
          </div>
        ))}
      </div>

      {/* Pair 頻率表 */}
      {topPairs.length > 0 && (
        <div>
          <p className="text-xs font-medium text-gray-500 mb-1">
            {step < maxStep ? "下一步將合併最高頻的 pair：" : "剩餘 pair 頻率："}
          </p>
          <div className="flex gap-2 flex-wrap">
            {topPairs.map(([pair, count], i) => {
              const [a, b] = pair.split("|");
              return (
                <span
                  key={pair}
                  className={`text-xs px-2 py-1 rounded border ${
                    i === 0 && step < maxStep
                      ? "bg-yellow-100 border-yellow-400 font-bold"
                      : "bg-white border-gray-200"
                  }`}
                >
                  {a}+{b}
                  <span className="text-gray-400 ml-1">({count})</span>
                </span>
              );
            })}
          </div>
        </div>
      )}

      <div className="bg-green-50 rounded-lg p-3 text-xs text-green-800 space-y-1">
        <p>
          <strong>BPE 核心概念：</strong>
        </p>
        <p>
          從字元開始，反覆找出最常「相鄰出現」的 pair 並合併。
          訓練完成後，常見單詞會被保留為完整 token，罕見單詞則拆成已知的子片段。
        </p>
      </div>
    </div>
  );
}

/* ──────────────────────────────────────────────
   Tab 3: 中英文 Token 成本比較
   ────────────────────────────────────────────── */

const COST_EXAMPLES = [
  {
    zh: "人工智慧改變了醫療產業",
    en: "AI has transformed healthcare",
  },
  {
    zh: "深度學習模型可以輔助臨床診斷",
    en: "Deep learning models can assist clinical diagnosis",
  },
  {
    zh: "今天天氣很好",
    en: "The weather is nice today",
  },
];

function TokenCostCompare() {
  const [exIdx, setExIdx] = useState(0);
  const ex = COST_EXAMPLES[exIdx];

  const zhTokens = useMemo(() => tokenizeSubword(ex.zh), [ex.zh]);
  const enTokens = useMemo(() => tokenizeSubword(ex.en), [ex.en]);

  // 模擬 token 成本（中文每 token ~1.5 字，英文每 token ~4 字元）
  const zhCharPerToken = ex.zh.replace(/\s/g, "").length / zhTokens.length;
  const enCharPerToken = ex.en.replace(/\s/g, "").length / enTokens.length;

  return (
    <div className="space-y-4">
      <div className="flex gap-1.5 flex-wrap">
        {COST_EXAMPLES.map((_, i) => (
          <button
            key={i}
            onClick={() => setExIdx(i)}
            className={`px-2.5 py-1 rounded text-xs transition-colors ${
              i === exIdx
                ? "bg-blue-500 text-white"
                : "bg-gray-100 text-gray-600 hover:bg-gray-200"
            }`}
          >
            範例 {i + 1}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* 中文 */}
        <div className="bg-red-50 rounded-lg p-4 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm font-semibold text-red-800">中文</span>
            <span className="text-xs bg-red-200 text-red-800 px-2 py-0.5 rounded-full">
              {zhTokens.length} tokens
            </span>
          </div>
          <p className="text-sm text-gray-700">{ex.zh}</p>
          <div className="flex flex-wrap gap-1">
            {zhTokens.map((t, i) => (
              <TokenBadge key={`zh-${i}`} token={t} index={i} />
            ))}
          </div>
          <p className="text-xs text-gray-500">
            平均 {zhCharPerToken.toFixed(1)} 字/token
          </p>
        </div>

        {/* 英文 */}
        <div className="bg-blue-50 rounded-lg p-4 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm font-semibold text-blue-800">English</span>
            <span className="text-xs bg-blue-200 text-blue-800 px-2 py-0.5 rounded-full">
              {enTokens.length} tokens
            </span>
          </div>
          <p className="text-sm text-gray-700">{ex.en}</p>
          <div className="flex flex-wrap gap-1">
            {enTokens.map((t, i) => (
              <TokenBadge key={`en-${i}`} token={t} index={i} />
            ))}
          </div>
          <p className="text-xs text-gray-500">
            平均 {enCharPerToken.toFixed(1)} 字元/token
          </p>
        </div>
      </div>

      {/* 長條圖比較 */}
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <p className="text-xs font-medium text-gray-500 mb-3">
          Token 數量比較
        </p>
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <span className="text-xs w-10 text-right text-gray-500">中文</span>
            <div className="flex-1 bg-gray-100 rounded-full h-5 overflow-hidden">
              <div
                className="bg-red-400 h-full rounded-full flex items-center justify-end pr-2 transition-all duration-500"
                style={{
                  width: `${(zhTokens.length / Math.max(zhTokens.length, enTokens.length)) * 100}%`,
                  minWidth: "2rem",
                }}
              >
                <span className="text-[10px] font-bold text-white">
                  {zhTokens.length}
                </span>
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs w-10 text-right text-gray-500">英文</span>
            <div className="flex-1 bg-gray-100 rounded-full h-5 overflow-hidden">
              <div
                className="bg-blue-400 h-full rounded-full flex items-center justify-end pr-2 transition-all duration-500"
                style={{
                  width: `${(enTokens.length / Math.max(zhTokens.length, enTokens.length)) * 100}%`,
                  minWidth: "2rem",
                }}
              >
                <span className="text-[10px] font-bold text-white">
                  {enTokens.length}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-amber-50 rounded-lg p-3 text-xs text-amber-800 space-y-1">
        <p>
          <strong>為什麼中文通常需要更多 Token？</strong>
        </p>
        <p>
          主流 LLM 的 tokenizer 以英文語料為主訓練，英文常見詞往往是單一
          token，而中文字在 tokenizer 的訓練資料中出現頻率較低，常被拆得更碎。
          這代表相同語意的中文 prompt 會消耗更多 token，也意味著更高的 API 費用。
        </p>
      </div>
    </div>
  );
}

/* ──────────────────────────────────────────────
   主元件：三分頁
   ────────────────────────────────────────────── */

type Tab = "live" | "bpe" | "cost";

const TABS: { key: Tab; label: string; desc: string }[] = [
  { key: "live", label: "即時分詞", desc: "比較三種分詞策略" },
  { key: "bpe", label: "BPE 演算法", desc: "逐步合併動畫" },
  { key: "cost", label: "中英文比較", desc: "Token 成本差異" },
];

export default function TokenizationViz() {
  const [tab, setTab] = useState<Tab>("live");

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold">
        Token 與 Tokenization 分詞互動實驗
      </h3>

      <div className="flex gap-1 bg-gray-100 rounded-lg p-1">
        {TABS.map((t) => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            className={`flex-1 px-3 py-2 rounded-md text-xs font-medium transition-colors ${
              tab === t.key
                ? "bg-white text-blue-700 shadow-sm"
                : "text-gray-500 hover:text-gray-700"
            }`}
          >
            <div>{t.label}</div>
            <div className="text-[10px] font-normal text-gray-400">
              {t.desc}
            </div>
          </button>
        ))}
      </div>

      {tab === "live" && <LiveTokenizer />}
      {tab === "bpe" && <BPEDemo />}
      {tab === "cost" && <TokenCostCompare />}
    </div>
  );
}
