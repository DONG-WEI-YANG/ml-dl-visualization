#!/usr/bin/env python3
"""
ML/DL 視覺化教學系統 — Kaggle 資料集下載腳本
ML/DL Visualization Teaching System — Dataset Download Script

本腳本提供每週課程所需資料集的自動下載功能，
支援 Kaggle API 下載與 scikit-learn 內建資料集載入。

使用前準備 Prerequisites:
    1. 安裝相依套件 Install dependencies:
       pip install kaggle scikit-learn pandas numpy

    2. 設定 Kaggle API 認證 Setup Kaggle API credentials:
       - 前往 https://www.kaggle.com/settings → 建立 API Token
       - 將下載的 kaggle.json 放到:
         - Windows: C:\\Users\\<user>\\.kaggle\\kaggle.json
         - macOS/Linux: ~/.kaggle/kaggle.json
       - 設定檔案權限 (macOS/Linux): chmod 600 ~/.kaggle/kaggle.json

使用方式 Usage:
    # 下載所有週次的資料集
    python download.py --all

    # 下載特定週次的資料集
    python download.py --week 1
    python download.py --week 4 7 12

    # 列出所有資料集資訊（不下載）
    python download.py --list

    # 指定下載目錄
    python download.py --week 1 --output ./my_data

    # 下載並顯示資料集基本資訊
    python download.py --week 1 --info
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# 每週建議資料集清單 Weekly Recommended Datasets
# ---------------------------------------------------------------------------
# 格式說明 Format:
#   "week_N": {
#       "title": "中文主題 / English Topic",
#       "datasets": [
#           {
#               "name": "資料集名稱 Dataset Name",
#               "source": "kaggle" | "sklearn" | "pytorch" | "url",
#               "identifier": "Kaggle slug / sklearn 函式名 / 下載網址",
#               "description": "資料集說明",
#               "task": "分類 Classification / 迴歸 Regression / ...",
#               "size": "預估大小 Estimated Size",
#           }
#       ]
#   }

WEEKLY_DATASETS: dict = {
    "week_01": {
        "title": "課程導論、Python 與資料科學環境 / Introduction, Python & Data Science Environment",
        "datasets": [
            {
                "name": "Iris 鳶尾花資料集",
                "source": "sklearn",
                "identifier": "load_iris",
                "description": "經典的多類別分類資料集，含 150 筆樣本、4 個特徵、3 個類別。適合初學者練習資料載入與基本操作。",
                "task": "分類 Classification",
                "size": "~7 KB",
            },
            {
                "name": "Tips 小費資料集",
                "source": "kaggle",
                "identifier": "ranjeetjain3/seaborn-tips-dataset",
                "description": "餐廳小費資料，含帳單金額、小費、性別、吸菸與否等欄位。適合 Pandas 入門練習。",
                "task": "探索分析 EDA",
                "size": "~8 KB",
            },
        ],
    },
    "week_02": {
        "title": "資料視覺化與 EDA / Data Visualization & EDA",
        "datasets": [
            {
                "name": "Titanic 鐵達尼號資料集",
                "source": "kaggle",
                "identifier": "yasserh/titanic-dataset",
                "description": "經典的 Kaggle 入門資料集，包含乘客資訊與存活狀態。適合 EDA 與視覺化練習，含缺失值處理情境。",
                "task": "分類 Classification / EDA",
                "size": "~60 KB",
            },
            {
                "name": "Penguins 企鵝資料集",
                "source": "kaggle",
                "identifier": "parulpandey/palmer-archipelago-antarctica-penguin-data",
                "description": "Palmer 群島企鵝體態量測資料，含嘴峰長度、鰭長度、體重等。Iris 的現代替代品，適合視覺化。",
                "task": "分類 Classification / EDA",
                "size": "~12 KB",
            },
        ],
    },
    "week_03": {
        "title": "監督式學習、資料分割與交叉驗證 / Supervised Learning, Data Splitting & Cross-Validation",
        "datasets": [
            {
                "name": "Wine 葡萄酒資料集",
                "source": "sklearn",
                "identifier": "load_wine",
                "description": "義大利三種葡萄酒的化學分析結果，含 13 個特徵與 3 個類別。適合練習分割與交叉驗證。",
                "task": "分類 Classification",
                "size": "~11 KB",
            },
            {
                "name": "Breast Cancer 乳癌資料集",
                "source": "sklearn",
                "identifier": "load_breast_cancer",
                "description": "乳癌腫瘤特徵資料集，含 30 個數值特徵。適合練習二元分類與交叉驗證。",
                "task": "分類 Classification",
                "size": "~100 KB",
            },
        ],
    },
    "week_04": {
        "title": "線性回歸 — 損失函數、梯度下降視覺化 / Linear Regression — Loss Function & Gradient Descent",
        "datasets": [
            {
                "name": "Boston Housing（替代版）/ California Housing",
                "source": "sklearn",
                "identifier": "fetch_california_housing",
                "description": "加州房價資料集，含 8 個特徵（收入、房齡、房間數等）與房價中位數。取代已棄用的 Boston Housing。",
                "task": "迴歸 Regression",
                "size": "~450 KB",
            },
            {
                "name": "Student Performance 學生成績資料集",
                "source": "kaggle",
                "identifier": "spscientist/students-performance-in-exams",
                "description": "學生考試成績資料，含性別、種族、父母學歷、午餐類型等社經因素。適合迴歸分析與視覺化。",
                "task": "迴歸 Regression",
                "size": "~18 KB",
            },
        ],
    },
    "week_05": {
        "title": "分類 — 邏輯迴歸、決策邊界與 ROC/PR 曲線 / Classification — Logistic Regression & ROC/PR",
        "datasets": [
            {
                "name": "Heart Disease 心臟病資料集",
                "source": "kaggle",
                "identifier": "johnsmith88/heart-disease-dataset",
                "description": "心臟病診斷資料集，含年齡、性別、胸痛類型、血壓等 13 個特徵。適合二元分類與 ROC 分析。",
                "task": "分類 Classification",
                "size": "~12 KB",
            },
            {
                "name": "Make Moons / Make Circles (合成資料)",
                "source": "sklearn",
                "identifier": "make_moons",
                "description": "scikit-learn 合成的二維非線性可分資料，適合視覺化決策邊界。",
                "task": "分類 Classification",
                "size": "動態生成 Generated",
            },
        ],
    },
    "week_06": {
        "title": "SVM 與核方法視覺化 / SVM & Kernel Methods Visualization",
        "datasets": [
            {
                "name": "Make Blobs / Make Classification (合成資料)",
                "source": "sklearn",
                "identifier": "make_blobs",
                "description": "scikit-learn 合成的群集資料，可調整類別數與特徵數。適合 SVM 核方法比較視覺化。",
                "task": "分類 Classification",
                "size": "動態生成 Generated",
            },
            {
                "name": "Digits 手寫數字資料集",
                "source": "sklearn",
                "identifier": "load_digits",
                "description": "8x8 像素的手寫數字影像（0-9），共 1797 筆。輕量版的 MNIST，適合 SVM 多類別分類。",
                "task": "分類 Classification",
                "size": "~350 KB",
            },
        ],
    },
    "week_07": {
        "title": "樹模型與集成（RF、GBDT）/ Tree Models & Ensemble Methods",
        "datasets": [
            {
                "name": "Adult Income 成人收入資料集",
                "source": "kaggle",
                "identifier": "wenruliu/adult-income-dataset",
                "description": "美國人口普查資料，預測收入是否超過 50K 美元。含類別型與數值型特徵，適合樹模型。",
                "task": "分類 Classification",
                "size": "~4 MB",
            },
            {
                "name": "Diabetes 糖尿病資料集",
                "source": "sklearn",
                "identifier": "load_diabetes",
                "description": "糖尿病病程進展資料集，含 10 個基線特徵。適合迴歸樹與集成方法比較。",
                "task": "迴歸 Regression",
                "size": "~50 KB",
            },
        ],
    },
    "week_08": {
        "title": "特徵重要度與 SHAP 值 / Feature Importance & SHAP Values",
        "datasets": [
            {
                "name": "Ames Housing 房價資料集",
                "source": "kaggle",
                "identifier": "prevek18/ames-housing-dataset",
                "description": "Iowa 州 Ames 市房屋銷售資料，含 79 個解釋變數。特徵豐富，適合特徵重要度與 SHAP 分析。",
                "task": "迴歸 Regression",
                "size": "~460 KB",
            },
        ],
    },
    "week_09": {
        "title": "特徵工程與資料前處理管線 / Feature Engineering & Data Preprocessing Pipeline",
        "datasets": [
            {
                "name": "Spaceship Titanic",
                "source": "kaggle",
                "identifier": "competitions/spaceship-titanic",
                "description": "Kaggle 競賽資料集，包含多種特徵類型（數值、類別、布林）。含缺失值，適合建構完整前處理 Pipeline。",
                "task": "分類 Classification",
                "size": "~1 MB",
            },
            {
                "name": "Credit Card Fraud 信用卡詐欺資料集",
                "source": "kaggle",
                "identifier": "mlg-ulb/creditcardfraud",
                "description": "信用卡交易詐欺偵測資料集，極度不平衡（正例僅 0.17%）。適合前處理策略與重取樣練習。",
                "task": "分類 Classification（不平衡 Imbalanced）",
                "size": "~144 MB",
            },
        ],
    },
    "week_10": {
        "title": "超參數調校與學習曲線 / Hyperparameter Tuning & Learning Curves",
        "datasets": [
            {
                "name": "Wine Quality 葡萄酒品質資料集",
                "source": "kaggle",
                "identifier": "uciml/red-wine-quality-cortez-et-al-2009",
                "description": "紅酒品質評分資料，含 11 個理化特徵。適合超參數搜尋與學習曲線分析。",
                "task": "分類 / 迴歸",
                "size": "~84 KB",
            },
        ],
    },
    "week_11": {
        "title": "神經網路基礎 / Neural Network Basics",
        "datasets": [
            {
                "name": "MNIST 手寫數字資料集",
                "source": "pytorch",
                "identifier": "torchvision.datasets.MNIST",
                "description": "經典的手寫數字辨識資料集，28x28 灰階影像，60,000 筆訓練 + 10,000 筆測試。深度學習入門必備。",
                "task": "分類 Classification",
                "size": "~12 MB",
            },
            {
                "name": "Fashion-MNIST",
                "source": "pytorch",
                "identifier": "torchvision.datasets.FashionMNIST",
                "description": "時尚物件（衣服、鞋子、包包等）28x28 灰階影像。MNIST 的進階替代品。",
                "task": "分類 Classification",
                "size": "~30 MB",
            },
        ],
    },
    "week_12": {
        "title": "CNN 視覺化 / CNN Visualization",
        "datasets": [
            {
                "name": "CIFAR-10",
                "source": "pytorch",
                "identifier": "torchvision.datasets.CIFAR10",
                "description": "10 類彩色影像資料集（飛機、汽車、鳥、貓等），32x32 RGB 影像，50,000 筆訓練 + 10,000 筆測試。適合 CNN 卷積核與特徵圖視覺化。",
                "task": "分類 Classification",
                "size": "~170 MB",
            },
            {
                "name": "Cats vs Dogs（子集）",
                "source": "kaggle",
                "identifier": "biaiscience/dogs-vs-cats",
                "description": "貓狗影像分類資料集。適合 CNN 分類與 Grad-CAM 視覺化練習。建議使用子集以減少訓練時間。",
                "task": "分類 Classification",
                "size": "~800 MB（完整版）",
            },
        ],
    },
    "week_13": {
        "title": "RNN / 序列建模 / RNN & Sequence Modeling",
        "datasets": [
            {
                "name": "IMDB 電影評論情感分析",
                "source": "pytorch",
                "identifier": "torchtext.datasets.IMDB",
                "description": "IMDB 電影評論正負面情感分類，50,000 筆評論。適合 RNN/LSTM 文本分類。",
                "task": "分類 Classification（NLP）",
                "size": "~84 MB",
            },
            {
                "name": "Airline Passengers 航空旅客時間序列",
                "source": "url",
                "identifier": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv",
                "description": "1949-1960 年國際航空旅客月度數據。經典的時間序列預測資料集。",
                "task": "時間序列預測 Time Series Forecasting",
                "size": "~2 KB",
            },
        ],
    },
    "week_14": {
        "title": "深度學習訓練技巧 / Deep Learning Training Techniques",
        "datasets": [
            {
                "name": "CIFAR-100",
                "source": "pytorch",
                "identifier": "torchvision.datasets.CIFAR100",
                "description": "100 類彩色影像資料集，適合比較不同訓練策略（LR 排程、資料增強等）的效果。",
                "task": "分類 Classification",
                "size": "~170 MB",
            },
        ],
    },
    "week_15": {
        "title": "模型評估與偏誤檢測、公平性與穩健性 / Model Evaluation, Bias, Fairness & Robustness",
        "datasets": [
            {
                "name": "German Credit 德國信用風險資料集",
                "source": "kaggle",
                "identifier": "uciml/german-credit",
                "description": "信用風險評估資料集，含性別、年齡等敏感屬性 (Sensitive Attributes)。適合公平性分析。",
                "task": "分類 Classification",
                "size": "~56 KB",
            },
            {
                "name": "COMPAS Recidivism 再犯率預測資料集",
                "source": "kaggle",
                "identifier": "danofer/compass",
                "description": "美國刑事司法系統再犯率預測資料。經典的 AI 公平性 (Fairness) 案例研究資料集。",
                "task": "分類 Classification / 公平性分析 Fairness",
                "size": "~5 MB",
            },
        ],
    },
    "week_16": {
        "title": "MLOps 入門 / MLOps Introduction",
        "datasets": [
            {
                "name": "California Housing（同第 4 週）",
                "source": "sklearn",
                "identifier": "fetch_california_housing",
                "description": "加州房價資料集。在 MLOps 週複用此資料集，重點在模型部署與版本管理流程。",
                "task": "迴歸 Regression",
                "size": "~450 KB",
            },
        ],
    },
    "week_17": {
        "title": "LLM 與嵌入應用 / LLM & Embedding Applications",
        "datasets": [
            {
                "name": "20 Newsgroups 新聞群組資料集",
                "source": "sklearn",
                "identifier": "fetch_20newsgroups",
                "description": "20 個新聞群組的文本資料，約 20,000 篇文章。適合文字嵌入 (Text Embedding) 與 RAG 練習。",
                "task": "分類 / NLP",
                "size": "~14 MB",
            },
            {
                "name": "Wikipedia Simple English（子集）",
                "source": "url",
                "identifier": "https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2",
                "description": "簡易英文維基百科文本，適合建構 RAG 系統的知識庫。建議使用預處理後的子集。",
                "task": "NLP / RAG",
                "size": "~250 MB（壓縮版）",
            },
        ],
    },
    "week_18": {
        "title": "綜合專題開發與展示 / Final Project Presentation",
        "datasets": [
            {
                "name": "自選資料集 Student's Choice",
                "source": "kaggle",
                "identifier": "（由學生自行選擇）",
                "description": "期末專題由學生自行選擇適合的資料集。建議從 Kaggle Datasets 或公開資料平台挑選與專題主題相關的資料集。",
                "task": "依專題而定",
                "size": "依資料集而定",
            },
        ],
    },
}


def get_script_dir() -> Path:
    """取得本腳本所在目錄 Get the directory containing this script."""
    return Path(__file__).resolve().parent


def print_dataset_info_table(week_key: str) -> None:
    """印出指定週次的資料集資訊表 Print dataset info table for a given week."""
    week_data = WEEKLY_DATASETS[week_key]
    week_num = week_key.replace("week_", "")
    print(f"\n{'='*70}")
    print(f"  第 {week_num} 週 Week {week_num}: {week_data['title']}")
    print(f"{'='*70}")

    for ds in week_data["datasets"]:
        print(f"\n  -- {ds['name']}")
        print(f"     來源 Source    : {ds['source']}")
        print(f"     識別碼 ID     : {ds['identifier']}")
        print(f"     任務類型 Task : {ds['task']}")
        print(f"     大小 Size     : {ds['size']}")
        print(f"     說明          : {ds['description']}")


def list_all_datasets() -> None:
    """列出所有週次的資料集資訊 List all weekly datasets."""
    print("\n" + "#" * 70)
    print("#  ML/DL 視覺化教學系統 — 每週資料集清單")
    print("#  ML/DL Visualization — Weekly Dataset Catalog")
    print("#" * 70)

    for week_key in sorted(WEEKLY_DATASETS.keys()):
        print_dataset_info_table(week_key)

    print(f"\n{'='*70}")
    total = sum(len(w["datasets"]) for w in WEEKLY_DATASETS.values())
    print(f"  共 {len(WEEKLY_DATASETS)} 週，{total} 個資料集條目")
    print(f"{'='*70}\n")


def download_kaggle_dataset(identifier: str, output_dir: Path) -> bool:
    """
    使用 Kaggle API 下載資料集 Download dataset via Kaggle API.

    Args:
        identifier: Kaggle 資料集 slug（如 'yasserh/titanic-dataset'）
                     或競賽名稱（如 'competitions/spaceship-titanic'）
        output_dir: 下載目標目錄

    Returns:
        bool: 是否下載成功
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()

        output_dir.mkdir(parents=True, exist_ok=True)

        if identifier.startswith("competitions/"):
            # 競賽資料集 Competition dataset
            comp_name = identifier.replace("competitions/", "")
            print(f"    下載競賽資料集 Downloading competition: {comp_name} ...")
            api.competition_download_files(comp_name, path=str(output_dir))
        else:
            # 一般資料集 Regular dataset
            print(f"    下載資料集 Downloading: {identifier} ...")
            api.dataset_download_files(identifier, path=str(output_dir), unzip=True)

        # 解壓縮 zip 檔案（若競賽下載產生 zip）
        for zip_file in output_dir.glob("*.zip"):
            print(f"    解壓縮 Unzipping: {zip_file.name} ...")
            with zipfile.ZipFile(zip_file, "r") as zf:
                zf.extractall(output_dir)
            zip_file.unlink()  # 刪除 zip

        print(f"    完成 Done. 儲存於 Saved to: {output_dir}")
        return True

    except ImportError:
        print("    [錯誤 ERROR] 未安裝 kaggle 套件。請執行: pip install kaggle")
        return False
    except Exception as e:
        print(f"    [錯誤 ERROR] 下載失敗: {e}")
        return False


def load_sklearn_dataset(identifier: str, output_dir: Path) -> bool:
    """
    載入 scikit-learn 內建資料集並存為 CSV
    Load sklearn built-in dataset and save as CSV.

    Args:
        identifier: sklearn 資料集函式名稱（如 'load_iris'）
        output_dir: 儲存目標目錄

    Returns:
        bool: 是否載入成功
    """
    try:
        import numpy as np
        import pandas as pd
        from sklearn import datasets

        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"    載入 sklearn 資料集 Loading: {identifier} ...")

        loader = getattr(datasets, identifier, None)
        if loader is None:
            print(f"    [錯誤 ERROR] 找不到 sklearn.datasets.{identifier}")
            return False

        # 區分 fetch_* (需下載) 與 load_* (內建) 類型
        if identifier.startswith("fetch_20newsgroups"):
            data = loader(subset="all")
            df = pd.DataFrame({"text": data.data, "target": data.target})
            target_names = data.target_names
        elif identifier.startswith("fetch_"):
            data = loader()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df["target"] = data.target
            target_names = getattr(data, "target_names", None)
        elif identifier.startswith("make_"):
            # 合成資料集 Synthetic datasets
            if identifier == "make_moons":
                X, y = datasets.make_moons(n_samples=500, noise=0.2, random_state=42)
            elif identifier == "make_blobs":
                X, y = datasets.make_blobs(
                    n_samples=500, centers=3, cluster_std=1.0, random_state=42
                )
            elif identifier == "make_circles":
                X, y = datasets.make_circles(
                    n_samples=500, noise=0.1, factor=0.5, random_state=42
                )
            else:
                X, y = datasets.make_classification(
                    n_samples=500, n_features=2, random_state=42
                )

            feature_cols = [f"feature_{i}" for i in range(X.shape[1])]
            df = pd.DataFrame(X, columns=feature_cols)
            df["target"] = y
            target_names = None
        else:
            data = loader()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df["target"] = data.target
            target_names = getattr(data, "target_names", None)

        # 儲存為 CSV Save as CSV
        csv_path = output_dir / f"{identifier.replace('fetch_', '').replace('load_', '')}.csv"
        df.to_csv(csv_path, index=False)
        print(f"    已儲存 Saved: {csv_path}")

        # 顯示基本資訊 Print basic info
        print(f"    樣本數 Samples: {len(df)}")
        print(f"    特徵數 Features: {len(df.columns) - 1}")
        if target_names is not None:
            print(f"    類別 Classes: {list(target_names)}")

        return True

    except ImportError as e:
        print(f"    [錯誤 ERROR] 缺少套件: {e}")
        return False
    except Exception as e:
        print(f"    [錯誤 ERROR] 載入失敗: {e}")
        return False


def download_from_url(url: str, output_dir: Path, filename: Optional[str] = None) -> bool:
    """
    從 URL 下載資料集 Download dataset from URL.

    Args:
        url: 下載網址
        output_dir: 儲存目標目錄
        filename: 儲存檔名（若為 None 則從 URL 推斷）

    Returns:
        bool: 是否下載成功
    """
    try:
        import urllib.request

        output_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            filename = url.split("/")[-1]

        filepath = output_dir / filename
        print(f"    下載 Downloading: {url} ...")
        urllib.request.urlretrieve(url, filepath)
        print(f"    已儲存 Saved: {filepath}")
        return True

    except Exception as e:
        print(f"    [錯誤 ERROR] 下載失敗: {e}")
        return False


def print_csv_info(csv_path: Path) -> None:
    """印出 CSV 檔案的基本統計資訊 Print basic statistics of a CSV file."""
    try:
        import pandas as pd

        df = pd.read_csv(csv_path)
        print(f"\n    --- {csv_path.name} 基本資訊 Basic Info ---")
        print(f"    形狀 Shape: {df.shape[0]} 列 (rows) x {df.shape[1]} 欄 (columns)")
        print(f"    欄位 Columns: {list(df.columns)}")
        print(f"\n    資料型別 Data Types:")
        for col, dtype in df.dtypes.items():
            null_count = df[col].isnull().sum()
            null_pct = null_count / len(df) * 100
            null_info = f" (缺失 Missing: {null_count}, {null_pct:.1f}%)" if null_count > 0 else ""
            print(f"      {col}: {dtype}{null_info}")
        print(f"\n    前 5 筆 First 5 rows:")
        print(df.head().to_string(index=False))
    except ImportError:
        print("    [提示 INFO] 安裝 pandas 以查看詳細資訊: pip install pandas")
    except Exception as e:
        print(f"    [錯誤 ERROR] 無法讀取: {e}")


def download_week(week_num: int, output_base: Path, show_info: bool = False) -> None:
    """
    下載指定週次的所有資料集 Download all datasets for a given week.

    Args:
        week_num: 週次（1-18）
        output_base: 下載基準目錄
        show_info: 是否顯示資料集基本資訊
    """
    week_key = f"week_{week_num:02d}"

    if week_key not in WEEKLY_DATASETS:
        print(f"[錯誤 ERROR] 找不到第 {week_num} 週的資料集資訊。")
        return

    week_data = WEEKLY_DATASETS[week_key]
    week_dir = output_base / week_key

    print(f"\n{'='*60}")
    print(f"第 {week_num} 週 Week {week_num}: {week_data['title']}")
    print(f"下載目錄 Download to: {week_dir}")
    print(f"{'='*60}")

    for ds in week_data["datasets"]:
        print(f"\n  [{ds['name']}]")

        success = False
        if ds["source"] == "kaggle":
            if ds["identifier"].startswith("（"):
                print("    [跳過 SKIP] 此為學生自選資料集，無需下載。")
                continue
            success = download_kaggle_dataset(ds["identifier"], week_dir)
        elif ds["source"] == "sklearn":
            success = load_sklearn_dataset(ds["identifier"], week_dir)
        elif ds["source"] == "pytorch":
            print(f"    [提示 INFO] PyTorch 資料集請在程式碼中使用以下方式載入:")
            print(f"      import torchvision")
            print(f"      dataset = {ds['identifier']}(root='./data', download=True)")
            print(f"    （首次執行會自動下載）")
            continue
        elif ds["source"] == "url":
            success = download_from_url(ds["identifier"], week_dir)
        else:
            print(f"    [錯誤 ERROR] 不支援的來源類型: {ds['source']}")

        # 顯示資料集資訊 Show dataset info
        if show_info and success:
            for csv_file in week_dir.glob("*.csv"):
                print_csv_info(csv_file)


def main():
    """主程式 Main entry point."""
    parser = argparse.ArgumentParser(
        description="ML/DL 視覺化教學系統 — 資料集下載工具\n"
        "ML/DL Visualization Teaching System — Dataset Download Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例 Examples:
  python download.py --list              列出所有資料集資訊
  python download.py --all               下載所有週次的資料集
  python download.py --week 1            下載第 1 週的資料集
  python download.py --week 4 7 12       下載第 4、7、12 週的資料集
  python download.py --week 1 --info     下載並顯示資料集基本資訊
  python download.py --week 1 -o ./data  指定下載目錄
        """,
    )

    parser.add_argument(
        "--week",
        type=int,
        nargs="+",
        help="指定要下載的週次 Specify week number(s) to download (1-18)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="下載所有週次的資料集 Download all weekly datasets",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出所有資料集資訊（不下載）List all datasets without downloading",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="下載後顯示資料集基本統計資訊 Show dataset statistics after download",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="指定下載目錄 Specify output directory (default: script directory)",
    )

    args = parser.parse_args()

    # 若沒有給任何參數，顯示說明 If no arguments, show help
    if not args.week and not args.all and not args.list:
        parser.print_help()
        sys.exit(0)

    # 列出資料集 List datasets
    if args.list:
        list_all_datasets()
        sys.exit(0)

    # 決定輸出目錄 Determine output directory
    output_base = Path(args.output) if args.output else get_script_dir()

    # 決定要下載的週次 Determine weeks to download
    if args.all:
        weeks = list(range(1, 19))
    else:
        weeks = args.week

    # 驗證週次範圍 Validate week range
    for w in weeks:
        if w < 1 or w > 18:
            print(f"[錯誤 ERROR] 週次必須在 1-18 之間，收到: {w}")
            sys.exit(1)

    print("\n" + "#" * 60)
    print("#  ML/DL 視覺化教學系統 — 資料集下載工具")
    print("#  ML/DL Visualization — Dataset Download Tool")
    print(f"#  下載目錄 Output: {output_base}")
    print(f"#  週次 Weeks: {weeks}")
    print("#" * 60)

    # 逐週下載 Download week by week
    for week_num in weeks:
        download_week(week_num, output_base, show_info=args.info)

    print("\n" + "=" * 60)
    print("  所有下載任務完成 All download tasks completed.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
