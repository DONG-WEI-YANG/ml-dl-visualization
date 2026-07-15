from app.llm import quick_answer


def test_build_quick_answer_uses_two_current_week_chunks(monkeypatch):
    calls = []

    def fake_retrieve(query, week, top_k=5):
        calls.append((query, week, top_k))
        return "梯度下降會沿著損失函數的負梯度更新參數。學習率控制每次更新的步幅。"

    monkeypatch.setattr(quick_answer, "retrieve_context", fake_retrieve)
    answer = quick_answer.build_quick_answer("梯度下降是什麼？", 4, "線性迴歸與梯度下降")

    assert calls == [("梯度下降是什麼？", 4, 2)]
    assert answer is not None
    assert "梯度下降" in answer
    assert 2 <= sum(answer.count(mark) for mark in "。！？") <= 4
    assert len(answer) <= 500


def test_build_quick_answer_limits_context(monkeypatch):
    monkeypatch.setattr(quick_answer, "retrieve_context", lambda *args, **kwargs: "重點內容。" * 1000)
    answer = quick_answer.build_quick_answer("請說明本週重點", 3, "監督式學習")
    assert answer is not None
    assert len(answer) <= 500


def test_high_risk_question_returns_scope_notice(monkeypatch):
    monkeypatch.setattr(quick_answer, "retrieve_context", lambda *args, **kwargs: "")
    answer = quick_answer.build_quick_answer("我胸痛該吃什麼藥？", 1, "Python 環境")
    assert answer is not None
    assert "不能提供醫療判斷" in answer
    assert "專業人員" in answer
