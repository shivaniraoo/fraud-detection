import pandas as pd
import pytest

from analyze_fraud import summarize_results


@pytest.fixture
def scored():
    return pd.DataFrame([
        {"transaction_id": 1, "risk_label": "high", "amount_usd": 500.0},
        {"transaction_id": 2, "risk_label": "high", "amount_usd": 300.0},
        {"transaction_id": 3, "risk_label": "low",  "amount_usd": 50.0},
        {"transaction_id": 4, "risk_label": "low",  "amount_usd": 30.0},
    ])


@pytest.fixture
def chargebacks():
    # Only transaction 1 resulted in a chargeback
    return pd.DataFrame([{"transaction_id": 1, "loss_amount_usd": 500.0}])


def label_row(summary: pd.DataFrame, label: str) -> pd.Series:
    return summary.loc[summary["risk_label"] == label].iloc[0]


class TestSummarizeResults:
    def test_transaction_counts_per_label(self, scored, chargebacks):
        summary = summarize_results(scored, chargebacks)
        assert label_row(summary, "high")["transactions"] == 2
        assert label_row(summary, "low")["transactions"] == 2

    def test_total_amount_per_label(self, scored, chargebacks):
        summary = summarize_results(scored, chargebacks)
        assert label_row(summary, "high")["total_amount_usd"] == 800.0
        assert label_row(summary, "low")["total_amount_usd"] == 80.0

    def test_avg_amount_per_label(self, scored, chargebacks):
        summary = summarize_results(scored, chargebacks)
        assert label_row(summary, "high")["avg_amount_usd"] == 400.0
        assert label_row(summary, "low")["avg_amount_usd"] == 40.0

    def test_chargeback_rate_with_confirmed_fraud(self, scored, chargebacks):
        summary = summarize_results(scored, chargebacks)
        # 1 chargeback out of 2 high-risk transactions = 0.5
        assert label_row(summary, "high")["chargeback_rate"] == 0.5

    def test_chargeback_rate_zero_when_no_fraud(self, scored, chargebacks):
        summary = summarize_results(scored, chargebacks)
        assert label_row(summary, "low")["chargeback_rate"] == 0.0

    def test_chargeback_count_per_label(self, scored, chargebacks):
        summary = summarize_results(scored, chargebacks)
        assert label_row(summary, "high")["chargebacks"] == 1
        assert label_row(summary, "low")["chargebacks"] == 0

    def test_all_risk_labels_present(self, scored, chargebacks):
        summary = summarize_results(scored, chargebacks)
        assert set(summary["risk_label"]) == {"high", "low"}

    def test_transaction_not_in_chargebacks_not_counted(self, scored):
        # No chargebacks at all — every chargeback_rate should be 0
        empty_chargebacks = pd.DataFrame(columns=["transaction_id"])
        summary = summarize_results(scored, empty_chargebacks)
        assert (summary["chargeback_rate"] == 0.0).all()
