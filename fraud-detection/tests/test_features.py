import pandas as pd
import pytest

from features import build_model_frame


@pytest.fixture
def sample_data():
    transactions = pd.DataFrame([
        {"transaction_id": 1, "account_id": 101, "amount_usd": 1200.0, "failed_logins_24h": 0},
        {"transaction_id": 2, "account_id": 102, "amount_usd": 800.0,  "failed_logins_24h": 2},
        {"transaction_id": 3, "account_id": 103, "amount_usd": 50.0,   "failed_logins_24h": 5},
    ])
    accounts = pd.DataFrame([
        {"account_id": 101, "customer_name": "Alice"},
        {"account_id": 102, "customer_name": "Bob"},
        {"account_id": 103, "customer_name": "Carol"},
    ])
    return transactions, accounts


def row(df: pd.DataFrame, txn_id: int) -> pd.Series:
    return df.loc[df["transaction_id"] == txn_id].iloc[0]


class TestBuildModelFrame:
    def test_join_preserves_all_transactions(self, sample_data):
        txns, accts = sample_data
        result = build_model_frame(txns, accts)
        assert len(result) == 3

    def test_account_fields_merged(self, sample_data):
        txns, accts = sample_data
        result = build_model_frame(txns, accts)
        assert "customer_name" in result.columns
        assert row(result, 1)["customer_name"] == "Alice"

    # is_large_amount
    def test_is_large_amount_set_for_1000_plus(self, sample_data):
        txns, accts = sample_data
        result = build_model_frame(txns, accts)
        assert row(result, 1)["is_large_amount"] == 1

    def test_is_large_amount_clear_below_1000(self, sample_data):
        txns, accts = sample_data
        result = build_model_frame(txns, accts)
        assert row(result, 2)["is_large_amount"] == 0
        assert row(result, 3)["is_large_amount"] == 0

    # login_pressure bins: (-1,0] → none, (0,2] → low, (2,100] → high
    def test_login_pressure_none_at_zero(self, sample_data):
        txns, accts = sample_data
        result = build_model_frame(txns, accts)
        assert str(row(result, 1)["login_pressure"]) == "none"

    def test_login_pressure_low_at_two(self, sample_data):
        txns, accts = sample_data
        result = build_model_frame(txns, accts)
        assert str(row(result, 2)["login_pressure"]) == "low"

    def test_login_pressure_high_above_two(self, sample_data):
        txns, accts = sample_data
        result = build_model_frame(txns, accts)
        assert str(row(result, 3)["login_pressure"]) == "high"

    def test_unmatched_account_produces_null_not_dropped(self):
        txns = pd.DataFrame([
            {"transaction_id": 99, "account_id": 999, "amount_usd": 100.0, "failed_logins_24h": 0},
        ])
        accts = pd.DataFrame([{"account_id": 101, "customer_name": "Alice"}])
        result = build_model_frame(txns, accts)
        # Left join: transaction must still be present even with no matching account
        assert len(result) == 1
        assert result.iloc[0]["transaction_id"] == 99
