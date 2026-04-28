from risk_rules import label_risk, score_transaction


def base_tx(**overrides) -> dict:
    """Minimal zero-risk transaction; override individual signals to test each rule."""
    tx = {
        "device_risk_score": 10,
        "is_international": 0,
        "amount_usd": 50.0,
        "velocity_24h": 1,
        "failed_logins_24h": 0,
        "prior_chargebacks": 0,
    }
    tx.update(overrides)
    return tx


# ---------------------------------------------------------------------------
# label_risk — boundary values
# ---------------------------------------------------------------------------

class TestLabelRisk:
    def test_low_bottom(self):
        assert label_risk(0) == "low"

    def test_low_top(self):
        assert label_risk(29) == "low"

    def test_medium_bottom(self):
        assert label_risk(30) == "medium"

    def test_medium_top(self):
        assert label_risk(59) == "medium"

    def test_high_bottom(self):
        assert label_risk(60) == "high"

    def test_high_top(self):
        assert label_risk(100) == "high"


# ---------------------------------------------------------------------------
# score_transaction — each signal in isolation with exact expected values
# ---------------------------------------------------------------------------

class TestScoreTransactionSignals:
    def test_clean_transaction_scores_zero(self):
        assert score_transaction(base_tx()) == 0

    # device_risk_score
    def test_high_device_risk_adds_25(self):
        assert score_transaction(base_tx(device_risk_score=70)) == 25

    def test_medium_device_risk_adds_10(self):
        assert score_transaction(base_tx(device_risk_score=40)) == 10

    def test_low_device_risk_adds_nothing(self):
        assert score_transaction(base_tx(device_risk_score=39)) == 0

    # is_international
    def test_international_adds_15(self):
        assert score_transaction(base_tx(is_international=1)) == 15

    def test_domestic_adds_nothing(self):
        assert score_transaction(base_tx(is_international=0)) == 0

    # amount_usd
    def test_large_amount_adds_25(self):
        assert score_transaction(base_tx(amount_usd=1000)) == 25

    def test_medium_amount_adds_10(self):
        assert score_transaction(base_tx(amount_usd=500)) == 10

    def test_small_amount_adds_nothing(self):
        assert score_transaction(base_tx(amount_usd=499)) == 0

    # velocity_24h
    def test_high_velocity_adds_20(self):
        assert score_transaction(base_tx(velocity_24h=6)) == 20

    def test_medium_velocity_adds_5(self):
        assert score_transaction(base_tx(velocity_24h=3)) == 5

    def test_low_velocity_adds_nothing(self):
        assert score_transaction(base_tx(velocity_24h=2)) == 0

    # failed_logins_24h
    def test_many_failed_logins_adds_20(self):
        assert score_transaction(base_tx(failed_logins_24h=5)) == 20

    def test_few_failed_logins_adds_10(self):
        assert score_transaction(base_tx(failed_logins_24h=2)) == 10

    def test_one_failed_login_adds_nothing(self):
        assert score_transaction(base_tx(failed_logins_24h=1)) == 0

    # prior_chargebacks
    def test_multiple_prior_chargebacks_adds_20(self):
        assert score_transaction(base_tx(prior_chargebacks=2)) == 20

    def test_one_prior_chargeback_adds_5(self):
        assert score_transaction(base_tx(prior_chargebacks=1)) == 5

    def test_no_prior_chargebacks_adds_nothing(self):
        assert score_transaction(base_tx(prior_chargebacks=0)) == 0


# ---------------------------------------------------------------------------
# score_transaction — compound scoring and clamping
# ---------------------------------------------------------------------------

class TestScoreTransactionCompound:
    def test_all_signals_clamped_at_100(self):
        # 25 + 15 + 25 + 20 + 20 + 20 = 125, must clamp to 100
        tx = base_tx(
            device_risk_score=90,
            is_international=1,
            amount_usd=2000,
            velocity_24h=10,
            failed_logins_24h=10,
            prior_chargebacks=3,
        )
        assert score_transaction(tx) == 100

    def test_high_risk_profile_labeled_high(self):
        # device=85 (+25), international (+15), amount=1400 (+25),
        # velocity=8 (+20), logins=5 (+20), chargebacks=2 (+20) = 125 → 100
        tx = base_tx(
            device_risk_score=85,
            is_international=1,
            amount_usd=1400,
            velocity_24h=8,
            failed_logins_24h=5,
            prior_chargebacks=2,
        )
        assert label_risk(score_transaction(tx)) == "high"

    def test_two_moderate_signals_reach_medium(self):
        # medium device (+10) + medium amount (+10) = 20 → low
        # add international (+15) → 35 → medium
        tx = base_tx(device_risk_score=50, amount_usd=600, is_international=1)
        assert label_risk(score_transaction(tx)) == "medium"

    def test_score_never_negative(self):
        assert score_transaction(base_tx()) >= 0
