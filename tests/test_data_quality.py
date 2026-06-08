import pandas as pd

from src.data_quality import (
    build_data_quality_report,
    build_feature_profile,
    count_id_overlaps,
    fraud_rate_check_passed,
    fraud_rate_difference,
    identify_feature_source,
)


def test_fraud_rate_difference_and_check() -> None:
    assert fraud_rate_difference(0.10, 0.101) == 0.0010000000000000009
    assert fraud_rate_check_passed(0.10, 0.101, 0.002)
    assert not fraud_rate_check_passed(0.10, 0.105, 0.002)


def test_feature_source_identification() -> None:
    assert identify_feature_source("TransactionAmt", ["TransactionAmt"], ["id_01"], []) == "transaction"
    assert identify_feature_source("id_01", ["TransactionAmt"], ["id_01"], []) == "identity"
    assert identify_feature_source("missing_count", [], [], []) == "derived_missing"
    assert identify_feature_source("id_01_is_missing", [], [], ["id_01_is_missing"]) == "derived_missing"


def test_feature_profile_missing_rate_and_unique_count() -> None:
    data = pd.DataFrame(
        {
            "TransactionAmt": [1.0, 2.0, None],
            "id_01": [1.0, 1.0, 2.0],
            "missing_count": [0, 1, 1],
        }
    )
    groups = {
        "transaction_basic": ["TransactionAmt"],
        "transaction_identity": ["TransactionAmt", "id_01"],
        "transaction_identity_missing": ["TransactionAmt", "id_01", "missing_count"],
    }

    profile = build_feature_profile(
        data,
        groups,
        transaction_columns=["TransactionAmt"],
        identity_columns=["id_01"],
        missing_indicator_columns=[],
    )

    amt = profile.loc[profile["feature"] == "TransactionAmt"].iloc[0]
    assert amt["source"] == "transaction"
    assert amt["missing_rate"] == 1 / 3
    assert int(amt["unique_count"]) == 2


def test_data_quality_report_numbers_come_from_inputs() -> None:
    transaction = pd.DataFrame({"TransactionID": [1, 2, 3, 4], "isFraud": [0, 1, 0, 1]})
    identity = pd.DataFrame({"TransactionID": [1, 2], "id_01": [10, 20]})
    merged = transaction.merge(identity, on="TransactionID", how="left", validate="one_to_one")
    sampled = merged.copy()
    splits = {
        "train": sampled.iloc[:2].copy(),
        "valid": sampled.iloc[2:3].copy(),
        "test": sampled.iloc[3:].copy(),
    }
    groups = {
        "transaction_basic": ["isFraud"],
        "transaction_identity": ["isFraud", "id_01"],
        "transaction_identity_missing": ["isFraud", "id_01", "missing_count"],
    }

    report = build_data_quality_report(
        transaction,
        identity,
        merged,
        sampled,
        splits,
        groups,
        requested_sample_size=4,
        missing_candidate_columns_not_found=["missing_col"],
        id_column="TransactionID",
        target_column="isFraud",
    )

    assert report["transaction_rows"] == 4
    assert report["identity_rows"] == 2
    assert report["identity_matched_rows"] == 2
    assert report["identity_match_rate"] == 0.5
    assert report["original_fraud_count"] == 2
    assert report["actual_sample_size"] == 4
    assert report["train_valid_id_overlap"] == 0
    assert report["missing_candidate_columns_not_found"] == ["missing_col"]


def test_id_overlap_count() -> None:
    left = pd.DataFrame({"TransactionID": [1, 2, 3]})
    right = pd.DataFrame({"TransactionID": [3, 4]})

    assert count_id_overlaps(left, right, "TransactionID") == 1
