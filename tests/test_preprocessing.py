"""Tests for preprocessing helper functions."""

from __future__ import annotations

import pandas as pd

from src.preprocessing import (
    build_preprocessor,
    split_features_target,
    validate_requested_features,
)


def test_numeric_missing_values_are_processed() -> None:
    dataframe = pd.DataFrame({"amount": [1.0, None, 3.0]})
    preprocessor = build_preprocessor(["amount"], [], {"scale_numeric_for_logistic": True})

    transformed = preprocessor.fit_transform(dataframe)

    assert transformed.shape[0] == 3


def test_categorical_missing_values_are_processed() -> None:
    dataframe = pd.DataFrame({"product": ["a", None, "b"]})
    preprocessor = build_preprocessor(
        [],
        ["product"],
        {
            "categorical_fill_value": "__MISSING__",
            "one_hot_handle_unknown": "ignore",
            "one_hot_min_frequency": 1,
            "one_hot_max_categories": 10,
        },
    )

    transformed = preprocessor.fit_transform(dataframe)

    assert transformed.shape[0] == 3


def test_unknown_categories_do_not_error() -> None:
    train_df = pd.DataFrame({"product": ["a", "b"]})
    test_df = pd.DataFrame({"product": ["new_category"]})
    preprocessor = build_preprocessor(
        [],
        ["product"],
        {
            "one_hot_handle_unknown": "ignore",
            "one_hot_min_frequency": 1,
            "one_hot_max_categories": 10,
        },
    )

    preprocessor.fit(train_df)
    transformed = preprocessor.transform(test_df)

    assert transformed.shape[0] == 1


def test_illegal_feature_group_columns_are_rejected() -> None:
    dataframe = pd.DataFrame({"TransactionID": [1], "amount": [1.0], "isFraud": [0]})

    try:
        validate_requested_features(
            dataframe,
            {"transaction_basic": ["TransactionID", "amount"]},
            "transaction_basic",
            "isFraud",
            "TransactionID",
            "TransactionDT",
        )
    except ValueError as error:
        assert "TransactionID" in str(error)
    else:
        raise AssertionError("Expected illegal feature to raise ValueError.")


def test_id_target_and_time_do_not_enter_model_features() -> None:
    dataframe = pd.DataFrame(
        {"TransactionID": [1], "TransactionDT": [10], "amount": [1.0], "isFraud": [0]}
    )
    features = validate_requested_features(
        dataframe,
        {"transaction_basic": ["amount"]},
        "transaction_basic",
        "isFraud",
        "TransactionID",
        "TransactionDT",
    )

    assert features == ["amount"]


def test_build_preprocessor_handles_empty_column_groups() -> None:
    numeric_only = build_preprocessor(["amount"], [], {"scale_numeric_for_logistic": False})
    categorical_only = build_preprocessor(
        [],
        ["product"],
        {"one_hot_handle_unknown": "ignore", "one_hot_min_frequency": 1},
    )

    assert numeric_only.fit_transform(pd.DataFrame({"amount": [1.0, None]})).shape[0] == 2
    assert categorical_only.fit_transform(pd.DataFrame({"product": ["a", None]})).shape[0] == 2


def test_split_features_target_returns_expected_x_and_y() -> None:
    dataframe = pd.DataFrame({"amount": [1.0, 2.0], "isFraud": [0, 1], "extra": [9, 9]})

    X, y = split_features_target(dataframe, ["amount"], "isFraud")

    assert list(X.columns) == ["amount"]
    assert y.tolist() == [0, 1]
