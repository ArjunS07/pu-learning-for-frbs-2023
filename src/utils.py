import ast
from enum import Enum
import re
from collections import Counter
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, recall_score


def get_subburst_preserved_train_test(
    original_df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.DataFrame,
    test_size: float,
    stratify=True,
) -> Tuple[pd.DataFrame, np.array, pd.DataFrame, np.array]:
    """
    The purpose of this function is to make sure that all sub-bursts for a particular burst are together, whether in the training set or the test set.
    This is done by performing the train-test split first only with the first sub-burst of a particular FRB. Then, we add back the corresponding
    sub-bursts for each burst back to the training and test sets.
    """
    subburst_tns_names = original_df[original_df["sub_num"] != 0]["tns_name"].unique()
    # print(f' {len(subburst_tns_names)=} FRBs have sub-bursts. FRBs: {subburst_tns_names=}')

    # Isolate the first bursts of each FRB. sub_num is 0 for bursts with only one sub-burst (i.e., one burst) as per the format provided by the CHIME catalog
    first_burst_indices = list(original_df[original_df["sub_num"] == 0].index)
    first_burst_X = X.iloc[first_burst_indices]
    first_burst_Y = y.iloc[first_burst_indices]

    # Train test split only on non-subburst bursts. We add the corresponding subbursts for each burst later.
    if (
        stratify
    ):  # Stratification is done to preserve the ratio of repeaters to non-repeaters in the training and test sets. We allow the user of the function to specify whether they want to implement it
        X_train, X_test, y_train, y_test = train_test_split(
            first_burst_X,
            first_burst_Y,
            test_size=test_size,
            stratify=first_burst_Y,
        )
        # print(f'After initial split, {Counter(y_train)}')
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            first_burst_X, first_burst_Y, test_size=test_size
        )

    # print(f'Before adding subbursts: {X_train.shape=}, {y_train.shape=}. {Counter(y_train)=}')
    # print('Before adding subbursts, n unique TNS names:', X_train['tns_name'].nunique())
    # X_train only contains the first sub-burst of each FRB
    for index, row in X_train.iterrows():
        # Because X_train only contains raw feature data, we need to refer back to the original df to get the corresponding TNS name for that burst.
        # NOTE: This assumes that the order of the indexing has been preserved, which it should be as per how pandas works
        tns_name = original_df.iloc[index]["tns_name"]
        if tns_name in subburst_tns_names:
            # A TNS name only repeats for the same sub-burst. Note that each individual repeating event has its own tns_name.
            burst_subbursts = original_df[original_df["tns_name"] == tns_name]
            burst_subburst_indices = list(burst_subbursts.index)
            burst_subburst_indices = [
                i for i in burst_subburst_indices if i != index
            ]  # remove the index of the first sub-burst itself

            # Add the corresponding subbursts for that burst to the training set
            subburst_rows = X.iloc[burst_subburst_indices]
            X_train = pd.concat([X_train, subburst_rows])
            y_train = pd.concat(
                [y_train, original_df.iloc[burst_subburst_indices]["is_repeater"]]
            )
    # print(f'After adding subbursts: {X_train.shape=} {y_train.shape=}. {Counter(y_train)=}')
    # print('After adding subbursts, n unique TNS names:', X_train['tns_name'].nunique())

    # Perform the same for X_test
    for index, row in X_test.iterrows():
        tns_name = original_df.iloc[index]["tns_name"]
        if tns_name in subburst_tns_names:
            subbursts = original_df[original_df["tns_name"] == tns_name]
            subburst_indices = list(subbursts.index)
            subburst_indices = [i for i in subburst_indices if i != index]
            subburst_rows = X.iloc[subburst_indices]
            X_test = pd.concat([X_test, subburst_rows])
            y_test = pd.concat(
                [y_test, original_df.iloc[subburst_indices]["is_repeater"]]
            )

    return (
        X_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )


def lee_liu_score(y_known, y_pred) -> float:
    # As per method develoepd by Lee and Liu (2003) in Learning with positive and unlabeled examples using weighted logistic regression.
    recall = recall_score(y_known, y_pred)
    pos_predictions_frequency = sum(y_pred == 1) / len(y_pred)
    # Avoid divide-by-zero error
    if pos_predictions_frequency == 0:
        return 0
    return (recall**2) / pos_predictions_frequency


ll_scorer = make_scorer(lee_liu_score)


def round_to_n_significant_figures(df, n):
    """
    Round all numerical columns in a pandas DataFrame to N significant figures.

    Parameters:
        df (pandas.DataFrame): Input DataFrame.
        n (int): Number of significant figures to round to.

    Returns:
        pandas.DataFrame: DataFrame with rounded numerical columns.
    """

    def round_to_n_significant(value, n):
        if value == 0:
            return 0
        return round(value, -int(np.floor(np.log10(abs(value)))) + (n - 1))

    rounded_df = df.copy()

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            rounded_df[col] = df[col].apply(lambda x: round_to_n_significant(x, n))

    return rounded_df


def round_df_to_n_significant_figures(df, n):
    def round_to_n_significant(value, n):
        if value == 0:
            return 0
        return round(value, -int(np.floor(np.log10(abs(value)))) + (n - 1))

    def remove_trailing_zeros(value):
        if isinstance(value, (int, float)):
            str_value = str(value)
            stripped_value = re.sub(r"\.?0*$", "", str_value)
            return (
                float(stripped_value) if "." in stripped_value else int(stripped_value)
            )
        return value

    rounded_df = df.copy()

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            rounded_df[col] = (
                df[col]
                .apply(lambda x: round_to_n_significant(x, n))
                .apply(remove_trailing_zeros)
            )

    return rounded_df


PCC_REPEATER_CANDIDATES = [
    "FRB20190303D",
    "FRB20181201D",
    "FRB20190328C",
    "FRB20191105B",
    "FRB20190107B",
    "FRB20200320A",
    "FRB20190210C",
    "FRB20190812A",
    "FRB20200828A",
    "FRB20190905A",
    "FRB20190127B",
    "FRB20210323C",
    "FRB20180909A",
    "FRB20200508H",
]

# Luo et al 2022
SUPERVISED_PAPER_REPEATERS = [
    "FRB20181229B",
    "FRB20190423B",
    "FRB20190410A",
    "FRB20181017B",
    "FRB20181128C",
    "FRB20190422A",
    "FRB20190409B",
    "FRB20190329A",
    "FRB20190423B",
    "FRB20190206A",
    "FRB20190128C",
    "FRB20190106A",
    "FRB20190129A",
    "FRB20181030E",
    "FRB20190527A",
    "FRB20190218B",
    "FRB20190609A",
    "FRB20190412B",
    "FRB20190125B",
    "FRB20181231B",
    "FRB20181221A",
    "FRB20190112A",
    "FRB20190125A",
    "FRB20181218C",
    "FRB20190429B",
    "FRB20190109B",
    "FRB20190206B",
] + [
    "FRB20180801A",
    "FRB20180925A",
    "FRB20180928A",
    "FRB20181017B",
    "FRB20181119B",
    "FRB20181231B",
    "FRB20190107A",
    "FRB20190109B",
    "FRB20190124E",
    "FRB20190125B",
    "FRB20190128C",
    "FRB20190206A",
    "FRB20190329A",
    "FRB20190412B",
    "FRB20190423B",
    "FRB20190423B",
    "FRB20190429B",
    "FRB20190527A",
    "FRB20190617B",
]

# Zhu-Ge et al 2022
UNSUPERVISED_PAPER_REPEATERS = [
    "FRB20180907E",
    "FRB20180911A",
    "FRB20180915B",
    "FRB20180920B",
    "FRB20180923A",
    "FRB20180923C",
    "FRB20180928A",
    "FRB20181013E",
    "FRB20181017B",
    "FRB20181030E",
    "FRB20181125A",
    "FRB20181125A",
    "FRB20181125A",
    "FRB20181130A",
    "FRB20181214A",
    "FRB20181220A",
    "FRB20181221A",
    "FRB20181226E",
    "FRB20181229B",
    "FRB20181231B",
    "FRB20190106B",
    "FRB20190109B",
    "FRB20190110C",
    "FRB20190111A",
    "FRB20190112A",
    "FRB20190129A",
    "FRB20190204A",
    "FRB20190206A",
    "FRB20190218B",
    "FRB20190220A",
    "FRB20190221A",
    "FRB20190222B",
    "FRB20190223A",
    "FRB20190228A",
    "FRB20190308C",
    "FRB20190308C",
    "FRB20190308B",
    "FRB20190308B",
    "FRB20190323D",
    "FRB20190329A",
    "FRB20190403E",
    "FRB20190409B",
    "FRB20190410A",
    "FRB20190412B",
    "FRB20190418A",
    "FRB20190419A",
    "FRB20190422A",
    "FRB20190422A",
    "FRB20190423A",
    "FRB20190423B",
    "FRB20190423B",
    "FRB20190429B",
    "FRB20190430A",
    "FRB20190517C",
    "FRB20190527A",
    "FRB20190527A",
    "FRB20190531C",
    "FRB20190601B",
    "FRB20190601C",
    "FRB20190601C",
    "FRB20190609A",
    "FRB20190617A",
    "FRB20190617B",
    "FRB20190618A",
    "FRB20190625A",
]

# Pleunis et al. 2021
MORPHOLOGY_REPEATERS = [
    "FRB20181125A",
    "FRB20190308C",
    "FRB20190527A",
    "FRB20190422A",
    "FRB20190423A",
    "FRB20190601C",
]

REFERENCE_PAPER_REPEATERS = list(
    set(
        SUPERVISED_PAPER_REPEATERS + UNSUPERVISED_PAPER_REPEATERS + MORPHOLOGY_REPEATERS
    )
)


def c_to_alpha(s, c):
    # As per result in Bekker et al. 2020
    prob_labelled = sum(s == 1) / len(s)
    alpha = prob_labelled / c
    return alpha


class OptunaParamType(Enum):
    integer = "integer"
    categorical = "categorical"


class OptunaParam:
    name: str
    type: OptunaParamType
    options: list


from matplotlib.colors import LinearSegmentedColormap

custom_violet = "#5D3A9B"
custom_orange = "#E66100"
colors = [custom_violet, custom_orange]
values = [0.0, 1.0]
positions = np.linspace(0, 1, len(values))
custom_cmap = LinearSegmentedColormap.from_list(
    "custom_cmap", list(zip(positions, colors))
)
