import pandas as pd
import re


def convert_to_snake_case(column):
    if column == "CustomerID":
        column = "customer_id"
    snake_case = re.sub(r"(?<!^)([A-Z])", r"_\1", column)
    snake_case = snake_case.lower()
    return snake_case


def rename_col(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.columns = [convert_to_snake_case(col) for col in df.columns]
        return df
    except Exception as e:
        print(e)


def summarize_categoricals(df, show_levels=False):
    """
    Display uniqueness in each column
    df: dataframe contains only catergorical features
    """
    data = [
        [df[c].unique().tolist(), len(df[c].unique()), df[c].isnull().sum()]
        for c in df.columns
    ]
    df_temp = pd.DataFrame(
        data,
        index=df.columns,
        columns=["Levels", "No. of Levels", "No. of Missing Values"],
    )
    return df_temp.iloc[:, 0 if show_levels else 1 :]


def find_categorical(df, cutoff=12):
    """
    Function to find categorical columns in the dataframe.
    cutoff: is determinied when plotting the histogram distribution for numerical cols
    """
    cat_cols = []
    for col in df.columns:
        if len(df[col].unique()) <= cutoff:
            cat_cols.append(col)
    return cat_cols


def to_categorical(cat_cols, df):
    """
    Converts the columns passed in `columns` to categorical datatype for keras model
    """
    for col in cat_cols:
        df[col] = df[col].astype("category")
    return df
