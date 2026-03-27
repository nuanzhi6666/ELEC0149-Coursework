from datetime import datetime
import pandas as pd
import sqlite3
import os
from . import my_base_func as bafu
import numpy as np

def SplitDatasetByTime(
    df: pd.DataFrame,
    ratio: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> dict[str, pd.DataFrame]:
    """
    Split dataset into train / validation / test sets by time order.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset (must be sorted by date).
    ratio : tuple[float, float, float]
        Split ratio (train, val, test), default = 70/15/15.

    Returns
    -------
    dict[str, pd.DataFrame]
        {
            "train": train_df,
            "val": val_df,
            "test": test_df
        }
    """

    log_path = "data_fetched/data_fetch.log"

    bafu.WriteLog(
        log_path,
        "Dataset time-based split started"
    )

    total = len(df)

    if total == 0:
        bafu.WriteLog(
            log_path,
            "Dataset split failed: empty DataFrame"
        )
        return {}

    train_end = int(total * ratio[0])
    val_end = train_end + int(total * ratio[1])

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    bafu.WriteLog(
        log_path,
        f"Dataset split completed "
        f"(train={len(train_df)}, val={len(val_df)}, test={len(test_df)})"
    )

    return {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }


def BuildStateDataset(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    rolling_mean_cols: list[str],
    lookback: int = 20,
    horizon: int = 5,
    task: str = "regression",
    theta: float = 0.002,
) -> pd.DataFrame:
    """
    Build state-level dataset from time series.

    - No lag features
    - Rolling mean features only
    - Future cumulative target

    Each row corresponds to one timestep t.
    """

    log_path = "data_fetched/data_fetch.log"

    bafu.WriteLog(log_path, "State dataset construction started")

    data = df.copy().reset_index(drop=True)

    if target_col not in data.columns:
        bafu.WriteLog(
            log_path,
            f"Dataset construction failed: target column not found ({target_col})"
        )
        return pd.DataFrame()

    # -----------------------------
    # future cumulative target
    # -----------------------------
    future = data[target_col].shift(-1).rolling(horizon).sum()

    if task == "regression":
        y = future
        bafu.WriteLog(log_path, "Regression task selected")

    elif task == "classification":
        y = pd.Series(0, index=data.index)
        y[future > theta] = 1
        y[future < -theta] = -1
        bafu.WriteLog(
            log_path,
            f"Classification task selected (theta={theta})"
        )
    else:
        bafu.WriteLog(
            log_path,
            f"Dataset construction failed: invalid task type ({task})"
        )
        return pd.DataFrame()

    # -----------------------------
    # valid range (no future leak)
    # -----------------------------
    valid_end = len(data) - horizon
    data = data.iloc[:valid_end].copy()
    y = y.iloc[:valid_end].copy()

    # -----------------------------
    # feature column check
    # -----------------------------
    feature_cols = [c for c in feature_cols if c in data.columns]

    if rolling_mean_cols is None:
        rolling_mean_cols = []

    rows = []

    for t in range(lookback, len(data)):

        row = {}

        # -----------------------------
        # current timestep features
        # -----------------------------
        for col in feature_cols:
            row[col] = data.loc[t, col]

        # -----------------------------
        # rolling mean features
        # -----------------------------
        for col in rolling_mean_cols:

            if col not in data.columns:
                bafu.WriteLog(
                    log_path,
                    f"Rolling mean skipped: column not found ({col})"
                )
                continue

            window = data.loc[t - lookback:t - 1, col]
            row[f"{col}_mean_{lookback}"] = window.mean()

        # -----------------------------
        # target
        # -----------------------------
        row["target"] = y.iloc[t]

        # -----------------------------
        # optional timestamp
        # -----------------------------
        if "date" in data.columns:
            row["date"] = data.loc[t, "date"]

        rows.append(row)

    dataset = pd.DataFrame(rows)

    bafu.WriteLog(
        log_path,
        f"State dataset construction completed "
        f"(samples={len(dataset)}, "
        f"features={len(feature_cols)}, "
        f"rolling_mean_cols={len(rolling_mean_cols)})"
    )

    return dataset


def GaussianNormalizeByTarget(
    df_dict: dict[str, pd.DataFrame],
    target_dfname: str = "train",
    exclude_cols: list[str] = ["date", "target"]
) -> dict[str, pd.DataFrame]:
    """
    Apply Gaussian normalization to all datasets using statistics
    computed from a target dataset (e.g. train set).

    x' = (x - mean_train) / std_train
    """

    log_path = "data_fetched/data_fetch.log"

    if target_dfname not in df_dict:
        bafu.WriteLog(
            log_path,
            f"Gaussian normalization failed: target dataset not found ({target_dfname})"
        )
        return df_dict

    bafu.WriteLog(
        log_path,
        f"Gaussian normalization started using statistics from '{target_dfname}'"
    )

    df_target = df_dict[target_dfname]

    # select numeric columns only
    numeric_cols = [
        c for c in df_target.columns
        if c not in exclude_cols
        and pd.api.types.is_numeric_dtype(df_target[c])
    ]

    if len(numeric_cols) == 0:
        bafu.WriteLog(
            log_path,
            "Gaussian normalization failed: no numeric columns found in target dataset"
        )
        return df_dict

    # compute statistics from target dataset only
    mean_map = {}
    std_map = {}

    for col in numeric_cols:
        mean_map[col] = df_target[col].mean()
        std_map[col] = df_target[col].std()

    bafu.WriteLog(
        log_path,
        f"Computed normalization statistics (features={len(numeric_cols)})"
    )

    normalized_dict = {}

    for name, df in df_dict.items():

        df_norm = df.copy()

        for col in numeric_cols:

            if col not in df_norm.columns:
                continue

            std = std_map[col]

            if std == 0 or pd.isna(std):
                continue

            df_norm[col] = (df_norm[col] - mean_map[col]) / std

        normalized_dict[name] = df_norm

        bafu.WriteLog(
            log_path,
            f"Gaussian normalized dataset: {name} (rows={len(df_norm)})"
        )

    bafu.WriteLog(
        log_path,
        "Gaussian normalization completed for all datasets"
    )

    return normalized_dict


def DropAbnormalRows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows containing NaN or infinite values.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with abnormal rows removed.
    """

    log_path = "data_fetched/data_fetch.log"

    bafu.WriteLog(log_path, "Abnormal row removal started")

    df_clean = df.copy()

    numeric_df = df_clean.select_dtypes(include="number")

    mask = (
        numeric_df.isna() |
        np.isinf(numeric_df)
    ).any(axis=1)

    removed_rows = int(mask.sum())

    df_clean = df_clean.loc[~mask].reset_index(drop=True)

    bafu.WriteLog(
        log_path,
        f"Abnormal row removal completed (removed_rows={removed_rows})"
    )

    return df_clean


def AddLogReturnForColumns(
    df: pd.DataFrame,
    cols: list[str],
    suffix: str = "_logret"
) -> pd.DataFrame:
    """
    Compute log returns for specified columns.

    log_return_t = log(x_t / x_{t-1})

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (must contain 'date' column).
    cols : list[str]
        Columns to compute log returns for.
    suffix : str
        Suffix for new log return columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with added log return columns.
    """

    log_path = "data_fetched/data_fetch.log"

    bafu.WriteLog(log_path, "Log return computation started")

    out = df.copy()

    # ensure time order
    if "date" in out.columns:
        out = out.sort_values("date")

    for col in cols:

        if col not in out.columns:
            bafu.WriteLog(
                log_path,
                f"Log return skipped: column not found ({col})"
            )
            continue

        series = pd.to_numeric(out[col], errors="coerce")

        # log requires positive values
        series = series.where(series > 0)

        out[f"{col}{suffix}"] = np.log(series / series.shift(1))

        bafu.WriteLog(
            log_path,
            f"Log return computed for column {col}"
        )

    out.reset_index(drop=True, inplace=True)

    bafu.WriteLog(log_path, "Log return computation completed")

    return out


def LoadSqliteTablesAsDataframes(
    db_name: str,
    db_dir: str = "data_fetched",
) -> dict[str, pd.DataFrame]:
    """
    Load all tables from a SQLite database into pandas DataFrames.

    Each table is returned as one DataFrame.

    Parameters
    ----------
    db_name : str
        SQLite database file name.
    db_dir : str
        Directory containing the database.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping:
        {
            table_name: pandas.DataFrame
        }
    """

    log_path = "data_fetched/data_processing.log"
    db_path = os.path.join(db_dir, db_name)

    if not os.path.exists(db_path):
        bafu.WriteLog(
            log_path,
            f"Database not found: {db_path}"
        )
        return {}

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # retrieve all user-defined tables
    cursor.execute("""
        SELECT name
        FROM sqlite_master
        WHERE type='table'
          AND name NOT LIKE 'sqlite_%';
    """)
    tables = [row[0] for row in cursor.fetchall()]

    dataframes: dict[str, pd.DataFrame] = {}

    for table in tables:

        try:
            df = pd.read_sql_query(
                f"SELECT * FROM {table}",
                conn
            )

            dataframes[table] = df

            bafu.WriteLog(
                log_path,
                f"Loaded table: {table} ({len(df)} rows)"
            )

        except Exception as e:
            bafu.WriteLog(
                log_path,
                f"Failed to load table {table}: {e}"
            )

    conn.close()

    bafu.WriteLog(
        log_path,
        "SQLite tables successfully loaded into DataFrames"
    )

    return dataframes


def CleanAndRekeyDataframes(
    dbdict: dict[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    """
    Clean raw FRED DataFrames and re-key dictionary by series_id.

    Parameters
    ----------
    dbdict : dict[str, pd.DataFrame]
        Dictionary mapping table names to raw DataFrames.

    Returns
    -------
    dict[str, pd.DataFrame]
        Cleaned DataFrames mapped by series_id.
    """

    log_path = "data_fetched/data_processing.log"
    cleaned_dict: dict[str, pd.DataFrame] = {}

    bafu.WriteLog(
        log_path,
        f"Data cleaning started (tables={len(dbdict)})"
    )

    for table_name, df in dbdict.items():

        if df.empty:
            bafu.WriteLog(
                log_path,
                f"Table {table_name} skipped — empty DataFrame"
            )
            continue

        df = df.copy()

        # series_id should be constant in each table
        series_id = str(df["series_id"].iloc[0])

        bafu.WriteLog(
            log_path,
            f"Cleaning table {table_name} → series_id={series_id}"
        )

        # drop unused columns
        drop_cols = ["realtime_start", "realtime_end"]
        df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

        # type conversion
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        # remove duplicates
        before_rows = len(df)
        df.drop_duplicates(inplace=True)
        after_rows = len(df)

        if before_rows != after_rows:
            bafu.WriteLog(
                log_path,
                f"{series_id}: duplicates removed "
                f"({before_rows} → {after_rows})"
            )

        # sort by time
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)

        cleaned_dict[series_id] = df

        bafu.WriteLog(
            log_path,
            f"{series_id} cleaned successfully (rows={len(df)})"
        )

    bafu.WriteLog(
        log_path,
        f"Data cleaning completed (series={len(cleaned_dict)})"
    )

    return cleaned_dict


def ForwardFillMergedDataframe(
    merged_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Forward-fill NaN values in merged time series DataFrame.

    Each column is filled independently using the last
    available observation.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Merged DataFrame with 'date' column.

    Returns
    -------
    pd.DataFrame
        Forward-filled DataFrame.
    """

    df = merged_df.copy()

    if "date" not in df.columns:
        raise ValueError("DataFrame must contain 'date' column")

    df = df.sort_values("date")

    factor_cols = [c for c in df.columns if c != "date"]

    df[factor_cols] = df[factor_cols].ffill()

    return df


def MergeSeriesDataframes(
    dbdict: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Merge multiple time series DataFrames by date.

    Each DataFrame must contain:
    - date
    - value

    The output DataFrame uses 'date' as index column
    and each series_id as a separate column.

    Parameters
    ----------
    dbdict : dict[str, pd.DataFrame]
        Dictionary mapping series_id to DataFrame.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame.
    """

    merged_df: pd.DataFrame | None = None

    for series_id, df in dbdict.items():

        if df.empty:
            continue

        temp_df = df[["date", "value"]].copy()
        temp_df.rename(columns={"value": series_id}, inplace=True)

        if merged_df is None:
            merged_df = temp_df
        else:
            merged_df = pd.merge(
                merged_df,
                temp_df,
                on="date",
                how="outer"
            )

    if merged_df is None:
        return pd.DataFrame()

    merged_df.sort_values("date", inplace=True)
    merged_df.reset_index(drop=True, inplace=True)

    return merged_df
