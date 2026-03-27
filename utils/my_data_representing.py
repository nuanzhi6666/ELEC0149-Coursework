import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from . import my_base_func as bafu

def BuildSlidingWindowDict(
    data_dict: dict[str, pd.DataFrame],
    feature_cols: list[str],
    exclude_cols: list[str] | None = None,
    lookback: int = 20,
) -> dict[str, pd.DataFrame]:
    """
    Build sliding-window dataset using expanded columns (no ndarray).

    Each row contains:
        - pca_x_t-1 ... pca_x_t-lookback
        - target
        - date (optional)

    Suitable for storage in SQLite.
    """

    if exclude_cols is None:
        exclude_cols = []

    output = {}

    for name, df in data_dict.items():

        df = df.reset_index(drop=True)
        rows = []

        for t in range(lookback, len(df)):

            row = {}

            # --------------------------
            # sliding window features
            # --------------------------
            for lag in range(1, lookback + 1):
                for col in feature_cols:
                    row[f"{col}_t-{lag}"] = df.loc[t - lag, col]

            # --------------------------
            # exclude_cols
            # --------------------------
            for col in exclude_cols:
            
                row[col] = df.loc[t, col]

            rows.append(row)

        output[name] = pd.DataFrame(rows)

    return output




def ProjectDataframeByEigenvectors(
    df: pd.DataFrame,
    exclude_cols: list[str],
    eigvectors: np.ndarray,
    prefix: str = "pca"
) -> pd.DataFrame:
    """
    Project DataFrame features onto given eigenvectors.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (e.g. train / val / test).
    exclude_cols : list[str]
        Columns to exclude from projection (e.g. ["target", "date"]).
    eigvectors : np.ndarray
        PCA eigenvectors, shape = (n_components, n_features).
    prefix : str
        Prefix for generated PCA feature names.

    Returns
    -------
    pd.DataFrame
        DataFrame containing projected features and preserved excluded columns.
    """

    log_path = "data_fetched/data_representing.log"

    bafu.WriteLog(
        log_path,
        "PCA projection started"
    )

    # ------------------------------
    # select numeric feature columns
    # ------------------------------
    feature_cols = [
        c for c in df.columns
        if c not in exclude_cols
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    if len(feature_cols) == 0:
        bafu.WriteLog(
            log_path,
            "PCA projection failed: no valid feature columns"
        )
        return pd.DataFrame()

    X = df[feature_cols].values

    n_features = X.shape[1]
    n_components = eigvectors.shape[0]

    if eigvectors.shape[1] != n_features:
        bafu.WriteLog(
            log_path,
            f"PCA projection failed: eigenvector dimension mismatch "
            f"(eigvectors={eigvectors.shape}, features={n_features})"
        )
        return pd.DataFrame()

    # ------------------------------
    # projection
    # ------------------------------
    Z = X @ eigvectors.T   # (n_samples, n_components)

    pca_cols = [
        f"{prefix}_{i+1}"
        for i in range(n_components)
    ]

    df_pca = pd.DataFrame(
        Z,
        columns=pca_cols,
        index=df.index
    )

    # ------------------------------
    # keep excluded columns
    # ------------------------------
    preserved_cols = [
        c for c in exclude_cols
        if c in df.columns
    ]

    out_df = pd.concat(
        [df_pca, df[preserved_cols]],
        axis=1
    )

    bafu.WriteLog(
        log_path,
        f"PCA projection completed "
        f"(samples={len(df)}, components={n_components})"
    )

    return out_df



def ComputePCAFromDataframe(
    df: pd.DataFrame,
    exclude_cols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute PCA on a DataFrame after excluding specified columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (typically train set).
    exclude_cols : list[str], optional
        Column names to exclude before PCA (e.g. ["target", "date"]).

    Returns
    -------
    eigenvalues : np.ndarray
        Explained variance of each principal component.
    eigenvectors : np.ndarray
        Principal axes (components), shape = (n_components, n_features).
    """

    log_path = "data_fetched/data_representing.log"

    bafu.WriteLog(
        log_path,
        "PCA computation started"
    )

    if exclude_cols is None:
        exclude_cols = []

    # ------------------------------
    # select numeric feature columns
    # ------------------------------
    feature_cols = [
        c for c in df.columns
        if c not in exclude_cols
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    if len(feature_cols) == 0:
        bafu.WriteLog(
            log_path,
            "PCA failed: no valid numeric feature columns"
        )
        return np.array([]), np.array([])

    X = df[feature_cols].values

    bafu.WriteLog(
        log_path,
        f"PCA input prepared "
        f"(samples={X.shape[0]}, features={X.shape[1]})"
    )

    # ------------------------------
    # PCA (full components)
    # ------------------------------
    pca = PCA()
    pca.fit(X)

    eigenvalues = pca.explained_variance_
    eigenvectors = pca.components_

    bafu.WriteLog(
        log_path,
        f"PCA completed "
        f"(components={len(eigenvalues)})"
    )

    return eigenvalues, eigenvectors
