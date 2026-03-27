from datetime import datetime
import pandas as pd
import sqlite3
import os
import time
import requests
from . import my_base_func as bafu

def FetchFredSeriesList(
    series_ids: list[str],
    start_date: str,
    end_date: str,
    api_key: str,
    max_retries: int = 3,
    retry_delay: int = 1
) -> list[pd.DataFrame]:
    """
    Fetch multiple time series from the FRED API.
    """

    if end_date == "now":
        end_date = datetime.today().strftime("%Y-%m-%d")

    base_url = "https://api.stlouisfed.org/fred/series/observations"
    df_list: list[pd.DataFrame] = []

    bafu.WriteLog("data_fetched/data_fetch.log", 
        f"Data fetch started | start={start_date}, end={end_date}"
    )

    for sid in series_ids:

        bafu.WriteLog("data_fetched/data_fetch.log", f"Fetching series: {sid}")

        params = {
            "series_id": sid,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": start_date,
            "observation_end": end_date
        }

        success = False

        for attempt in range(1, max_retries + 1):

            try:
                response = requests.get(
                    base_url,
                    params=params,
                    timeout=10
                )
                response.raise_for_status()

                observations = response.json()["observations"]

                df = pd.DataFrame(observations)
                df["series_id"] = sid
                df_list.append(df)

                bafu.WriteLog("data_fetched/data_fetch.log", 
                    f"{sid} successfully retrieved "
                    f"(rows={len(df)})"
                )

                success = True
                break

            except Exception as e:
                bafu.WriteLog("data_fetched/data_fetch.log", 
                    f"{sid} retry {attempt}/{max_retries} failed: {e}"
                )
                time.sleep(retry_delay)

        if not success:
            bafu.WriteLog("data_fetched/data_fetch.log", f"{sid} failed after {max_retries} retries")

    bafu.WriteLog("data_fetched/data_fetch.log", "Data fetch completed")

    return df_list


def StoreDataframesToSqlite(
    table_map: dict[str, pd.DataFrame],
    db_name: str,
    mode: str = "append",
    db_dir: str = "data_fetched"
) -> None:
    """
    Store multiple DataFrames into SQLite tables.

    Parameters
    ----------
    table_map : dict[str, pd.DataFrame]
        Table name -> DataFrame
    mode :
        "replace" : drop table and recreate
        "append"  : append rows
    """

    log_path = "data_fetched/data_fetch.log"

    if mode not in {"replace", "append"}:
        raise ValueError("mode must be 'replace' or 'append'")

    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, db_name)

    bafu.WriteLog(log_path, f"Connecting to database: {db_path}")
    bafu.WriteLog(log_path, f"Write mode selected: {mode}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for table_name, df in table_map.items():

        bafu.WriteLog(log_path, f"Processing table: {table_name}")

        if df.empty:
            bafu.WriteLog(log_path, f"Table {table_name}: empty — skipped")
            continue

        # ---------- REPLACE MODE ----------
        if mode == "replace":
            cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
            bafu.WriteLog(
                log_path,
                f"Table {table_name}: dropped"
            )

            df.to_sql(
                table_name,
                conn,
                if_exists="replace",
                index=False
            )

        # ---------- APPEND MODE ----------
        else:
            df.to_sql(
                table_name,
                conn,
                if_exists="append",
                index=False
            )

        bafu.WriteLog(
            log_path,
            f"Table {table_name} written (rows={len(df)})"
        )

    conn.commit()
    conn.close()

    bafu.WriteLog(log_path, "Database commit completed")


def CreateDictName(
    dataframes: list[pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    """
    Generate a dictionary mapping table names to DataFrames.

    Table names follow the format:
    raw_{series_id}_{start_date}_{end_date}
    """

    table_map: dict[str, pd.DataFrame] = {}

    for df in dataframes:

        if df.empty:
            continue

        series_id = str(df["series_id"].iloc[0])

        start_date = df["date"].min().replace("-", "")
        end_date = df["date"].max().replace("-", "")

        table_name = f"raw_{series_id}_{start_date}_{end_date}"

        table_map[table_name] = df

    return table_map


def FilterExistingSeriesIds(
    data_id: list[str],
    db_name: str,
    db_dir: str = "data_fetched",
) -> list[str]:
    """
    Remove series IDs that already exist in the SQLite database.

    A series is considered existing if any table name contains
    the corresponding series_id.

    Parameters
    ----------
    data_id : list[str]
        List of FRED series IDs.
    db_name : str
        SQLite database file name.
    db_dir : str
        Directory containing the database.

    Returns
    -------
    list[str]
        Filtered list of series IDs not yet stored.
    """

    db_path = os.path.join(db_dir, db_name)

    # If database does not exist, nothing is stored yet
    if not os.path.exists(db_path):
        bafu.WriteLog("data_fetched/data_fetch.log", "Database not found — all series will be fetched.")
        return data_id

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Retrieve all table names
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table';"
    )
    tables = [row[0] for row in cursor.fetchall()]

    conn.close()

    remaining_ids = []

    for sid in data_id:

        # Check whether this series already exists
        exists = any(sid in table_name for table_name in tables)

        if exists:
            bafu.WriteLog("data_fetched/data_fetch.log", f"{sid} already exists — skipped")
        else:
            remaining_ids.append(sid)

    bafu.WriteLog("data_fetched/data_fetch.log", 
        f"Series remaining for fetch: {remaining_ids}"
    )

    return remaining_ids



def SanitizeColumnNamesDict(
    df_dict: dict[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    """
    Make all DataFrame column names SQLite-safe.

    Parameters
    ----------
    df_dict : dict[str, pd.DataFrame]

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary with sanitized column names.
    """

    log_path = "data_fetched/data_fetch.log"

    bafu.WriteLog(
        log_path,
        f"Column name sanitization started (tables={len(df_dict)})"
    )

    cleaned_dict: dict[str, pd.DataFrame] = {}

    for name, df in df_dict.items():

        if df.empty:
            bafu.WriteLog(
                log_path,
                f"{name}: empty DataFrame — skipped"
            )
            continue

        df_clean = df.copy()

        original_cols = df_clean.columns.tolist()

        df_clean.columns = (
            df_clean.columns
            .str.replace("-", "_", regex=False)
            .str.replace(" ", "_", regex=False)
            .str.replace(".", "_", regex=False)
            .str.replace("/", "_", regex=False)
        )

        if original_cols != df_clean.columns.tolist():
            bafu.WriteLog(
                log_path,
                f"{name}: column names sanitized"
            )

        cleaned_dict[name] = df_clean

    bafu.WriteLog(
        log_path,
        "Column name sanitization completed"
    )

    return cleaned_dict
