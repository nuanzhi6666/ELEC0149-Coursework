from utils import my_data_acqusition as dtac
from utils import my_data_processing as dtpr
from utils import my_data_representing as dtre
from utils import my_model_train as mdtr
from utils import my_base_func as dtba
from utils import my_llm_explainer as llmx

import os
import random
import json
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42):
    """
    Set random seed for full reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def map_return_to_action(value: float, theta: float) -> int:
    """
    Map continuous return to action.
    """
    if value > theta:
        return 1
    elif value < -theta:
        return -1
    else:
        return 0


def action_strength(pred_return: float, theta: float) -> float:
    """
    A simple confidence-style quantity for regression action mapping.
    Larger means the model is farther from the action boundary.
    """
    if pred_return > theta:
        return pred_return - theta
    elif pred_return < -theta:
        return abs(pred_return) - theta
    else:
        return theta - abs(pred_return)


def boundary_gap(pred_return: float, theta: float) -> float:
    """
    Distance to nearest decision boundary (+theta or -theta).
    Smaller means closer to action switch.
    """
    return abs(abs(pred_return) - theta)


def collect_test_regression_records(
    model,
    dataloader,
    source_df,
    seq_len,
    device,
    selected_features,
    theta,
):
    """
    Collect all test predictions for regression reporting.

    For each test sample, store:
    - date
    - true continuous return
    - predicted continuous return
    - mapped true action
    - mapped predicted action
    - correctness in action space
    - selected raw standardized features
    """
    model.eval()

    records = []
    row_index = seq_len  # BuildSequences starts from i=seq_len

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)

            preds = model(X_batch).detach().cpu().numpy()
            y_true_batch = y_batch.detach().cpu().numpy()

            batch_size = len(preds)

            for j in range(batch_size):
                current_row = source_df.iloc[row_index + j]

                pred_return = float(preds[j])
                true_return = float(y_true_batch[j])

                pred_action = map_return_to_action(pred_return, theta)
                true_action = map_return_to_action(true_return, theta)

                feature_snapshot = {}
                for col in selected_features:
                    if col in source_df.columns:
                        feature_snapshot[col] = current_row[col]

                records.append({
                    "date": str(current_row["date"]),
                    "pred_return": pred_return,
                    "true_return": true_return,
                    "pred_action": pred_action,
                    "true_action": true_action,
                    "action_correct": bool(pred_action == true_action),
                    "action_strength": float(action_strength(pred_return, theta)),
                    "boundary_gap": float(boundary_gap(pred_return, theta)),
                    "abs_error": float(abs(pred_return - true_return)),
                    "features": feature_snapshot,
                })

            row_index += batch_size

    return records


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ============================================================
    # Build switches
    # IMPORTANT:
    # If your current database was last built in classification mode,
    # keep these True once to rebuild regression targets.
    # After one successful run, change them back to False.
    # ============================================================
    REBUILD_RAW = False
    REBUILD_PROCESSED = False
    REBUILD_REPRESENTED = False

    # ============================================================
    # Task settings
    # ============================================================
    TASK_TYPE = "regression"
    RETRAIN = False
    THETA = 0.025

    # ============================================================
    # Gemini reporting settings
    # ============================================================
    USE_LLM_REPORTING = True
    LLM_MODEL_NAME = "gemini-2.5-flash"

    REPORT_FEATURES = [
        "SP500",
        "SP500_logret",
        "VIXCLS",
        "T10Y2Y",
        "GS10",
        "UNRATE",
        "UMCSENT",
    ]

    # ============================================================
    # Environment keys
    # ============================================================
    os.environ["GEMINI_API_KEY"] = "AIzaSyApFXUZsiDvpJXlXSh0d3Af3PrVpHxKpiQ"

    FRED_API_KEY = os.getenv("FRED_API_KEY", None)
    GEMINI_API_KEY_EXISTS = bool(
        os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    )
    # ============================================================
    # Paths
    # ============================================================
    RAW_DB = "data_raw.db"
    PROCESSED_DB = "data_processed.db"
    REPRESENTED_DB = "data_represented.db"
    MODEL_PATH = "models/model_best_regression_llm.pth"

    os.makedirs("models", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    # ============================================================
    # Date range
    # ============================================================
    START_DATE = "2015-01-01"
    END_DATE = "2025-12-31"

    # ============================================================
    # Required FRED series
    # ============================================================
    data_id = [
        "SP500",
        "DJIA",
        "NASDAQCOM",
        "VIXCLS",
        "TEDRATE",
        "STLFSI4",
        "BAMLH0A0HYM2",
        "BAMLC0A0CM",
        "BAMLC0A1CAAA",
        "BAMLC0A4CBBB",
        "GS3M",
        "GS2",
        "GS10",
        "GS30",
        "T10Y2Y",
        "T10Y3M",
        "CPIAUCSL",
        "CPILFESL",
        "PCEPI",
        "M2SL",
        "WALCL",
        "UNRATE",
        "PAYEMS",
        "INDPRO",
        "TCU",
        "DGORDER",
        "CSUSHPISA",
        "HOUST",
        "PERMIT",
        "UMCSENT",
        "RSAFS"
    ]

    # ============================================================
    # 1) Acquire raw data
    # ============================================================
    raw_db_path = os.path.join("data_fetched", RAW_DB)

    if (not os.path.exists(raw_db_path)) or REBUILD_RAW:
        remaining_ids = dtac.FilterExistingSeriesIds(data_id, RAW_DB)

        if len(remaining_ids) > 0:
            if not FRED_API_KEY:
                raise ValueError(
                    "FRED_API_KEY is not set, but missing raw series need to be fetched."
                )

            dataframes = dtac.FetchFredSeriesList(
                series_ids=remaining_ids,
                start_date=START_DATE,
                end_date=END_DATE,
                api_key=FRED_API_KEY
            )

            if len(dataframes) > 0:
                dtac.StoreDataframesToSqlite(
                    dtac.CreateDictName(dataframes),
                    RAW_DB
                )
        else:
            print("No new raw series to fetch. Using existing local raw database.")

    dict_df_raw = dtpr.LoadSqliteTablesAsDataframes(RAW_DB)

    if len(dict_df_raw) == 0:
        raise FileNotFoundError("No raw data found in data_fetched/data_raw.db")

    # ============================================================
    # 2) Processed dataset
    # ============================================================
    processed_db_path = os.path.join("data_fetched", PROCESSED_DB)

    if (not os.path.exists(processed_db_path)) or REBUILD_PROCESSED:
        dict_df_retype = dtpr.CleanAndRekeyDataframes(dict_df_raw)

        for k, df in dict_df_retype.items():
            print(k + str(df.shape))

        df_processing = dtpr.MergeSeriesDataframes(dict_df_retype)
        df_processing = dtpr.ForwardFillMergedDataframe(df_processing)

        price_cols = ["SP500", "DJIA", "NASDAQCOM"]
        df_processing = dtpr.AddLogReturnForColumns(df_processing, price_cols)

        df_processing = df_processing.iloc[2:].reset_index(drop=True)
        df_processing = dtpr.DropAbnormalRows(df_processing)

        feature_cols = [c for c in df_processing.columns if c not in ["date"]]

        df_processing = dtpr.BuildStateDataset(
            df_processing,
            feature_cols=feature_cols,
            target_col="SP500_logret",
            rolling_mean_cols=["SP500", "SP500_logret"],
            lookback=30,
            horizon=20,
            task=TASK_TYPE,
            theta=THETA
        )

        df_processing = dtpr.DropAbnormalRows(df_processing)

        dict_df_processed = dtpr.SplitDatasetByTime(df_processing)

        dict_df_processed = dtpr.GaussianNormalizeByTarget(
            dict_df_processed,
            "train",
            ["date", "target"]
        )

        dict_df_processed = dtac.SanitizeColumnNamesDict(dict_df_processed)

        dtac.StoreDataframesToSqlite(
            dict_df_processed,
            PROCESSED_DB,
            "replace"
        )

        print("Processed dataset rebuilt.")
    else:
        print("Using existing processed dataset.")

    dict_df_processed = dtpr.LoadSqliteTablesAsDataframes(PROCESSED_DB)

    if len(dict_df_processed) == 0:
        raise FileNotFoundError("Processed dataset not found.")

    # ============================================================
    # 3) PCA representation
    # ============================================================
    ISPICK = True
    PCA_RANGE = 0.1
    PCA_PICK = 10
    exclude_col = ["date", "target"]

    represented_db_path = os.path.join("data_fetched", REPRESENTED_DB)

    if (not os.path.exists(represented_db_path)) or REBUILD_REPRESENTED:
        dict_df_representing = dict_df_processed.copy()

        df_train = dict_df_representing["train"]
        evalue, evector = dtre.ComputePCAFromDataframe(df_train, exclude_col)

        num = int((evalue > PCA_RANGE).sum())
        print(num)

        for k, df in dict_df_representing.items():
            dict_df_representing[k] = dtre.ProjectDataframeByEigenvectors(
                df,
                exclude_col,
                evector[0:num]
            )

        x_cols = []
        if ISPICK:
            for i in range(PCA_PICK):
                x_cols.append("pca_" + str(i + 1))
        else:
            for i in range(num):
                x_cols.append("pca_" + str(i + 1))

        dtba.PlotPCAFeaturesVsTarget(
            dict_df_representing["train"],
            x_cols,
            "target",
            "img.png"
        )

        dtac.StoreDataframesToSqlite(
            dict_df_representing,
            REPRESENTED_DB,
            "replace"
        )
    else:
        print("Using existing represented dataset.")
        dict_df_representing = dtpr.LoadSqliteTablesAsDataframes(REPRESENTED_DB)

        all_pca_cols = [
            c for c in dict_df_representing["train"].columns
            if c.startswith("pca_")
        ]

        x_cols = []
        if ISPICK:
            x_cols = all_pca_cols[:PCA_PICK]
        else:
            x_cols = all_pca_cols

    print(dict_df_representing)

    # ============================================================
    # 4) Hyperparameters
    # ============================================================
    NUM_EPOCHS = 300
    BATCH_SIZE = 64
    SEQ_LEN = 20
    LEARNING_RATE = 1e-3

    # ============================================================
    # 5) Dataloaders
    # ============================================================
    train_loader, val_loader, test_loader = mdtr.PrepareDataloaders(
        dict_df_representing,
        x_cols,
        SEQ_LEN,
        BATCH_SIZE,
        task_type=TASK_TYPE
    )

    # ============================================================
    # 6) Model
    # ============================================================
    model = mdtr.TradingNet(
        input_dim=len(x_cols),
        task_type=TASK_TYPE
    ).to(device)

    # ============================================================
    # 7) Loss / Optimizer
    # ============================================================
    criterion = nn.SmoothL1Loss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    # ============================================================
    # 8) Training
    # ============================================================
    train_losses_list = []
    val_losses_list = []
    val_accs_list = []

    print(torch.cuda.is_available())
    print(next(model.parameters()).device)

    if RETRAIN:
        best_acc = -1.0
        best_val_loss = float("inf")

        for epoch in range(NUM_EPOCHS):
            train_loss = mdtr.TrainOneEpoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device
            )

            val_loss, val_acc = mdtr.Evaluate(
                model,
                val_loader,
                criterion,
                device,
                TASK_TYPE,
                THETA
            )

            train_losses_list.append(train_loss)
            val_losses_list.append(val_loss)
            val_accs_list.append(val_acc)

            print(
                f"Epoch {epoch:03d} | "
                f"train loss: {train_loss:.6f} | "
                f"val loss: {val_loss:.6f} | "
                f"decision acc: {val_acc:.4f}"
            )

            improved = (
                (val_acc > best_acc) or
                (val_acc == best_acc and val_loss < best_val_loss)
            )

            if improved:
                best_acc = val_acc
                best_val_loss = val_loss
                torch.save(model.state_dict(), MODEL_PATH)

        print(f"Best validation decision accuracy: {best_acc:.4f}")

        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
    else:
        print("Loading trained model...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()

    # ============================================================
    # 9) Final evaluation
    # ============================================================
    val_loss, val_acc = mdtr.Evaluate(
        model,
        val_loader,
        criterion,
        device,
        TASK_TYPE,
        THETA
    )

    test_loss, test_acc = mdtr.Evaluate(
        model,
        test_loader,
        criterion,
        device,
        TASK_TYPE,
        THETA
    )

    print(f"Final validation | loss: {val_loss:.6f} | decision acc: {val_acc:.4f}")
    print(f"Final test       | loss: {test_loss:.6f} | decision acc: {test_acc:.4f}")

    # ============================================================
    # 10) Curves
    # ============================================================
    dtba.PlotCurves(
        {
            "train loss": train_losses_list,
        },
        xlabel="Epoch",
        ylabel="Value",
        title="Training curves",
        img_name="Training curves1.png"
    )

    dtba.PlotCurves(
        {
            "val loss": val_losses_list
        },
        xlabel="Epoch",
        ylabel="Value",
        title="Validation curves",
        img_name="Training curves2.png"
    )

    dtba.PlotCurves(
        {
            "val decision acc": val_accs_list,
        },
        xlabel="Epoch",
        ylabel="Value",
        title="Validation decision accuracy",
        img_name="Training curves3.png"
    )

    # ============================================================
    # 11) Confusion matrix on TEST
    # ============================================================
    y_true, y_pred = mdtr.GetPredictions(model, test_loader, device)
    y_true = mdtr.ReturnToAction(y_true, THETA)
    y_pred = mdtr.ReturnToAction(y_pred, THETA)

    dtba.PlotConfusionMatrix(y_true, y_pred)

       # ============================================================
    # 12) Gemini representative quarterly reports + quantitative tables
    # ============================================================
    if USE_LLM_REPORTING and GEMINI_API_KEY_EXISTS:
        try:
            test_records = collect_test_regression_records(
                model=model,
                dataloader=test_loader,
                source_df=dict_df_processed["test"],
                seq_len=SEQ_LEN,
                device=device,
                selected_features=REPORT_FEATURES,
                theta=THETA,
            )

            quarterly_reports = llmx.generate_quarterly_regression_reports(
                test_records,
                theta=THETA,
                model_name=LLM_MODEL_NAME,
            )

            json_path = os.path.join("figures", "gemini_regression_quarterly_reports.json")
            summary_csv_path = os.path.join("figures", "regression_quarter_summary_table.csv")
            case_csv_path = os.path.join("figures", "regression_representative_case_table.csv")

            llmx.save_reports_to_json(quarterly_reports, json_path)
            llmx.save_quantitative_tables(
                quarterly_reports,
                summary_csv_path=summary_csv_path,
                case_csv_path=case_csv_path
            )

            print(f"Saved representative quarterly reports to: {json_path}")
            print(f"Saved quantitative quarter summary table to: {summary_csv_path}")
            print(f"Saved representative case table to: {case_csv_path}")

            for report in quarterly_reports:
                print("=" * 100)
                print("Quarter:", report["quarter"])
                print("Stats:", report["stats"])
                print("Representative cases:")
                for case in report["representative_cases"]:
                    print(
                        f"  - {case['role']} | {case['date']} | "
                        f"pred_return={case['pred_return']:.4f} | "
                        f"true_return={case['true_return']:.4f} | "
                        f"pred_action={case['pred_action']} | "
                        f"true_action={case['true_action']} | "
                        f"correct={case['action_correct']}"
                    )
                print("-" * 100)
                print(report["report"])

        except Exception as e:
            print("Gemini regression reporting step failed:", str(e))
    else:
        print("Gemini reporting skipped (GEMINI_API_KEY / GOOGLE_API_KEY not set).")

if __name__ == "__main__":
    set_seed(1114)
    main()