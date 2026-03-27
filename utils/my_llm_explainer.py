import os
import json
from typing import Dict, List, Any

import pandas as pd
from google import genai

GEMINI_API_KEY = "AIzaSyApFXUZsiDvpJXlXSh0d3Af3PrVpHxKpiQ"

ACTION_NAME = {
    -1: "Sell",
    0: "Hold",
    1: "Buy",
}


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _format_feature_snapshot(features: Dict[str, Any]) -> str:
    lines = []
    for k, v in features.items():
        val = _safe_float(v)
        if val is None:
            lines.append(f"- {k}: {v}")
        else:
            lines.append(f"- {k}: {val:.4f}")
    return "\n".join(lines)


def _build_quarter_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)

    action_correct = sum(1 for r in records if r["action_correct"])
    action_acc = action_correct / total if total > 0 else 0.0

    mae = sum(abs(r["pred_return"] - r["true_return"]) for r in records) / total if total > 0 else 0.0
    rmse = (
        sum((r["pred_return"] - r["true_return"]) ** 2 for r in records) / total
    ) ** 0.5 if total > 0 else 0.0

    avg_pred_return = sum(r["pred_return"] for r in records) / total if total > 0 else 0.0
    avg_true_return = sum(r["true_return"] for r in records) / total if total > 0 else 0.0
    avg_action_strength = sum(r["action_strength"] for r in records) / total if total > 0 else 0.0
    avg_abs_error = sum(r["abs_error"] for r in records) / total if total > 0 else 0.0

    predicted_counts = {"Sell": 0, "Hold": 0, "Buy": 0}
    true_counts = {"Sell": 0, "Hold": 0, "Buy": 0}

    for r in records:
        predicted_counts[ACTION_NAME[r["pred_action"]]] += 1
        true_counts[ACTION_NAME[r["true_action"]]] += 1

    return {
        "total_samples": total,
        "action_correct": action_correct,
        "action_accuracy": action_acc,
        "mae": mae,
        "rmse": rmse,
        "avg_abs_error": avg_abs_error,
        "avg_pred_return": avg_pred_return,
        "avg_true_return": avg_true_return,
        "avg_action_strength": avg_action_strength,
        "predicted_counts": predicted_counts,
        "true_counts": true_counts,
    }


def _select_representative_cases(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Select representative cases for one quarter:
    - Typical Buy
    - Typical Hold
    - Typical Sell
    - Boundary Case
    - High-Confidence Mistake
    """
    selected = []
    used_dates = set()

    def add_case(case: Dict[str, Any], role: str):
        if case is None:
            return
        if case["date"] in used_dates:
            return
        copied = dict(case)
        copied["role"] = role
        selected.append(copied)
        used_dates.add(case["date"])

    # Strongest Buy
    buy_cases = [r for r in records if r["pred_action"] == 1]
    if buy_cases:
        buy_cases = sorted(buy_cases, key=lambda r: r["pred_return"], reverse=True)
        add_case(buy_cases[0], "Typical Buy")

    # Most central Hold
    hold_cases = [r for r in records if r["pred_action"] == 0]
    if hold_cases:
        hold_cases = sorted(hold_cases, key=lambda r: abs(r["pred_return"]))
        add_case(hold_cases[0], "Typical Hold")

    # Strongest Sell
    sell_cases = [r for r in records if r["pred_action"] == -1]
    if sell_cases:
        sell_cases = sorted(sell_cases, key=lambda r: r["pred_return"])
        add_case(sell_cases[0], "Typical Sell")

    # Closest to threshold
    if records:
        boundary_cases = sorted(records, key=lambda r: r["boundary_gap"])
        add_case(boundary_cases[0], "Boundary Case")

    # Wrong with strongest conviction
    wrong_cases = [r for r in records if not r["action_correct"]]
    if wrong_cases:
        wrong_cases = sorted(wrong_cases, key=lambda r: r["action_strength"], reverse=True)
        add_case(wrong_cases[0], "High-Confidence Mistake")

    return selected


def _build_case_block(case: Dict[str, Any]) -> str:
    return f"""
Role: {case["role"]}
Date: {case["date"]}
Predicted return: {case["pred_return"]:.6f}
True return: {case["true_return"]:.6f}
Predicted action: {ACTION_NAME[case["pred_action"]]}
True action: {ACTION_NAME[case["true_action"]]}
Action correct: {case["action_correct"]}
Action strength: {case["action_strength"]:.6f}
Distance to decision threshold: {case["boundary_gap"]:.6f}
Absolute prediction error: {case["abs_error"]:.6f}

Selected standardized features (z-scores relative to the training set):
{_format_feature_snapshot(case["features"])}
""".strip()


def _build_quarter_prompt(
    quarter_name: str,
    stats: Dict[str, Any],
    cases: List[Dict[str, Any]],
    theta: float,
) -> str:
    case_text = "\n\n".join(_build_case_block(c) for c in cases)

    prompt = f"""
You are a financial ML reporting assistant.

Your task is to write a detailed, evidence-based quarterly report for a regression-based trading signal model.

Model decision rule:
- Buy if predicted return > {theta:.4f}
- Sell if predicted return < -{theta:.4f}
- Hold otherwise

Important interpretation note:
The feature values are standardized z-scores relative to the training period.
Positive means above the training-period average.
Negative means below the training-period average.

Quarter: {quarter_name}

Quarter-level quantitative statistics:
- Total evaluated samples: {stats["total_samples"]}
- Action accuracy: {stats["action_accuracy"]:.4f}
- MAE: {stats["mae"]:.6f}
- RMSE: {stats["rmse"]:.6f}
- Average absolute error: {stats["avg_abs_error"]:.6f}
- Average predicted return: {stats["avg_pred_return"]:.6f}
- Average true return: {stats["avg_true_return"]:.6f}
- Average action strength: {stats["avg_action_strength"]:.6f}

Predicted action counts:
- Sell: {stats["predicted_counts"]["Sell"]}
- Hold: {stats["predicted_counts"]["Hold"]}
- Buy: {stats["predicted_counts"]["Buy"]}

True action counts:
- Sell: {stats["true_counts"]["Sell"]}
- Hold: {stats["true_counts"]["Hold"]}
- Buy: {stats["true_counts"]["Buy"]}

Representative cases:
{case_text}

Write a detailed report with the following sections:

1. Quarter Overview
- 6 to 8 sentences.
- State whether the model was cautious, neutral, or optimistic.
- Use the action counts and average predicted/true return as evidence.
- Comment on whether the quarter was stable, mixed, or trend-dominant.

2. Quantitative Signal Analysis
- 2 paragraphs.
- Use action accuracy, MAE, RMSE, and action counts as quantitative evidence.
- Explain whether the model overstated or understated directional conviction.
- Compare predicted action mix and true action mix.
- Mention whether the model tended to over-predict Buy, Sell, or Hold.

3. Representative Case Analysis
- Analyze each representative case one by one.
- For each case:
  a) explain why the continuous predicted return led to Buy / Hold / Sell,
  b) compare predicted return against true return,
  c) use at least 3 named features as support when possible.
- If a case is wrong, explain specifically what may have misled the model.

4. Practical Interpretation
- 3 to 4 sentences.
- Explain what this quarter suggests about how the model behaves in real trading-signal use.
- Mention whether the model seems better at trend capture, neutral filtering, or downside detection.

5. Short Limitation Note
- 1 or 2 sentences.
- State clearly that this is a model-based interpretation, not financial advice.

Requirements:
- Be specific and quantitative.
- Do not be generic.
- Use the actual numbers and feature names.
- Explicitly connect predicted return values to final action mapping.
- Keep the tone professional and analytical.
""".strip()

    return prompt


def _generate_text(prompt: str, model_name: str = "gemini-2.5-flash") -> str:
    import time

    client = genai.Client(api_key=GEMINI_API_KEY)

    last_error = None
    wait_seconds = [5, 10, 20, 40]

    for wait in wait_seconds:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            return response.text.strip()
        except Exception as e:
            last_error = e
            err_text = str(e)

            if "503" in err_text or "UNAVAILABLE" in err_text or "429" in err_text:
                print(f"Gemini temporary error: {err_text}")
                print(f"Retrying in {wait} seconds...")
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(f"Gemini failed after retries: {last_error}")


def _pick_key_quarters(quarter_payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Keep only 3 most useful quarters:
    1. best quarter by action accuracy
    2. worst quarter by action accuracy
    3. most representative quarter (closest to average action accuracy)
    """
    if len(quarter_payloads) <= 3:
        return sorted(quarter_payloads, key=lambda q: q["quarter"])

    accs = [q["stats"]["action_accuracy"] for q in quarter_payloads]
    mean_acc = sum(accs) / len(accs)

    best_q = max(quarter_payloads, key=lambda q: q["stats"]["action_accuracy"])
    worst_q = min(quarter_payloads, key=lambda q: q["stats"]["action_accuracy"])

    candidates = [
        q for q in quarter_payloads
        if q["quarter"] not in {best_q["quarter"], worst_q["quarter"]}
    ]

    representative_q = min(
        candidates,
        key=lambda q: abs(q["stats"]["action_accuracy"] - mean_acc)
    )

    selected = [best_q, worst_q, representative_q]
    selected = sorted(selected, key=lambda q: q["quarter"])
    return selected


def _build_summary_table_from_payloads(quarter_payloads: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for q in quarter_payloads:
        s = q["stats"]
        rows.append({
            "quarter": q["quarter"],
            "total_samples": s["total_samples"],
            "action_accuracy": s["action_accuracy"],
            "mae": s["mae"],
            "rmse": s["rmse"],
            "avg_abs_error": s["avg_abs_error"],
            "avg_pred_return": s["avg_pred_return"],
            "avg_true_return": s["avg_true_return"],
            "avg_action_strength": s["avg_action_strength"],
            "pred_sell": s["predicted_counts"]["Sell"],
            "pred_hold": s["predicted_counts"]["Hold"],
            "pred_buy": s["predicted_counts"]["Buy"],
            "true_sell": s["true_counts"]["Sell"],
            "true_hold": s["true_counts"]["Hold"],
            "true_buy": s["true_counts"]["Buy"],
        })
    return pd.DataFrame(rows)


def _build_case_table_from_payloads(selected_payloads: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for q in selected_payloads:
        for case in q["representative_cases"]:
            row = {
                "quarter": q["quarter"],
                "role": case.get("role", ""),
                "date": case["date"],
                "pred_return": case["pred_return"],
                "true_return": case["true_return"],
                "pred_action": ACTION_NAME[case["pred_action"]],
                "true_action": ACTION_NAME[case["true_action"]],
                "action_correct": case["action_correct"],
                "action_strength": case["action_strength"],
                "boundary_gap": case["boundary_gap"],
                "abs_error": case["abs_error"],
            }

            for feature_name, feature_value in case.get("features", {}).items():
                row[feature_name] = feature_value

            rows.append(row)

    return pd.DataFrame(rows)


def generate_quarterly_regression_reports(
    records: List[Dict[str, Any]],
    theta: float,
    model_name: str = "gemini-2.5-flash",
) -> List[Dict[str, Any]]:
    """
    Generate detailed reports only for 3 representative quarters:
    - best
    - worst
    - most representative

    Each returned report also contains:
    - all_quarter_summary_table
    """
    if len(records) == 0:
        return []

    df = pd.DataFrame(records)
    df["date_dt"] = pd.to_datetime(df["date"])
    df["quarter"] = df["date_dt"].dt.to_period("Q").astype(str)

    quarter_payloads = []
    quarter_names = sorted(df["quarter"].unique())

    for q in quarter_names:
        quarter_df = df[df["quarter"] == q].copy()
        quarter_df = quarter_df.drop(columns=["date_dt", "quarter"], errors="ignore")
        quarter_records = quarter_df.to_dict(orient="records")

        stats = _build_quarter_stats(quarter_records)
        cases = _select_representative_cases(quarter_records)

        quarter_payloads.append({
            "quarter": q,
            "theta": theta,
            "stats": stats,
            "representative_cases": cases,
        })

    all_quarter_summary_df = _build_summary_table_from_payloads(quarter_payloads)
    selected_payloads = _pick_key_quarters(quarter_payloads)

    reports = []
    for payload in selected_payloads:
        prompt = _build_quarter_prompt(
            quarter_name=payload["quarter"],
            stats=payload["stats"],
            cases=payload["representative_cases"],
            theta=theta,
        )

        report_text = _generate_text(prompt, model_name=model_name)

        reports.append({
            "quarter": payload["quarter"],
            "theta": payload["theta"],
            "stats": payload["stats"],
            "representative_cases": payload["representative_cases"],
            "report": report_text,
        })

    # attach summary table once for convenience
    if reports:
        summary_records = all_quarter_summary_df.to_dict(orient="records")
        for r in reports:
            r["all_quarter_summary_table"] = summary_records

    return reports


def save_reports_to_json(
    reports: List[Dict[str, Any]],
    save_path: str
) -> None:
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2, default=str)


def save_quantitative_tables(
    reports: List[Dict[str, Any]],
    summary_csv_path: str,
    case_csv_path: str
) -> None:
    """
    Save:
    1. quarter summary table
    2. representative case table
    """
    if len(reports) == 0:
        return

    # all-quarter summary table is attached to each report, so just use the first one
    summary_records = reports[0].get("all_quarter_summary_table", [])
    summary_df = pd.DataFrame(summary_records)

    selected_payloads = []
    for r in reports:
        selected_payloads.append({
            "quarter": r["quarter"],
            "representative_cases": r["representative_cases"],
        })

    case_df = _build_case_table_from_payloads(selected_payloads)

    summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
    case_df.to_csv(case_csv_path, index=False, encoding="utf-8-sig")