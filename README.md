# Financial Decision Support with LSTM and LLM Reporting

This project builds a regression-based financial decision support pipeline for the S&P 500 using macro-financial time-series data from FRED.  
The system predicts future cumulative returns with an LSTM model, maps predicted returns into **Buy / Hold / Sell** actions through a fixed threshold rule, and optionally uses **Gemini** to generate quarterly natural-language reports for human-readable interpretation.

## Project Overview

The pipeline is designed for **decision support**, not fully automated trading.

It contains three main stages:

1. **Data acquisition and preprocessing**
   - Retrieve financial and macroeconomic time series from FRED
   - Merge all series by date
   - Forward-fill missing values
   - Add log-return features for major equity indices

2. **Prediction and decision mapping**
   - Build a regression target based on future cumulative S&P 500 return
   - Apply PCA for feature representation
   - Train an LSTM model on sequential inputs
   - Convert predicted returns into **Buy / Hold / Sell** actions using a fixed threshold

3. **LLM-based reporting**
   - Collect structured test-period prediction records
   - Summarise quarterly statistics and representative cases
   - Use Gemini to generate readable quarterly reports for interpretation

## Main Settings

The current default configuration in `main.py` includes:

- **Task type:** regression
- **Decision threshold:** `0.025`
- **Date range:** `2015-01-01` to `2025-12-31`
- **Lookback window for state construction:** `30`
- **Prediction horizon:** `20`
- **Sequence length for LSTM:** `20`
- **Batch size:** `64`
- **Epochs:** `300`
- **Learning rate:** `1e-3`
- **Loss function:** `SmoothL1Loss`
- **LLM model:** `gemini-2.5-flash`

## Project Structure

A typical project structure is as follows:

```text
.
├── main.py
├── README.md
├── requirements.txt
├── utils/
│   ├── my_data_acqusition.py
│   ├── my_data_processing.py
│   ├── my_data_representing.py
│   ├── my_model_train.py
│   ├── my_base_func.py
│   └── my_llm_explainer.py
├── data_fetched/
│   ├── data_raw.db
│   ├── data_processed.db
│   └── data_represented.db
├── models/
│   └── model_best_regression_llm.pth
└── figures/
    ├── img.png
    ├── Training curves1.png
    ├── Training curves2.png
    ├── Training curves3.png
    ├── gemini_regression_quarterly_reports.json
    ├── regression_quarter_summary_table.csv
    └── regression_representative_case_table.csv
```

## Requirements

Install the required packages first:

```bash
pip install -r requirements.txt
```

A minimal `requirements.txt` may include:

```txt
numpy
pandas
matplotlib
scikit-learn
torch
google-genai
```

If your local utility modules use additional packages for FRED access, add them as needed.

## Environment Variables

This project may require API keys depending on whether data or LLM reports need to be generated.

### FRED API
If raw FRED data needs to be fetched, set:

```bash
FRED_API_KEY=your_fred_api_key
```

### Gemini API
If quarterly LLM reports are enabled, set either:

```bash
GEMINI_API_KEY=your_gemini_api_key
```

or

```bash
GOOGLE_API_KEY=your_google_api_key
```

If no Gemini key is provided, the model training and evaluation pipeline can still run, but the LLM reporting step will be skipped.

## How to Run

Run the full pipeline with:

```bash
python main.py
```

The script will:

1. Load or fetch raw data
2. Build processed and represented datasets
3. Train or load the regression model
4. Evaluate validation and test performance
5. Generate training curves and a confusion matrix
6. Optionally generate Gemini quarterly reports

## Outputs

After running the pipeline, the following outputs are typically produced:

- **Model**
  - `models/model_best_regression_llm.pth`

- **Figures**
  - `figures/img.png`  
    PCA feature scatter plots against the target
  - `figures/Training curves1.png`  
    Training loss curve
  - `figures/Training curves2.png`  
    Validation loss curve
  - `figures/Training curves3.png`  
    Validation decision accuracy curve

- **LLM Reporting Outputs**
  - `figures/gemini_regression_quarterly_reports.json`
  - `figures/regression_quarter_summary_table.csv`
  - `figures/regression_representative_case_table.csv`

## Notes

- The model is trained as a **regression model**, not a direct classifier.
- Buy / Hold / Sell labels are obtained by threshold-mapping the predicted return.
- The LLM does **not** replace the model or change the final prediction in the current implementation.
- The role of the LLM is to improve **readability, interpretability, and human-centred analysis** by turning structured prediction results into quarterly natural-language reports.

## Reproducibility

The script sets a fixed random seed for reproducibility.  
To rerun the pipeline from scratch, you may need to adjust the rebuild switches in `main.py`:

- `REBUILD_RAW`
- `REBUILD_PROCESSED`
- `REBUILD_REPRESENTED`
- `RETRAIN`

## Disclaimer

This project is developed for academic and research purposes only.  
It is not financial advice and should not be used as the sole basis for real investment decisions.