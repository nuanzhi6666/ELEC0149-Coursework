# ELEC0136: Machine Learning System Design

## Summative assessment

### Brief

You are the junior data scientist on “Money Money Money!”. Your product manager, David, asks you to design, justify, and reproduce an AI-based decision system that informs a policy (Buy / Hold / Sell) to automatically perform trading actions (Buy / Hold / Sell) on the S&P 500, a stock market index that tracks the stock performance of 500 of the largest companies listed in the US stock exchanges. The aim of the decision system is to increase the company’s revenue.

David would like you to report back with a documented data pipeline: data discovery, acquisition, validated, storage, processing, feature analysis, training and evaluation protocols, and an auditable trail of choices and trade-offs that allows them to interpret the system’s decisions.

### Constraints

- No paid services. Use only free/public data & infrastructure.
- Do not use external database services. Please spawn your own database on a local machine. If, for any reason, you want to use an external service (e.g., Mongo Atlas), make sure the database is running and accessible for 2 months after submission.
- No feedback is allowed before submission.
- **AI Usage Disclosure.** We adopt the UCL Category 2 level of use of GenAI: <https://www.ucl.ac.uk/teaching-learning/generative-ai-hub/three-categories-genai-use-assessment>. Reports and codes that do not employ GenAI will be rewarded. Reports and codes that use GenAI, but that do not declare its use will be penalised.

---

## Deliverables

### Report

- Must be max 8 pages, excluded references and an optional appendix.
- Must TMRL template attached. Overleaf is strongly advised.
- Must be submitted in PDF format. Reports that do not respect template, or submitted in different formats will be desk-rejected and have a score of 0 (zero).
- Please include your student number and the repo URL in the report. Do not include your name.

### Code

- Produces the evidence that you presented in your report.
- Plain Python: no notebooks, no Makefiles.
- Single entry point: a `main.py` file, in the root of the repository, **must** run all the data pipeline: collecting the data, validating it, storing it, analysing it, saving plots on disk, running the training, and evaluating the model.
- Deterministic: fixed seeds; no interactive input; plots saved to disk.
- Environment: `environment.yml` in the root of the repo are mandatory and must contain all the packages required to run your code.
- **Autograding:** we run an automated routine to score your code. The routine will install your `environment.yml` file, and run the `main.py` file in the root of your repository.
- No manual intervention: we will not perform any manual check to score your code. Any instruction in the readme file or sent by email will be ignored.

---

## How to submit

### Report

Please use the “Submission point” on Moodle.

### Code

Please merge the “Feedback” pull request, as discussed in class.

---

## Marking scheme

### REPORT — 70%

| Task                                                      | Score |
| --------------------------------------------------------- | ----- |
| T1 – Introduction and Conclusions                         | 7%    |
| T2 – System overview, architecture, and data flow         | 7%    |
| T3 – Data discovery and acquisition                       | 8%    |
| T4 – Object model, data validation and storage            | 8%    |
| T5 – Dataset construction and sampling                    | 8%    |
| T6 – Problem formalisation                                | 8%    |
| T7 – Representation learning, feature design and feature impact analysis | 8% |
| T8 – Training and evaluation protocol                     | 8%    |
| Writing quality (grammar, clarity, concision)             | 4%    |
| Figures (readability, appropriateness)                    | 4%    |

### CODE — 30%

| Task                         | Score |
| ---------------------------- | ----- |
| Reproducibility              | 10%   |
| Code quality and documentation | 10% |
| Code organisation            | 5%    |
| Git and GitHub usage         | 5%    |

Performance is secondary. Clear reasoning, leak-free design, reproducibility, and defensible trade-offs score highest.

---

## Task details

### T1. Problem OVERVIEW

Describe your problem informally: Why is this a problem? Why your approach? What’s your objective? What would your solution unlock?

### T2. System overview & data flow

Describe the system components and how data flows end-to-end: acquisition, storage, validation, feature build, feature analysis, model training, evaluation, model selection. Do you plan to automatically ingest new data at regular intervals to continuously train the model? Identify the interfaces between systems, inputs/outputs, and failure points. Provide a schematic showing components, artifacts, and control flow (e.g., a simple block diagram), describing what components you used and how they relate to each other.

### T3. Data discovery & ACQUISITION

Acquire S&P500 prices, and additional external data that can be useful to solve your problem, each with provenance and a hypothesis for inclusion.

You can (not must) use data between 1st January 2020 and 31st December 2025 max. Define the time span of the acquisition. Did you acquire 1 year of data? 2 years? Why? Describe the method used to acquire the dataset, and provide details of the interface. Did you use an API? Which one? How did you interact with it?

Optimise data acquisition, for example, if data is already stored in the database, do not use the API to re-download it, but retrieve it from the database.

Provide a description of each set of data that you acquired.

### T4. Object model, DATA VALIDATION and STORAGE

Define an object model and choose a storage format. Why this object model? Why not another? Why this format? Why not another? Justify your choice.

Validate your data. Do all elements comply with your object model? Are there missing values? Duplicates? Outliers? Did you filter/alter your data? If so, why? Explore your data and justify any action on it.

Define a storage strategy. What is the most suitable storage strategy? Why? What database did you use? Why?

> **Infrastructure note:** You can spawn your own server locally, and use it for marking. If you do use a remote server, you must ensure it is running for at least 8 weeks after submission, or it will affect your marking.

### T5. Dataset assembly AND SAMPLING

Design your dataset. What does a data point look like? Why? How many do you have?

Describe the sampling strategy that you will use to train the model and justify it.

### T6. PROBLEM FORMALISATION

Formalise your ML problem and present it mathematically. What is your learning objective? How will you solve it? Why?

### T7. Representation learning & feature impact

Analyse how the representation drives predictions. Options include:

- **Learned representation:** Is there any pattern in the representation learned by the model?
- **Feature analysis:** How are features related to each other? Are there correlated, anticorrelated or orthogonal features? Knowing this, how can you improve the model?
- **Feature impact:** What’s the importance of each feature in the prediction? Run ablation studies.

This is a good place to provide evidence of your claims using figures.

### T8. TRAINING AND EVALUATION

Train a Deep Neural Network with the following architecture:

- `Linear(128) → LSTM(128) → LSTM(128) → Linear(128)`

where the number in parentheses indicates the number of hidden units. To this model, append an additional layer to represent the policy that you want to describe.

Use a loss function appropriate to your problem, and ensure your results are statistically significant (seeds). Training must be cut off after 15 minutes.

Implement at least one baseline, used as a control measure to understand how well the model is doing, for example, naïve last value, or a linear model.
