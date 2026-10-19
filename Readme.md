# FraudLens

> **Intelligent Transaction Risk Detection, ETL & Monitoring Platform**

FraudLens is an end-to-end financial fraud detection platform built with Python. It combines automated data ingestion, ETL processing, data quality validation, supervised machine learning, REST API development, relational data storage, scheduled pipeline execution, and operational monitoring in a single system.

The platform ingests financial transaction datasets from data.world, transforms raw records into a consistent analytical format, and trains an XGBoost classification model to estimate the likelihood that a transaction is fraudulent.

The trained model is exposed through a FastAPI service that supports real-time and batch predictions. Prediction results, pipeline execution history, scheduled jobs, and operational events are persisted in PostgreSQL, making it possible to trace model outputs, investigate suspicious transactions, monitor pipeline reliability, and evaluate model performance over time.

FraudLens is designed as a portfolio project that demonstrates how data engineering and traditional machine learning can be combined to build a maintainable, production-style application.

**Project status:** Development roadmap and target architecture. Individual endpoints, scheduling functionality, monitoring capabilities, and integrations become available as their implementation milestones are completed.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Project Objectives](#3-project-objectives)
4. [Key Features](#4-key-features)
5. [System Architecture](#5-system-architecture)
6. [Technology Stack](#6-technology-stack)
7. [Data Sources](#7-data-sources)
8. [ETL Pipeline](#8-etl-pipeline)
9. [Machine Learning Workflow](#9-machine-learning-workflow)
10. [API Documentation](#10-api-documentation)
11. [Database Design](#11-database-design)
12. [Scheduling and Pipeline Orchestration](#12-scheduling-and-pipeline-orchestration)
13. [Logging and Monitoring](#13-logging-and-monitoring)
14. [Project Structure](#14-project-structure)
15. [Installation and Setup](#15-installation-and-setup)
16. [Running the Application](#16-running-the-application)
17. [Example API Requests](#17-example-api-requests)
18. [Testing Strategy](#18-testing-strategy)
19. [Security Considerations](#19-security-considerations)
20. [Limitations](#20-limitations)
21. [Future Improvements](#21-future-improvements)
22. [Learning Outcomes](#22-learning-outcomes)
23. [Contributing](#23-contributing)
24. [License](#24-license)

---

## 1. Project Overview

FraudLens provides a complete workflow for processing financial transaction data and generating machine-learning-based fraud risk predictions.

The system is divided into five major components:

| # | Component | Purpose |
|---|-----------|---------|
| 1 | **Data ingestion and ETL** | Retrieves transaction data from data.world, validates its structure, applies transformations, and loads the resulting records into persistent storage. |
| 2 | **Machine learning** | Trains and evaluates an XGBoost classifier using labeled historical transactions. |
| 3 | **Prediction service** | Exposes the trained model through REST endpoints for individual and batch inference. |
| 4 | **Persistence and traceability** | Records transactions, predictions, model versions, ETL runs, scheduled jobs, and operational events. |
| 5 | **Monitoring and reporting** | Provides visibility into pipeline execution, prediction activity, flagged transactions, and model performance. |

```mermaid
flowchart LR
    C1["Data Ingestion<br/>& ETL"] --> C2["Machine Learning"]
    C2 --> C3["Prediction Service"]
    C3 --> C4["Persistence<br/>& Traceability"]
    C4 --> C5["Monitoring<br/>& Reporting"]
    C5 -.->|feedback| C2
```

The initial implementation focuses on batch ETL and traditional supervised machine learning. More advanced orchestration, automated retraining, and deployment features can be introduced incrementally.

### Example use case

A transaction is submitted to the FraudLens prediction API.

1. The service validates the supplied features and passes them to the currently deployed XGBoost model.
2. The model returns a fraud probability, which is compared with a configurable decision threshold.
3. The service then returns the prediction and, when persistence is enabled, records the result in the database.

For example:

```json
{
  "transaction_id": "TXN-1042",
  "fraud_probability": 0.87,
  "predicted_label": 1,
  "risk_level": "high",
  "model_version": "xgb-v1",
  "review_status": "pending"
}
```

> *This is an illustrative response, not a real model prediction.*

The result indicates that the transaction has been flagged according to the configured threshold. It does **not** establish that the transaction is definitively fraudulent.

---

## 2. Problem Statement

Financial fraud detection is challenging because transaction datasets are often large, inconsistent, and highly imbalanced. Fraudulent transactions may represent only a small fraction of all observed transactions.

```mermaid
pie title Illustrative class distribution in public fraud datasets
    "Legitimate (0)" : 99.5
    "Fraudulent (1)" : 0.5
```

> A classifier that always answers `0` scores 99.5% accuracy while detecting zero fraud.

A model that classifies every transaction as legitimate can achieve high accuracy while failing to detect any fraud.

In addition, a model alone does not provide a complete operational solution. Financial transaction data must be collected, validated, stored, processed, and made available to downstream applications. Predictions must be traceable, pipeline failures must be visible, and model performance must be evaluated against reliable labels.

**FraudLens addresses these challenges by combining a reproducible data pipeline with a supervised machine learning model and an API-driven application.**

The project investigates the following questions:

- How can transaction data be extracted and processed reliably?
- How can data quality issues be detected before model training?
- How effectively can XGBoost identify fraudulent transactions?
- How should precision, recall, and false-positive rates influence fraud detection decisions?
- How can model predictions be exposed through a REST API?
- How can historical predictions and pipeline runs be tracked?
- How can scheduled ingestion and retraining be managed safely?
- How can the application detect failures and changes in incoming data?

---

## 3. Project Objectives

The primary objectives are to:

- Build a modular Python ETL pipeline for financial transaction datasets.
- Integrate data.world as an external data source.
- Implement schema validation, duplicate detection, missing-value analysis, and data quality checks.
- Persist processed transactions in PostgreSQL.
- Train an XGBoost binary classification model.
- Evaluate the model using metrics appropriate for imbalanced classification.
- Expose predictions through FastAPI.
- Support single-record and batch prediction requests.
- Persist prediction history and model-version metadata.
- Track ETL runs, scheduled jobs, and application events.
- Provide a foundation for monitoring, dashboarding, and future model retraining.
- Package the application for reproducible local execution and eventual containerized deployment.

```mermaid
mindmap
  root((FraudLens<br/>Objectives))
    Data Engineering
      Modular ETL
      data.world integration
      Data quality checks
      PostgreSQL persistence
    Machine Learning
      XGBoost classifier
      Imbalance handling
      Precision-recall metrics
      Threshold tuning
    Backend
      FastAPI endpoints
      Request validation
      Prediction history
      Model versioning
    Operations
      Scheduling
      Structured logging
      Run tracking
      Containerization
```

---

## 4. Key Features

### 4.1 Data ingestion

- Authenticated access to data.world.
- Configurable dataset identifiers.
- Local storage of raw source files.
- Dataset metadata and ingestion timestamps.
- Extraction status and error logging.

### 4.2 ETL and data quality

- Schema and required-column validation.
- Duplicate record detection.
- Missing-value profiling.
- Fraud-label validation.
- Type conversion and feature preparation.
- Processed Parquet output.
- PostgreSQL loading.
- ETL execution history and row-count reconciliation.

### 4.3 Machine learning

- XGBoost binary classification.
- Training, validation, and test dataset separation.
- Class imbalance handling.
- Precision, recall, F1-score, average precision, and confusion matrix reporting.
- Decision threshold configuration.
- Model artifact persistence.
- Model version tracking.
- Reproducible evaluation runs.

### 4.4 Prediction API

- Single-transaction inference.
- Batch transaction inference.
- Fraud probability and classification responses.
- Request validation.
- Model metadata endpoint.
- Health checks.
- Prediction persistence.

### 4.5 Operational tracking

- Historical prediction records.
- ETL execution history.
- Scheduled job definitions.
- Scheduled execution history.
- Application and pipeline logs.
- Error tracking.
- Pipeline health monitoring.

### 4.6 Dashboard and reporting

The planned dashboard can provide:

- Total transactions processed.
- Transactions flagged for review.
- Prediction counts by time period.
- Historical model performance.
- ETL execution status.
- Failed scheduled jobs.
- Prediction review status.

---

## 5. System Architecture

The platform follows a modular architecture that separates data processing, model training, inference, persistence, and monitoring.

```mermaid
flowchart TD
    A["data.world Dataset"] --> B["Extraction Service"]
    B --> C["Raw Data Storage"]
    C --> D["Validation and Transformation"]
    D --> E["Processed Parquet"]
    D --> F[("PostgreSQL Transactions")]
    E --> G["Model Training Pipeline"]
    G --> H["Evaluation and Model Registry"]
    H --> I["Versioned XGBoost Artifact"]
    I --> J["FastAPI Prediction Service"]
    K["Client or Dashboard"] --> J
    J --> L["Prediction Service"]
    L --> M[("Prediction History in PostgreSQL")]
    N["Scheduler"] --> B
    N --> G
    N --> O[("Scheduled Job History")]
    B --> P["Application Logging"]
    D --> P
    G --> P
    J --> P
    P --> Q["Monitoring and Reporting"]
    M --> Q
    O --> Q
```

### Component responsibilities

| Component | Responsibility |
|-----------|----------------|
| data.world | External source of historical transaction datasets |
| Extraction service | Retrieves dataset files and source metadata |
| Transformation service | Cleans and validates records |
| Raw storage | Preserves the downloaded source for reproducibility |
| PostgreSQL | Stores transactions, predictions, job history, and model metadata |
| XGBoost | Estimates the likelihood of fraud |
| Model artifact storage | Stores the trained model and its associated metadata |
| FastAPI | Provides prediction and operational endpoints |
| Scheduler | Triggers configured ETL and training jobs |
| Logging service | Records operational events and failures |
| Dashboard | Displays predictions, pipeline health, and evaluation metrics |

### Design principle

The ETL pipeline, model training process, and prediction API should **run independently**.

The prediction API loads an already-trained model and serves requests without retraining on every call. Scheduled training, when enabled, runs as a separate job and promotes a new model only after the configured evaluation and approval requirements are satisfied.

```mermaid
flowchart LR
    subgraph Train["Training plane (scheduled job)"]
        direction TB
        T1["ETL"] --> T2["Fit XGBoost"] --> T3["Evaluate"] --> T4["Candidate model"]
    end
    subgraph Serve["Serving plane (always on)"]
        direction TB
        S1["Active model artifact"] --> S2["Inference API"] --> S3["Prediction history"]
    end
    T4 -.->|promote after approval| S1
```

---

## 6. Technology Stack

| Technology | Purpose |
|------------|---------|
| Python | Core programming language |
| pandas | Data manipulation and transformation |
| datadotworld | Access to data.world datasets |
| PyArrow / Parquet | Efficient local storage of processed data |
| PostgreSQL | Persistent relational storage |
| SQLAlchemy | Database access and ORM support |
| scikit-learn | Data splitting, preprocessing, and evaluation |
| XGBoost | Fraud classification model |
| joblib | Serialization of model artifacts |
| FastAPI | REST API development |
| Pydantic | API request and response validation |
| Uvicorn | ASGI server |
| Python logging | Application and pipeline logging |
| APScheduler or Prefect | Initial scheduling and workflow execution |
| Streamlit | Planned monitoring dashboard |
| pytest | Automated testing |
| Docker | Reproducible application deployment |
| Git and GitHub | Version control and collaboration |

```mermaid
flowchart TB
    subgraph ING["Ingestion"]
        n1["datadotworld"]
    end
    subgraph PROC["Processing"]
        n2["pandas"]
        n3["PyArrow / Parquet"]
    end
    subgraph STORE["Storage"]
        n4[("PostgreSQL")]
        n5["SQLAlchemy"]
    end
    subgraph ML["Machine Learning"]
        n6["scikit-learn"]
        n7["XGBoost"]
        n8["joblib"]
    end
    subgraph SERV["Serving"]
        n9["FastAPI"]
        n10["Pydantic"]
        n11["Uvicorn"]
    end
    subgraph OPS["Operations"]
        n12["APScheduler / Prefect"]
        n13["Streamlit"]
        n14["pytest"]
        n15["Docker"]
    end
    n1 --> n2 --> n3 --> n4
    n4 --> n6 --> n7 --> n8 --> n9
    n5 --> n4
    n10 --> n9
    n11 --> n9
    n12 -.-> n2
    n12 -.-> n6
    n13 --> n4
    n14 -.-> n2
    n14 -.-> n9
    n15 -.-> n9
```

The initial version does not require distributed computing or deep learning frameworks. Additional tools can be introduced when the system's needs justify them.

---

## 7. Data Sources

**Primary source: [data.world](https://data.world)**

FraudLens is designed to ingest publicly accessible financial transaction datasets hosted on data.world.

**Dataset discovery:**

- Credit card fraud datasets
- Financial transaction fraud datasets
- Transaction fraud detection datasets

> These are search terms, not fixed dataset links. The final dataset must be selected and its availability, license, schema, and download method verified.

### Expected data characteristics

The selected dataset should contain:

- A transaction-level record.
- A fraud target label or an equivalent classification label.
- Numeric or transformable transaction features.
- Sufficient examples of both fraudulent and legitimate transactions.
- Documentation describing the features and labels.

An example of a suitable target is:

| Value | Meaning |
|-------|---------|
| `0` | Legitimate transaction |
| `1` | Fraudulent transaction |

The actual target column may be named `Class`, `is_fraud`, or something else. **The ingestion and training configuration must use the real schema.**

### Example transaction features

Depending on the dataset, features may include:

- Transaction amount.
- Transaction timestamp.
- Transaction type.
- Sender or recipient attributes.
- Historical transaction counts.
- Anonymized numerical features.

Not every dataset contains all these fields. Feature engineering must be adapted to the selected source.

### Data licensing and privacy

The project must comply with the source dataset's license and usage restrictions. Sensitive payment credentials, full card numbers, and CVVs must not be collected or stored.

---

## 8. ETL Pipeline

The ETL pipeline is responsible for converting source datasets into consistent, validated, and reusable data.

```mermaid
flowchart LR
    A["Extract<br/>data.world download"] --> B["Validate<br/>schema, labels, dtypes"]
    B --> C["Transform<br/>clean, dedupe, features"]
    C --> D["Save processed<br/>Parquet"]
    D --> E["Load<br/>PostgreSQL staging + merge"]
    E --> F["Record etl_run<br/>row counts and status"]
    B -.->|schema failure| X["Reject run and log"]
    C -.->|quality rejection| X
```

### 8.1 Extract

The extraction component retrieves the configured dataset from data.world.

**Responsibilities:**

- Authenticate using a data.world API token.
- Retrieve the configured dataset.
- Download the required data file.
- Save a raw copy.
- Record the source identifier and extraction timestamp.
- Record the execution result.
- Handle download and authentication errors.

**Example raw output:**

```text
data/raw/transactions.csv
```

Raw data should be treated as immutable input wherever practical. Replacing raw files without preserving source versions makes experiments difficult to reproduce.

### 8.2 Transform

The transformation component performs data cleaning and validation.

**Expected operations include:**

- Normalize column names.
- Validate required columns.
- Check fraud-label values.
- Measure missing values.
- Detect duplicate records.
- Convert columns to appropriate types.
- Investigate invalid values.
- Apply appropriate feature transformations.
- Record quality-check results.

The pipeline should not silently remove every unusual record. Unusual amounts, rare patterns, and extreme values may contain important fraud signals.

Missing-value imputation and other learned transformations should be fitted on **training data only** when used for model training.

### 8.3 Load

Validated records are stored in PostgreSQL, with Parquet used as an efficient intermediate or analytical format.

**Responsibilities:**

- Load records in manageable batches.
- Avoid duplicate inserts when rerunning the pipeline.
- Record inserted, updated, rejected, and skipped row counts.
- Maintain a traceable ETL execution history.
- Support recovery from failed runs.

An initial development implementation may replace a destination table, but the recurring pipeline should use a safe staging and merge strategy.

```mermaid
flowchart TD
    A["Validated records"] --> B["Stage into temporary table"]
    B --> C["Deduplicate against destination key"]
    C --> D["Merge: insert new, update changed"]
    D --> E["Drop staging table"]
    E --> F["Update etl_run row counts"]
    C -.->|unique key missing| G["Reject and log for review"]
```

### 8.4 ETL execution tracking

Every ETL execution should produce an `etl_run` record.

**Suggested attributes:**

| Attribute | Description |
|-----------|-------------|
| `id` | Unique execution identifier |
| `dataset_key` | Source dataset identifier |
| `started_at` | Execution start time |
| `finished_at` | Execution completion time |
| `status` | `running`, `succeeded`, `failed`, or `partial` |
| `rows_extracted` | Number of source rows retrieved |
| `rows_processed` | Number of records processed |
| `rows_loaded` | Number of records loaded |
| `rows_rejected` | Number of rejected records |
| `error_message` | Error summary when applicable |

This makes it possible to investigate failed runs and determine whether the expected number of records reached the destination.

---

## 9. Machine Learning Workflow

FraudLens uses XGBoost as its primary supervised classification model.

### 9.1 Model objective

Given a transaction's input features, the model estimates a score representing the likelihood of the transaction belonging to the fraudulent class.

The output is converted into a binary prediction using a configurable decision threshold.

```mermaid
flowchart TD
    A["Transaction features"] --> B["Trained XGBoost model"]
    B --> C["Fraud probability"]
    C --> D{"Compare against decision threshold"}
    D -->|at or above| E["Predicted class: 1 (flagged)"]
    D -->|below| F["Predicted class: 0 (clear)"]
```

### 9.2 Training process

The training workflow consists of:

```mermaid
flowchart TD
    A["Load validated processed dataset"] --> B["Separate features and target"]
    B --> C["Split train / validation / test"]
    C --> D["Fit preprocessing on training data only"]
    D --> E["Train XGBoost classifier"]
    E --> F["Evaluate on validation set"]
    F --> G["Select decision threshold"]
    G --> H["Evaluate on held-out test set"]
    H --> I["Save model artifact and metadata"]
```

1. Load the validated processed dataset.
2. Separate features from the target label.
3. Split the data into training, validation, and test sets.
4. Fit preprocessing operations using training data only.
5. Train the XGBoost classifier.
6. Evaluate the model on validation data.
7. Select the decision threshold.
8. Evaluate the selected configuration on the held-out test set.
9. Save the model artifact and metadata.

When transaction timestamps are available and meaningful, **chronological splitting is preferred** to better reflect real deployment conditions.

### 9.3 Class imbalance

Fraudulent transactions often represent a small fraction of the dataset.

FraudLens will investigate class weighting and other appropriate techniques to address imbalance. Class weighting should be **evaluated against an unweighted baseline** rather than assumed to improve results.

### 9.4 Evaluation metrics

| Metric | Purpose |
|--------|---------|
| Precision | Measures the proportion of flagged transactions that are fraudulent |
| Recall | Measures the proportion of actual fraud cases detected |
| F1-score | Summarizes precision and recall |
| Average precision / PR-AUC | Evaluates ranking performance for the minority class |
| ROC-AUC | Measures class-ranking discrimination |
| Confusion matrix | Shows true positives, false positives, true negatives, and false negatives |

Accuracy is reported **only as a supplementary metric** because it can be misleading when fraud is rare.

```mermaid
flowchart LR
    subgraph CONF["Confusion matrix"]
        direction TB
        TP["True Positive<br/>fraud caught"]
        FN["False Negative<br/>fraud missed"]
        FP["False Positive<br/>false alarm"]
        TN["True Negative<br/>legitimate cleared"]
    end
    TP --> PREC["Precision = TP / (TP + FP)"]
    FN --> REC["Recall = TP / (TP + FN)"]
    CONF --> NOTE["Both errors have a cost.<br/>The threshold decides the trade-off."]
```

### 9.5 Threshold selection

A threshold of `0.5` may be a starting point, but it should not automatically be treated as the optimal fraud decision threshold.

The threshold should be selected on validation data based on the desired trade-off between missed fraud and false alarms.

```mermaid
flowchart TD
    T["Decision threshold"] --> L["Lower threshold"]
    T --> H["Higher threshold"]
    L --> L1["More transactions flagged"]
    L --> L2["Higher recall"]
    L --> L3["More false positives"]
    H --> H1["Fewer transactions flagged"]
    H --> H2["Lower recall"]
    H --> H3["Fewer false positives"]
```

### 9.6 Model artifact and versioning

Each trained model should be associated with metadata such as:

- Model version.
- Training timestamp.
- Training dataset identifier.
- Feature schema.
- Training configuration.
- Evaluation metrics.
- Selected decision threshold.
- Artifact location.
- Deployment status.

The API must load the intended model version and preserve feature ordering.

A newly trained model should **not** automatically replace the deployed model without evaluation and a controlled promotion process.

```mermaid
stateDiagram-v2
    [*] --> candidate
    candidate --> active: evaluated and approved
    candidate --> rejected: metrics below requirement
    active --> retired: superseded by new version
    retired --> [*]
    rejected --> [*]
```

---

## 10. API Documentation

FraudLens uses FastAPI to expose the prediction model and selected operational information.

**Base URL for local development:**

```text
http://localhost:8000
```

**Interactive documentation will be available at:**

```text
http://localhost:8000/docs
```

```mermaid
flowchart TB
    API["FraudLens API<br/>localhost:8000"]
    API --> H["Health and metadata<br/>GET /health<br/>GET /health/ready<br/>GET /model/info"]
    API --> P["Predictions<br/>POST /predict<br/>POST /predict/batch<br/>GET /predictions<br/>PATCH /predictions/id/review"]
    API --> E["ETL operations<br/>POST /etl/runs<br/>GET /etl/runs<br/>GET /etl/runs/id<br/>GET /etl/runs/id/logs"]
    API --> S["Scheduling<br/>/schedules"]
    API --> M["Monitoring<br/>GET /metrics"]
```

The following table defines the planned API contract. Each endpoint becomes available once its corresponding handler and supporting database operations have been implemented.

### 10.1 Health and metadata endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/health` | Check whether the application is responding |
| `GET` | `/health/ready` | Check whether required dependencies and the model are ready |
| `GET` | `/model/info` | Retrieve active model version and metadata |

#### `GET /health`

Example response:

```json
{
  "status": "ok"
}
```

This endpoint indicates that the application is responding. It should **not** be treated as proof that PostgreSQL or every external dependency is available.

#### `GET /health/ready`

Example response:

```json
{
  "status": "ready",
  "model_loaded": true,
  "database_connected": true
}
```

This endpoint should verify the dependencies actually required by the service. If a required dependency is unavailable, the API should return an appropriate unsuccessful health status.

#### `GET /model/info`

Example response:

```json
{
  "model_name": "FraudLens XGBoost Classifier",
  "model_version": "xgb-v1",
  "model_type": "XGBClassifier",
  "threshold": 0.5,
  "status": "active"
}
```

The returned values must come from the deployed model metadata rather than hardcoded assumptions.

### 10.2 Prediction endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/predict` | Predict fraud risk for a single transaction |
| `POST` | `/predict/batch` | Predict fraud risk for multiple transactions |

#### `POST /predict`

Accepts one transaction and returns its fraud probability and predicted classification.

Example request:

```json
{
  "transaction_id": "TXN-1042",
  "features": {
    "Amount": 850.0,
    "V1": -1.36,
    "V2": 0.72
  }
}
```

> This is a schematic example. The actual feature set must match the selected dataset and trained model.

Example response:

```json
{
  "prediction_id": 1042,
  "transaction_id": "TXN-1042",
  "fraud_probability": 0.87,
  "predicted_label": 1,
  "risk_level": "high",
  "threshold": 0.5,
  "model_version": "xgb-v1",
  "review_status": "pending"
}
```

**The API should:**

1. Validate the request.
2. Verify that all required features are present.
3. Reject unexpected features when strict schema validation is enabled.
4. Apply the same feature ordering and preprocessing used during training.
5. Generate the fraud probability.
6. Apply the configured decision threshold.
7. Persist the result when prediction persistence is enabled.
8. Return the prediction and tracking identifier.

> The sample feature values and probabilities above are illustrative. They do not represent an actual model result.

```mermaid
sequenceDiagram
    participant C as Client
    participant A as FastAPI Router
    participant V as Pydantic Validation
    participant M as XGBoost Model
    participant D as PostgreSQL

    C->>A: POST /predict
    A->>V: Validate payload and feature schema
    V-->>A: Validated, correctly ordered feature vector
    A->>M: predict_proba(features)
    M-->>A: Fraud probability
    A->>A: Apply threshold and derive risk level
    A->>D: Insert prediction record
    D-->>A: prediction_id
    A-->>C: 200 PredictionResponse
```

#### `POST /predict/batch`

Accepts a collection of transactions and returns a prediction for each valid record.

Example request:

```json
{
  "transactions": [
    {
      "transaction_id": "TXN-001",
      "features": {
        "Amount": 125.0
      }
    },
    {
      "transaction_id": "TXN-002",
      "features": {
        "Amount": 2400.0
      }
    }
  ]
}
```

> The example is schematic; both transactions must contain all required model features in a working implementation.

Example response:

```json
{
  "count": 2,
  "model_version": "xgb-v1",
  "predictions": [
    {
      "transaction_id": "TXN-001",
      "fraud_probability": 0.04,
      "predicted_label": 0
    },
    {
      "transaction_id": "TXN-002",
      "fraud_probability": 0.91,
      "predicted_label": 1
    }
  ]
}
```

Batch inference should validate the maximum batch size and use **vectorized model inference** instead of making a separate model call for every record.

For large offline scoring jobs, a separate background batch-processing workflow is preferable to sending extremely large requests through a synchronous HTTP endpoint.

### 10.3 Prediction history endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/predictions` | List historical predictions |
| `GET` | `/predictions/{prediction_id}` | Retrieve a specific prediction |
| `PATCH` | `/predictions/{prediction_id}/review` | Update the review status of a prediction |

#### `GET /predictions`

Supports pagination and optional filters such as:

- Predicted class.
- Minimum fraud probability.
- Review status.
- Model version.
- Date range.

Example:

```text
GET /predictions?predicted_label=1&limit=20&offset=0
```

Example response:

```json
{
  "total": 1,
  "limit": 20,
  "offset": 0,
  "items": [
    {
      "id": 1042,
      "transaction_id": "TXN-1042",
      "fraud_probability": 0.87,
      "predicted_label": 1,
      "model_version": "xgb-v1",
      "review_status": "pending"
    }
  ]
}
```

#### `GET /predictions/{prediction_id}`

Retrieves the full stored prediction record, including its model version and creation timestamp.

#### `PATCH /predictions/{prediction_id}/review`

Allows an authorized reviewer to update a review status.

Example request:

```json
{
  "review_status": "confirmed_fraud",
  "review_note": "Verified by an authorized reviewer"
}
```

**Suggested review statuses:**

| Status | Meaning |
|--------|---------|
| `pending` | Not yet reviewed |
| `confirmed_fraud` | Reviewer confirmed fraudulent |
| `legitimate` | Reviewer confirmed legitimate |
| `inconclusive` | Insufficient information to decide |

Review status must be kept **separate** from the model's predicted label. The model's output is a prediction; a review is a subsequent human or authoritative outcome.

> This endpoint should require authentication and authorization before deployment.

```mermaid
flowchart LR
    M["Model prediction<br/>predicted_label"] --> DB[(predictions)]
    R["Reviewer decision<br/>review_status"] --> DB
    DB --> DASH["Dashboard and reporting"]
    DB --> EVAL["Model performance evaluation"]
```

### 10.4 ETL operational endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/etl/runs` | Request an ETL execution |
| `GET` | `/etl/runs` | List historical ETL runs |
| `GET` | `/etl/runs/{run_id}` | Retrieve the status of an ETL run |

`POST /etl/runs` should create or enqueue an ETL job and return a tracking identifier rather than execute a long-running pipeline inside the HTTP request.

Example response:

```json
{
  "run_id": 205,
  "status": "queued",
  "message": "ETL execution has been scheduled"
}
```

Example status lookup:

```json
{
  "run_id": 205,
  "status": "succeeded",
  "rows_extracted": 100000,
  "rows_processed": 99800,
  "rows_loaded": 99800,
  "rows_rejected": 200
}
```

> These counts are illustrative. In the implemented system, the values should be calculated and stored by the actual ETL process.

### 10.5 Scheduling endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/schedules` | Create a scheduled job |
| `GET` | `/schedules` | List configured schedules |
| `GET` | `/schedules/{schedule_id}` | Retrieve a schedule |
| `PATCH` | `/schedules/{schedule_id}` | Update or enable/disable a schedule |
| `DELETE` | `/schedules/{schedule_id}` | Delete a schedule |
| `GET` | `/schedules/{schedule_id}/runs` | View execution history |

A **schedule** describes when a job should run; a **schedule run** records a particular execution.

Example schedule request:

```json
{
  "name": "daily-fraud-data-ingestion",
  "job_type": "etl",
  "cron_expression": "0 2 * * *",
  "timezone": "UTC",
  "enabled": true
}
```

> This example represents a daily 02:00 UTC schedule. The actual scheduler must parse and enforce the configured schedule.

Example response:

```json
{
  "schedule_id": 12,
  "name": "daily-fraud-data-ingestion",
  "job_type": "etl",
  "cron_expression": "0 2 * * *",
  "timezone": "UTC",
  "enabled": true
}
```

**Scheduled job types may include:**

| Job type | Purpose |
|----------|---------|
| `etl` | Run the extraction and loading pipeline |
| `model_training` | Train a candidate model |
| `batch_prediction` | Score newly ingested transactions |
| `data_quality_check` | Evaluate schema and quality expectations |

The implementation must prevent overlapping executions where they could corrupt data or compete for resources. Production scheduling should also support retries, timeouts, execution locks, and failure notifications.

### 10.6 Logging and monitoring endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/etl/runs/{run_id}/logs` | Retrieve available logs for an ETL run |
| `GET` | `/schedules/{schedule_id}/runs` | Retrieve schedule execution history |
| `GET` | `/metrics` | Expose selected application metrics |

These endpoints should be implemented according to the logging and monitoring architecture.

Logs must be filtered and paginated appropriately. The application should **not** expose API tokens, database credentials, raw payment information, or sensitive internal stack traces through public endpoints.

The `/metrics` endpoint can expose operational metrics such as request counts, response latency, and failure counts. It should **not** be confused with model-quality metrics, which require appropriate evaluation data.

---

## 11. Database Design

PostgreSQL is the primary relational database.

The schema separates source transactions, predictions, model versions, ETL runs, scheduled jobs, execution history, and application events.

### 11.1 Entity relationship overview

```mermaid
erDiagram
    TRANSACTIONS ||--o{ PREDICTIONS : receives
    MODEL_VERSIONS ||--o{ PREDICTIONS : generates
    DATASET_SOURCES ||--o{ ETL_RUNS : produces
    SCHEDULES ||--o{ SCHEDULE_RUNS : triggers
    ETL_RUNS ||--o{ APPLICATION_EVENTS : emits
    SCHEDULE_RUNS ||--o{ APPLICATION_EVENTS : emits
    MODEL_VERSIONS ||--o{ TRAINING_RUNS : records
    TRAINING_RUNS }o--|| DATASET_SOURCES : uses

    TRANSACTIONS {
        bigint id PK
        varchar source_transaction_id
        jsonb features
        smallint actual_label
        timestamptz created_at
    }

    PREDICTIONS {
        bigint id PK
        bigint transaction_id FK
        bigint model_version_id FK
        float fraud_probability
        smallint predicted_label
        varchar review_status
        timestamptz created_at
    }

    MODEL_VERSIONS {
        bigint id PK
        varchar version
        varchar artifact_uri
        float threshold
        jsonb evaluation_metrics
        varchar status
    }

    DATASET_SOURCES {
        bigint id PK
        varchar dataset_key
        varchar source_name
        timestamptz created_at
    }

    ETL_RUNS {
        bigint id PK
        bigint dataset_source_id FK
        varchar status
        integer rows_extracted
        integer rows_loaded
        timestamptz started_at
        timestamptz finished_at
    }

    SCHEDULES {
        bigint id PK
        varchar name
        varchar job_type
        varchar cron_expression
        varchar timezone
        boolean enabled
    }

    SCHEDULE_RUNS {
        bigint id PK
        bigint schedule_id FK
        varchar status
        timestamptz started_at
        timestamptz finished_at
    }

    TRAINING_RUNS {
        bigint id PK
        bigint dataset_source_id FK
        varchar status
        jsonb metrics
        timestamptz started_at
        timestamptz finished_at
    }

    APPLICATION_EVENTS {
        bigint id PK
        varchar severity
        varchar event_type
        text message
        jsonb context
        timestamptz created_at
    }
```

> This is a logical design. The actual database migration scripts may introduce additional tables, foreign keys, indexes, and constraints as implementation progresses.

### 11.2 `transactions`

Stores normalized transaction records.

**Suggested columns:**

| Column | Type | Description |
|--------|------|-------------|
| `id` | `BIGSERIAL` | Internal primary key |
| `source_transaction_id` | `TEXT` | Source identifier, when available |
| `features` | `JSONB` | Dataset-specific feature values |
| `actual_label` | `SMALLINT` | Known label, if available |
| `created_at` | `TIMESTAMPTZ` | Record ingestion time |

The production schema may use explicit columns instead of `JSONB` for frequently queried transaction attributes. A source identifier should have a uniqueness constraint **only** when its uniqueness is guaranteed by the source.

### 11.3 `predictions`

Stores model inference history.

**Suggested columns:**

| Column | Type | Description |
|--------|------|-------------|
| `id` | `BIGSERIAL` | Prediction identifier |
| `transaction_id` | `BIGINT` | Reference to a stored transaction, when applicable |
| `model_version_id` | `BIGINT` | Model used for inference |
| `fraud_probability` | `DOUBLE PRECISION` | Model output |
| `predicted_label` | `SMALLINT` | Threshold-based classification |
| `review_status` | `TEXT` | Current review state |
| `created_at` | `TIMESTAMPTZ` | Prediction timestamp |

A prediction should remain traceable to the model version that generated it.

Repeated predictions for the same transaction may be legitimate, for example when comparing model versions. The system should **not** overwrite historical predictions merely because a new prediction is produced.

### 11.4 `model_versions`

Tracks trained models and deployed artifacts.

**Suggested attributes:**

| Attribute | Description |
|-----------|-------------|
| `version` | Human-readable version identifier |
| `artifact_uri` | Location of the serialized artifact |
| `threshold` | Decision threshold selected at training time |
| `evaluation_metrics` | Stored evaluation report |
| `training_dataset_id` | Dataset used for training |
| `created_at` | Creation timestamp |
| `status` | `candidate`, `active`, `retired`, or `rejected` |

### 11.5 `etl_runs`

Records each ETL execution, including its source, status, row counts, timestamps, and error details.

This table provides an audit trail for pipeline execution and makes failed runs easier to diagnose.

### 11.6 `schedules`

Stores job scheduling definitions.

**Suggested attributes:**

| Attribute | Description |
|-----------|-------------|
| `name` | Job name |
| `job_type` | `etl`, `model_training`, `batch_prediction`, `data_quality_check` |
| `cron_expression` | Cron definition |
| `timezone` | Timezone the cron expression is evaluated in |
| `enabled` | Whether the schedule is active |
| `next_run_at` | Next computed execution time |
| `created_at` | Creation timestamp |
| `updated_at` | Last modification timestamp |

The scheduler remains responsible for interpreting schedule definitions and triggering executions.

### 11.7 `schedule_runs`

Records every execution triggered by a schedule.

**Suggested attributes:**

| Attribute | Description |
|-----------|-------------|
| `schedule_id` | Parent schedule |
| `started_at` | Execution start time |
| `finished_at` | Execution completion time |
| `status` | Execution outcome |
| `attempt_number` | Retry counter |
| `error_message` | Failure detail |
| `result_summary` | Structured outcome summary |

A schedule may have many runs. This distinction is important because a **job definition is not the same as its execution history**.

### 11.8 `training_runs`

Tracks training experiments, evaluation results, and model artifact creation.

This table supports reproducibility and comparison between model versions.

### 11.9 `application_events`

Stores structured operational events for querying and dashboard reporting.

It is **not** a replacement for a complete logging system. The application can write logs to standard output or files while persisting selected important events in PostgreSQL.

### 11.10 Database design principles

The implementation should follow these principles:

- Use primary keys and appropriate foreign keys.
- Store timestamps consistently, preferably as timezone-aware values.
- Add indexes to frequently filtered fields.
- Use migrations to manage schema changes.
- Avoid duplicating large raw datasets in database records unnecessarily.
- Keep transaction labels, model predictions, and review outcomes separate.
- Use explicit transaction boundaries for related database writes.

---

## 12. Scheduling and Pipeline Orchestration

FraudLens includes a planned scheduling component to automate recurring data and machine learning workflows.

For the first implementation, **APScheduler** can be used for relatively simple scheduled jobs. **Prefect** is another option when execution history, retries, workflow dependencies, and operational visibility become more important.

The project should choose **one** scheduler initially rather than trying to operate both simultaneously.

### 12.1 Scheduled jobs

**Examples include:**

| Job | Purpose | Example frequency |
|-----|---------|-------------------|
| Data ingestion | Retrieve the latest available dataset | Daily |
| ETL processing | Validate, transform, and load records | After ingestion |
| Data quality checks | Evaluate schema and quality expectations | After ETL |
| Batch prediction | Score newly ingested transactions | After data preparation |
| Model training | Train a candidate model | Manual or periodic |
| Model evaluation | Compare candidate and active models | After training |

```mermaid
flowchart LR
    A["Data ingestion"] --> B["ETL processing"]
    B --> C["Data quality checks"]
    C --> D["Batch prediction"]
    D --> E["Model training"]
    E --> F["Model evaluation"]
    F -.->|approved| G["Promote model version"]
```

> These frequencies are examples, not guarantees that data.world provides new data on those schedules.

### 12.2 Scheduling workflow

```mermaid
flowchart TD
    A["Scheduler Trigger"] --> B["Create Schedule Run"]
    B --> C["Acquire Execution Lock"]
    C --> D["Execute ETL or ML Job"]
    D --> E{"Execution Successful?"}
    E -->|Yes| F["Record Success and Metrics"]
    E -->|No| G["Record Error and Failure"]
    G --> H{"Retries Remaining?"}
    H -->|Yes| D
    H -->|No| I["Mark Job Failed"]
    F --> J["Release Lock"]
    I --> J
```

### 12.3 Execution requirements

The scheduler should support:

- Configurable execution intervals.
- Timezone-aware scheduling.
- Enable and disable controls.
- Execution history.
- Retries for transient failures.
- Maximum execution duration.
- Prevention of overlapping executions.
- Error logging.
- Manual triggering.
- Safe shutdown and restart behavior.

### 12.4 Scheduling reliability

A schedule definition stored in PostgreSQL does not itself execute a job. **A scheduler process must load those definitions and trigger the appropriate handler.**

```mermaid
flowchart TD
    DB[("schedules table")] --> S["Scheduler process"]
    S --> H["Job handler"]
    H --> DB2[("schedule_runs table")]
    API["FastAPI /schedules"] --> DB
    note["Run the scheduler as a SEPARATE process.<br/>Do not embed it in API workers,<br/>or jobs duplicate across replicas."]
```

For production deployment, the scheduler should run as a separate process or worker. This avoids starting duplicate scheduled jobs whenever multiple API workers are running.

For the initial local implementation, a single scheduler process is sufficient.

---

## 13. Logging and Monitoring

Logging is essential because a successful API response does not guarantee that the data pipeline, database, or model is operating correctly.

FraudLens should use Python's standard `logging` module initially, with structured logging added as the project matures.

```mermaid
flowchart LR
    subgraph SRC["Log producers"]
        E["ETL logs"]
        T["Training logs"]
        A["API logs"]
        S["Scheduler logs"]
    end
    E --> OUT["Python logging<br/>stdout and files"]
    T --> OUT
    A --> OUT
    S --> OUT
    E --> EV[("application_events")]
    T --> EV
    A --> EV
    S --> EV
    EV --> DASH["Monitoring and reporting"]
```

### 13.1 Logging categories

**ETL logs**

- Extraction started and completed.
- Number of rows extracted.
- Validation failures.
- Number of duplicates found.
- Number of records loaded.
- Database errors.
- Execution duration.

**Model training logs**

- Dataset size.
- Fraud class distribution.
- Training configuration.
- Evaluation metrics.
- Artifact save status.
- Training failures.

**API logs**

- Request received.
- Request duration.
- Prediction success or failure.
- Validation errors.
- Database persistence failures.

**Scheduler logs**

- Scheduled execution started.
- Retry attempts.
- Execution completion.
- Missed or overlapping runs.
- Final failure status.

### 13.2 Structured log example

```json
{
  "timestamp": "2026-10-17T08:00:00Z",
  "level": "INFO",
  "component": "etl",
  "event": "etl_completed",
  "run_id": 205,
  "rows_extracted": 100000,
  "rows_loaded": 99800,
  "duration_seconds": 14.7
}
```

> This is an illustrative log event.

Structured fields make it easier to search for all events associated with a particular ETL run or scheduled execution.

### 13.3 Error handling

Expected failures should be handled explicitly.

**Examples include:**

- Missing dataset credentials.
- Source file unavailable.
- Invalid schema.
- Database connection failure.
- Model artifact missing.
- Invalid prediction request.
- Failed model training.
- Scheduled job timeout.

The application should log useful diagnostic details while returning safe, concise error responses to clients.

### 13.4 Monitoring metrics

The system can track:

- ETL success and failure rates.
- Rows processed per run.
- ETL execution duration.
- Number of API requests.
- API latency and error rate.
- Prediction volume.
- Percentage of predictions flagged.
- Model precision and recall when verified labels are available.
- Changes in feature distributions.

A rising percentage of flagged transactions may indicate a real change in activity, an upstream data issue, or a model problem. It should **trigger investigation** rather than automatically be interpreted as a rise in actual fraud.

---

## 14. Project Structure

The proposed repository structure is:

```text
fraudlens/
│
├── src/
│   └── fraudlens/
│       ├── __init__.py
│       ├── config.py
│       │
│       ├── etl/
│       │   ├── __init__.py
│       │   ├── extract.py
│       │   ├── transform.py
│       │   ├── validate.py
│       │   └── load.py
│       │
│       ├── ml/
│       │   ├── __init__.py
│       │   ├── train.py
│       │   ├── evaluate.py
│       │   ├── predict.py
│       │   └── model_registry.py
│       │
│       ├── api/
│       │   ├── __init__.py
│       │   ├── main.py
│       │   ├── schemas.py
│       │   ├── dependencies.py
│       │   └── routes/
│       │       ├── health.py
│       │       ├── predictions.py
│       │       ├── models.py
│       │       ├── etl.py
│       │       └── schedules.py
│       │
│       ├── database/
│       │   ├── __init__.py
│       │   ├── connection.py
│       │   ├── models.py
│       │   └── repositories.py
│       │
│       ├── scheduler/
│       │   ├── __init__.py
│       │   ├── scheduler.py
│       │   └── jobs.py
│       │
│       ├── monitoring/
│       │   ├── __init__.py
│       │   ├── logging_config.py
│       │   └── metrics.py
│       │
│       └── dashboard/
│           └── app.py
│
├── migrations/
├── tests/
│   ├── test_etl.py
│   ├── test_validation.py
│   ├── test_model.py
│   ├── test_api.py
│   ├── test_database.py
│   └── test_scheduler.py
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── logs/
│
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── README.md
└── LICENSE
```

### Module dependency map

```mermaid
flowchart TD
    CFG["config.py"]
    API["api/"]
    ETL["etl/"]
    MLM["ml/"]
    DBM["database/"]
    SCH["scheduler/"]
    MON["monitoring/"]
    DASH["dashboard/"]

    API --> DBM
    API --> MLM
    API --> MON
    API --> CFG
    ETL --> DBM
    ETL --> MON
    ETL --> CFG
    MLM --> DBM
    MLM --> MON
    MLM --> CFG
    SCH --> ETL
    SCH --> MLM
    SCH --> DBM
    SCH --> MON
    DASH --> DBM
    MON --> CFG
```

### Directory responsibilities

| Directory | Responsibility |
|-----------|----------------|
| `etl/` | Data extraction, transformation, validation, and loading. |
| `ml/` | Model training, evaluation, inference logic, and model artifact management. |
| `api/` | HTTP endpoints and request validation. |
| `database/` | Database connection management, ORM entities, and persistence operations. |
| `scheduler/` | Scheduling configuration and job handlers. |
| `monitoring/` | Logging configuration and operational metrics. |
| `dashboard/` | Streamlit reporting interface. |
| `tests/` | Automated unit and integration tests. |
| `migrations/` | Database schema migration scripts. |
| `notebooks/` | Exploratory work, not production pipeline logic. |
| `data/` | Local development data, excluded from version control where appropriate. |
| `models/` | Local model artifacts, excluded from version control if large or generated. |

---

## 15. Installation and Setup

**Prerequisites:**

- Python 3.11 or another supported Python version.
- PostgreSQL.
- Git.
- A data.world account and API token.
- Optional: Docker and Docker Compose.

```mermaid
flowchart TD
    A["Clone the repository"] --> B["Create a virtual environment"]
    B --> C["Install dependencies"]
    C --> D["Create .env configuration"]
    D --> E["Create PostgreSQL database and user"]
    E --> F["Run database migrations"]
    F --> G["Select and configure the dataset"]
    G --> H["Run the initial ETL pipeline"]
    H --> I["Train the XGBoost model"]
    I --> J["Start the API with uvicorn"]
    J --> K["Verify http://localhost:8000/docs"]
```

### Step 1: Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/fraudlens.git
cd fraudlens
```

Replace `YOUR_USERNAME` with your GitHub username after creating the repository.

### Step 2: Create a virtual environment

```bash
python -m venv .venv
```

**Activate it on Linux or macOS:**

```bash
source .venv/bin/activate
```

**On Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

Install the project dependencies from the package configuration once it is created.

The main packages will include:

```bash
pip install pandas pyarrow datadotworld
pip install sqlalchemy "psycopg[binary]" alembic
pip install scikit-learn xgboost joblib
pip install fastapi uvicorn pydantic python-dotenv
pip install apscheduler pytest httpx
pip install streamlit plotly
```

> Pin and lock tested dependency versions before deployment.

### Step 4: Configure environment variables

Create a `.env` file:

```dotenv
DW_AUTH_TOKEN=your_data_world_api_token
DW_DATASET_KEY=owner/dataset-id

DATABASE_URL=postgresql+psycopg://fraudlens_user:your_password@localhost:5432/fraudlens

MODEL_PATH=models/fraud_model.joblib
LOG_LEVEL=INFO
```

> These are example configuration values. Replace the dataset identifier, token, and database credentials with your own.

> **Never commit `.env` to version control.**

### Step 5: Create the database

Create a PostgreSQL database and user, then run the project's database migrations.

For example, using PostgreSQL's command-line tools:

```sql
CREATE DATABASE fraudlens;
```

Create a dedicated database user with **only** the permissions the application needs. Configure the connection string to use that user.

The migration commands depend on the database migration setup implemented in the repository.

### Step 6: Select and configure the dataset

Verify the selected data.world dataset and update the extraction configuration.

**Confirm:**

- The correct dataset identifier.
- The actual source file name.
- The fraud target column.
- The feature columns.
- The license and data limitations.

### Step 7: Run the initial ETL pipeline

The intended execution order is:

```mermaid
flowchart LR
    A["Extract"] --> B["Validate"] --> C["Transform"] --> D["Save processed data"] --> E["Load into PostgreSQL"]
```

Run the individual modules using the package entry points or commands implemented by the project.

### Step 8: Train the model

Train the XGBoost model after the processed dataset is ready.

The output should include:

- Model artifact.
- Model version metadata.
- Evaluation report.
- Selected decision threshold.

### Step 9: Start the API

```bash
uvicorn fraudlens.api.main:app --reload
```

This command assumes the package is installed or the Python path is configured appropriately.

Open [http://localhost:8000/docs](http://localhost:8000/docs) to explore the implemented endpoints.

---

## 16. Running the Application

A typical local workflow is:

```mermaid
flowchart LR
    A["1. Configure environment"] --> B["2. Extract dataset"]
    B --> C["3. Validate and transform"]
    C --> D["4. Load processed transactions"]
    D --> E["5. Train and evaluate XGBoost"]
    E --> F["6. Register model artifact"]
    F --> G["7. Start FastAPI service"]
    G --> H["8. Submit prediction requests"]
    H --> I["9. Verify prediction records"]
    I --> J["10. Inspect ETL runs and logs"]
    J --> K["11. Start dashboard"]
    K --> L["12. Start scheduler"]
```

1. Configure the environment.
2. Extract the dataset.
3. Validate and transform the data.
4. Load processed transactions.
5. Train and evaluate XGBoost.
6. Register the selected model artifact.
7. Start the FastAPI service.
8. Submit prediction requests.
9. Verify prediction records in PostgreSQL.
10. Inspect ETL runs and application logs.
11. Start the dashboard when implemented.
12. Start the scheduler when implemented.

For reproducibility, the repository should provide **documented CLI commands** for each stage rather than relying exclusively on manual notebook execution.

The ETL and training scripts should also return meaningful **exit codes** so they can be called safely by a scheduler or CI workflow.

---

## 17. Example API Requests

The following examples illustrate how a client would interact with the planned API. Request fields must be adapted to the actual dataset schema.

**Health check**

```bash
curl http://localhost:8000/health
```

**Model information**

```bash
curl http://localhost:8000/model/info
```

**Submit a prediction**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "TXN-1042",
    "features": {
      "Amount": 850.0,
      "V1": -1.36,
      "V2": 0.72
    }
  }'
```

> This request will work only if the supplied features match the trained model's expected schema.

**Retrieve prediction history**

```bash
curl "http://localhost:8000/predictions?predicted_label=1&limit=20"
```

**List ETL runs**

```bash
curl http://localhost:8000/etl/runs
```

**List scheduled jobs**

```bash
curl http://localhost:8000/schedules
```

> These commands represent the target interface. An endpoint will return a successful response only after its handler, dependencies, and database integration have been implemented.

---

## 18. Testing Strategy

FraudLens should include unit, integration, and API tests.

```mermaid
flowchart TB
    UNIT["Unit tests<br/>validation, features, thresholds, config"]
    MLT["ML tests<br/>split, leakage, metrics, artifact round-trip"]
    INT["Integration tests<br/>ETL, database, persistence, migrations"]
    APIT["API tests<br/>endpoints, filters, pagination, errors"]
    REL["Pipeline reliability tests<br/>retries, locks, idempotency"]
    UNIT --> INT
    MLT --> INT
    REL --> INT
    INT --> APIT
```

### Unit tests

Test individual components in isolation:

- Schema validation.
- Missing-value detection.
- Duplicate handling.
- Feature preparation.
- Prediction threshold logic.
- Request validation.
- Configuration loading.

### Integration tests

Test interactions between components:

- Extraction output and transformation input.
- ETL loading into PostgreSQL.
- Prediction persistence.
- Model artifact loading.
- Schedule execution tracking.
- Database migrations.

### API tests

Verify that:

- Health endpoints return the expected status.
- Valid requests produce predictions.
- Missing features are rejected.
- Invalid numeric values are rejected.
- Unknown model artifacts produce safe errors.
- Predictions are persisted when persistence is enabled.
- History endpoints apply filters and pagination correctly.

### ML tests

Verify that:

- The training dataset contains both classes.
- No target leakage is introduced.
- Train, validation, and test data are separated correctly.
- Evaluation metrics are calculated correctly.
- The saved model can be loaded and used for inference.
- The model's expected feature schema matches the API schema.

### Pipeline reliability tests

Verify that:

- Failed ETL runs are recorded.
- Retrying a load does not create unintended duplicates.
- Failed scheduled jobs receive the correct status.
- Overlapping executions are prevented where necessary.
- Logs contain the execution identifiers needed for debugging.

---

## 19. Security Considerations

Financial transaction systems require careful handling of data and model outputs.

FraudLens should follow these practices:

- Store credentials in environment variables or a dedicated secrets manager.
- Exclude `.env`, raw datasets, and sensitive generated files from Git.
- Use parameterized database operations.
- Validate and limit API request sizes.
- Require authentication for sensitive operational and review endpoints.
- Apply authorization to model management and scheduling operations.
- Avoid logging full transaction payloads.
- Never store full payment card numbers or CVVs.
- Restrict access to prediction history and transaction records.
- Use HTTPS and appropriate access controls in deployed environments.
- Treat model predictions as **risk estimates**, not definitive proof of fraud.

> The initial development API may run without authentication on a local machine, but it must **not** be exposed publicly in that state.

---

## 20. Limitations

FraudLens has several important limitations.

| Limitation | Description |
|------------|-------------|
| **Dataset limitations** | Public datasets may not represent current transaction patterns, real bank workflows, or the full range of fraud types. |
| **Anonymized features** | Some datasets use transformed or anonymized features that cannot be interpreted as real transaction attributes. |
| **Class imbalance** | Fraud is often rare, so model evaluation must account for false positives and false negatives. |
| **Probability calibration** | A model's predicted probability may not match the actual observed fraud frequency without calibration and appropriate validation. |
| **Concept drift** | Fraud patterns can change over time, reducing model performance. |
| **Label availability** | True fraud outcomes may arrive after the original transaction. Monitoring model quality requires reliable labels and appropriate time alignment. |
| **Scheduling** | A scheduled training job does not guarantee that the resulting model should be deployed. Candidate models require evaluation and controlled promotion. |
| **Operational scope** | The initial implementation is a portfolio and learning project. It is **not** a certified banking fraud prevention system. |

---

## 21. Future Improvements

Once the core pipeline, model, and API are reliable, the project can be expanded in the following directions.

### Data engineering

- Incremental data ingestion.
- Schema evolution detection.
- More extensive data quality checks.
- Prefect or Airflow orchestration.
- Object storage for raw and processed datasets.
- Improved retry and recovery mechanisms.

### Machine learning

- Compare XGBoost with Random Forest and logistic regression.
- Hyperparameter optimization.
- Probability calibration.
- Explainability using SHAP.
- Time-aware validation.
- Model drift monitoring.
- Controlled retraining and model promotion.
- Experiment tracking with MLflow.

### Application development

- Authentication and role-based access.
- Batch scoring jobs.
- Review workflows.
- A richer Streamlit dashboard.
- Prediction export functionality.
- Notifications for failed ETL runs.
- Model-version comparison.

### Deployment

- Docker Compose for local multi-service development.
- Automated testing in CI.
- Cloud deployment.
- Centralized structured logging.
- Metrics collection and alerting.
- Separate API and scheduler workers.

> These improvements should be introduced incrementally. A reliable baseline is more valuable than an elaborate architecture that is difficult to test or explain.

---

## 22. Learning Outcomes

This project is intended to demonstrate practical competence across several areas.

| Area | Skills demonstrated |
|------|---------------------|
| **Data engineering** | Extracting external datasets, building ETL workflows, validating data quality, loading data into relational storage, designing idempotent processing, tracking pipeline executions, automating recurring jobs |
| **Machine learning** | Supervised binary classification, gradient-boosted decision trees, class imbalance, leakage prevention, threshold selection, precision-recall analysis, model serialization and versioning |
| **Backend engineering** | REST API design, request and response validation, database integration, error handling, prediction persistence, automated testing |
| **Operational engineering** | Logging and monitoring, scheduling and retries, configuration management, reproducibility, containerization, deployment readiness |

The overall goal is to demonstrate that **a machine learning model is one component of a larger data product**, rather than the entire application.

---

## 23. Contributing

Contributions and suggestions are welcome.

A useful contribution workflow is:

1. Fork the repository.
2. Create a feature branch.
3. Implement a focused change.
4. Add or update tests.
5. Run the test suite.
6. Submit a pull request with a description of the change.

> Please do not submit API tokens, credentials, sensitive transaction data, or licensed datasets that cannot be redistributed.

---

## 24. License

The repository license should be selected before publishing the project.

A permissive license such as **MIT** may be suitable for the project's source code, but the source dataset may have separate terms. **The code license does not automatically grant permission to redistribute third-party data.**

Include a `LICENSE` file and document the selected dataset's license and attribution requirements.

---

## Final Note

FraudLens aims to demonstrate a complete machine learning workflow, from raw transaction ingestion to model inference and operational monitoring.

Its central design principle is **reliability and traceability across the entire data lifecycle**: every pipeline run should be observable, every prediction should be associated with a model version, and every automated job should have a recorded execution history.

```mermaid
flowchart LR
    A["Raw transaction data"] --> B["Validated dataset"]
    B --> C["Trained model version"]
    C --> D["Prediction linked to model version"]
    D --> E["Review outcome"]
    E --> F["Performance monitoring"]
    F -.->|drift detected, retrain| B
```

The initial milestone is to select a suitable dataset, build the ETL pipeline, and establish a reproducible XGBoost baseline. The API, prediction history, scheduling, and monitoring capabilities can then be implemented as independent, testable components.

---

**FraudLens** — *From raw transactions to actionable fraud risk insights.*
