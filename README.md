# End-to-End Production ML Pipeline with Metaflow 🚀

A production-ready ML pipeline using Metaflow for the complete lifecycle: data ingestion, training, evaluation, deployment, and monitoring. Built with the Palmer Penguins dataset but easily extensible to other datasets.

## ✨ Key Features

- End-to-end ML workflow with Metaflow orchestration
- Training pipeline with 5-fold cross-validation
- Inference serving with MLflow integration
- SQLite-based production backend for logging predictions
- Comprehensive unit tests
- Model monitoring and ground truth labeling
- Reproducible experiments with environment isolation

## 🚀 Quick Start

### Setup

```bash
just env       # Create .env file
uv sync        # Install dependencies
just mlflow    # Start MLflow tracking server (http://127.0.0.1:5000)
```

### Training

```bash
just train     # Run training pipeline with cross-validation
just cards     # View training metrics and cards
```

### Inference & Testing

```bash
just inference-test    # Run inference test suite
just serve             # Serve latest registered model (http://0.0.0.0:8080)
just sqlite            # View production database stats
```

## 📁 Project Structure

```
src/
├── pipelines/training.py       # Metaflow training pipeline
├── inference/
│   ├── model.py               # MLflow PythonModel
│   └── backend.py             # Backend abstraction (Local, Mock)
└── common/pipeline.py         # Base pipeline with decorators

tests/inference/               # Comprehensive test suite
├── test_model_backend.py
├── test_model_artifacts.py
└── test_model_predict.py

config/local.yml               # Configuration
data/penguins.csv             # Palmer Penguins dataset
```

## 🏗️ Pipeline Overview

**Training Pipeline**: Data loading → Cross-validation splits → Feature engineering → Model training → Evaluation → Conditional registration to MLflow

**Inference Pipeline**: Context loading → Input processing → Prediction → Output formatting → Backend logging

## 📊 Example Inference Request

Once the model is serving, make predictions with:

```bash
uv run -- curl -X POST http://0.0.0.0:8080/invocations \
    -H "Content-Type: application/json" \
    -d '{"inputs": [{"island": "Biscoe", "culmen_length_mm": 48.6, "culmen_depth_mm": 16.0, "flipper_length_mm": 230.0, "body_mass_g": 5800.0, "sex": "MALE"}]}'
```

Returns a JSON response with predicted penguin species and confidence scores.

## 🧪 Testing

Run the full test suite:

```bash
just inference-test
```

Tests cover:
- Backend configuration and initialization
- Model artifact loading
- Input/output processing
- Prediction generation and formatting

## 📝 License

MIT License - See [LICENSE](LICENSE) file for details