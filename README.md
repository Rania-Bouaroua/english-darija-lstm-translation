# English → Moroccan Darija Neural Machine Translation

Comparative Advanced NLP lab project implementing **Vanilla LSTM**, **CIFG-LSTM** and **Working Memory Connections (WMC-LSTM)** cells in PyTorch. Recurrent cells are implemented with `torch.nn.Linear`, without `torch.nn.LSTM`.

## Dataset
Hugging Face: [`imomayiz/darija-english`](https://huggingface.co/datasets/imomayiz/darija-english), configuration `sentences`. Translation direction: English → Latin-script Moroccan Darija. Dataset is downloaded when required; not redistributed here.

## Contents
- `src/`: dataset preparation, custom recurrent cells, seq2seq models, training, experiments, EDA and analysis.
- `tests/`: architecture and pipeline tests.
- `results/`: saved experimental outputs, predictions, and analysis (not a guarantee of generalization).
- `figures/`: plots generated from the experiment outputs.

## Environment
Python with a recent CUDA-compatible PyTorch installation is recommended. The experiments were run in Google Colab on a Tesla T4 GPU.

```bash
pip install -r requirements.txt
python -m pytest tests -q
```

Import and run modules from the project root. See `src/experiments.py` for main experiment and hyperparameter configuration, and `src/train.py` for the training routine. GPU setup and a full end-to-end entry notebook should be documented separately before claiming one-click reproducibility.

## Reported initial findings
Results in `results/table_main.csv` and `results/summary_findings.md` show very low translation-quality scores; these architectures are research baselines rather than a production translation service. Consult the saved artifacts for experimental settings and exact metrics. Training/test source-sentence overlap was addressed in the updated data split implementation.

## Reproducibility and limitations
The results use a limited sample and a single random seed; additional seeds and improved modeling/tokenization would be needed for stronger conclusions. Do not treat this repository as evidence of state-of-the-art translation quality.

## Academic attribution
Advanced NLP Lab Project (2026–2027). Add your team members and course attribution here before submitting.
