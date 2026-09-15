# Case Closed: A Custom Graph Dataset for Whodunit Prediction

A custom dataset built from the *Case Closed* (Detective Conan) anime, where each episode becomes a small graph and the task is to predict which suspect is the murderer. The idea was to build a clean, self-contained problem for graph machine learning without the noise and access issues of real crime data.
Each case has a known answer, the entities and relationships are well defined, and the murderer is labeled, so a model can be trained and checked against ground truth.

## The task

Every case is a directed, typed graph. Among the persons marked as suspects, exactly one is the murderer. The prediction target is a per-case label that picks that suspect out, framed as a per-graph classification problem. The dataset was designed with graph neural networks in mind.

## Dataset format

Each episode is stored as a JSON file under `data/case_closed_dataset/cases_v3/`, with names like `ep011_moonlight_sonata_murder_case.json`. The dataset covers 247 cases. After removing single-suspect cases, which offer no real choice to predict, 237 remain.

**Nodes** are heterogeneous and fall into four types:

- Persons, each with a role such as suspect, witness, or victim
- Locations
- Objects
- Clues

**Edges** are directed and typed. Examples include `was_at`, `owns`, `implicates`, and `relationship`, and relationship edges carry an extra `relation_type` attribute. So the graph captures who was where, who owns what, and what evidence points at whom.

## Repository structure

```
data/case_closed_dataset/cases_v3/   the case files, one JSON per episode
Reo analysis/v5/
  case_closedv5.ipynb                 exploratory analysis of the graphs
case_closed_dataset.py                loader (list_case_files, load_case, ...)
```

The loader exposes small helpers (`list_case_files`, `load_case`) so a notebook can pull cases without reparsing the JSON by hand.

## What the analysis notebook does

`case_closedv5.ipynb` is descriptive, not predictive. It uses numpy, pandas, and matplotlib to summarize the dataset: node and edge counts per case, the distribution of person roles, how often each edge type appears, and average degree computed as 2E/N. The goal at this stage was to understand the shape of the data before modeling, not to run graph algorithms.

## Running it

```bash
pip install numpy pandas matplotlib
# open Reo analysis/v5/case_closedv5.ipynb and run the cells
```

The notebook imports `case_closed_dataset` to load the JSON cases, so keep the loader and the `data/` folder in place relative to the notebook.
