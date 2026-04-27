# OntoGrasp: Ontology-Driven Grasp Interpretation

Companion code and ontology for:

> Boruah, A., Kakoty, N.M., Vinjamuri, R.K., et al. (2025).
> "Ontology-Driven Semantic Reasoning for Grasp Interpretation:
> An Approach beyond Data-Driven Classification."
> (under review).

## Contents

| Path | Description |
|---|---|
| `ontology/OntoGrasp_extended.owl` | Extended OntoGrasp OWL 2 DL ontology with SWRL rules and constraint axioms |
| `experiments/run_grasp_experiments.py` | Main CV experiment: data-driven baselines + ontology rule engine |
| `experiments/prepare_clinical_annotation.py` | Generates blinded annotation CSV for clinical validation of Constraint C2 |
| `experiments/analyze_clinical_annotation.py` | Analyses completed annotation: Cohen's κ, precision, recall |

## Requirements

Python 3.10+. Install dependencies:

```bash
pip install -r requirements.txt
```

Protégé 5.0.5 with HermiT reasoner is required for OWL-based reasoning.
Download: https://protege.stanford.edu

## Usage

```bash
# Run cross-validation experiments
python experiments/run_grasp_experiments.py

# Prepare clinical annotation file
python experiments/prepare_clinical_annotation.py

# Analyse completed annotation
python experiments/analyze_clinical_annotation.py
```

## License

MIT License. See LICENSE file.

## Contact

Abhijit Boruah — abhijit.boruah@dibru.ac.in  
Department of Computer Science and Engineering  
Dibrugarh University, Assam, India
