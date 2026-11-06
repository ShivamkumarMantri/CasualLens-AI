# Proposed research protocol

## Research hypothesis
H1: Evidence-grounded LLM explanations that are generated from structured statistical outputs have a lower unsupported-numerical-claim rate than unconstrained LLM explanations.

H2: Users evaluating the grounded explanations identify estimator assumptions and uncertainty more accurately than users evaluating unconstrained explanations.

## Experiment A — Estimator validity
Generate semi-synthetic data with known ATE values (e.g. 0.5, 1.0, 2.0), varying confounding strength, sample size, treatment prevalence and overlap. Run at least 200 Monte Carlo replications per setting. Report bias, RMSE and 95% CI coverage.

## Experiment B — Assistant faithfulness
Prepare a benchmark of questions whose answers are explicitly represented in the computed context. Compare:
1. No assistant / statistical output only
2. Unconstrained LLM
3. Evidence-grounded LLM

Blindly annotate numerical correctness, evidence support, causal overclaiming and assumption disclosure. Use two annotators and report agreement.

## Experiment C — User study
If institutionally permitted, recruit participants, obtain informed consent, anonymize responses and use a pre-registered task set. Measure task accuracy, time and calibrated confidence. Do not collect unnecessary personal information.

## Reproducibility
Pin dependency versions for final experiments, store random seeds, version data and code, preserve experiment configurations, and publish a model card / dataset card where appropriate.
