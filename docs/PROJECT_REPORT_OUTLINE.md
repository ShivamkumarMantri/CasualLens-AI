# CausalLens AI — Major Project Report

## Abstract
Describe the research problem, observational-data limitations, implemented estimators, evidence-grounded LLM layer, evaluation design and measured results. Do not insert fabricated results.

## Chapter 1 — Introduction
- Background: data-driven decision-making and limitations of correlation
- Motivation for combining causal inference with GenAI
- Problem statement
- Objectives and scope
- Research questions

## Chapter 2 — Literature Review
Review primary research on:
- Potential outcomes framework (Rubin)
- Directed acyclic graphs and causal identification (Pearl)
- Propensity-score methods (Rosenbaum & Rubin)
- Doubly robust estimation / AIPW
- Double/debiased machine learning (Chernozhukov et al.)
- Retrieval-augmented generation (Lewis et al.)
- Faithfulness and evaluation of LLM-generated explanations

Use verified bibliographic details and DOI/URL links from papers you actually read.

## Chapter 3 — System Analysis and Design
- Functional and non-functional requirements
- Use-case diagram
- Data-flow diagram
- Architecture diagram
- Data dictionary and threat model
- Privacy and ethical considerations

## Chapter 4 — Methodology
- Dataset inclusion and provenance
- Missing-data strategy
- Treatment, outcome and covariate definitions
- Causal estimand (ATE/ATT)
- Identification assumptions
- OLS adjustment and propensity-score matching
- Confidence intervals and diagnostics
- Evidence construction and LLM prompt constraints

## Chapter 5 — Implementation
Explain modules, algorithms, UI, API configuration, error handling and deployment.

## Chapter 6 — Experimental Evaluation
Use synthetic datasets with known causal effects and/or justified public datasets.
Report:
- Bias = estimate − ground truth
- RMSE over repeated seeds
- 95% interval coverage
- Propensity-score overlap and standardized mean differences
- Assistant numerical faithfulness
- Unsupported-claim rate
- Assumption-disclosure rate
- Latency and API cost

Compare: statistical-only baseline, unconstrained LLM, evidence-grounded LLM. Include ablation studies.

## Chapter 7 — Results and Discussion
Insert only observed results. Discuss failure cases, uncertainty, generalizability and threats to validity.

## Chapter 8 — Conclusion and Future Scope
Summarize contributions without overstating causality. Discuss AIPW/TMLE, DML, causal forests, sensitivity analysis and literature RAG.

## References
Use a consistent citation style (IEEE/APA). Verify every paper's authors, title, venue, year and DOI.

## Appendices
- User manual
- Test cases
- API configuration
- Sample dataset dictionary
- Screenshots
- Reproducibility checklist
