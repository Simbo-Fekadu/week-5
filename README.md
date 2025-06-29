# Credit Risk Probability Model

## Business Context: Credit Scoring & Regulation

## Basel II and Model Interpretability

Basel II sets strict requirements for credit risk models:

- Accurate Risk Quantification: Banks must justify capital reserves with reliable models.
- Transparency: Regulators and auditors need to clearly understand how risk scores are calculated.
- Documentation: All assumptions, especially proxy variables (like RFM metrics), must be statistically validated and well-documented.
- Auditability: Models should allow for easy tracing of decisions and outcomes.

> Fact: Basel II compliance is mandatory for international banks, and non-compliance can result in heavy penalties or increased capital requirements.

Example: Logistic Regression with Weights of Evidence (WoE) is often preferred because its coefficients are interpretable and auditable, unlike complex models such as Gradient Boosting Machines (GBM).

---

## Proxy Variables: Purpose and Pitfalls

Why use proxies?

- Many eCommerce datasets lack direct "default" labels.
- Behavioral metrics like RFM (Recency, Frequency, Monetary) are used to estimate credit risk:
  - Recency: Time since last transaction.
  - Frequency: Number of transactions in a period.
  - Monetary: Total transaction value.

Risks of proxies:

- Misalignment: RFM may not fully capture true default risk.
- Bias: May unfairly penalize infrequent but reliable customers.
- Regulatory Scrutiny: Proxies must be justified with statistical evidence and business rationale.
- Data Drift: Customer behavior can change over time, reducing proxy effectiveness.

> Fact: Regulators may require back-testing and ongoing validation of proxy-based models.

---

## Model Selection: Simplicity vs. Complexity

| Logistic Regression (Simple)    | Gradient Boosting (Complex)          |
| ------------------------------- | ------------------------------------ |
| ✅ Interpretable, easy to audit | ❌ Opaque, "black box"               |
| ✅ Basel II compliant           | ❌ May face regulatory challenges    |
| ❌ May miss complex patterns    | ✅ Captures non-linear relationships |
| ✅ Faster to deploy and monitor | ❌ Requires more resources           |

Recommendation:  
Begin with Logistic Regression for regulatory compliance and transparency. Consider advanced models like GBM only if they provide significant accuracy improvements and can be explained to regulators.

> Fact: Many banks use a two-stage approach: simple models for compliance, complex models for internal risk management.
