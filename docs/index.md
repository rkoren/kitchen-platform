# light-ml-platform

A two-module portfolio project bridging platform engineering with ML infrastructure.

---

## Modules

<div class="grid cards" markdown>

-   :material-console-line: **recipes**

    ---

    A lightweight CLI that converts simple YAML specs into Terraform configurations for AWS resources. Open-source and cloud-agnostic.

    [:octicons-arrow-right-24: Get started](recipes/quickstart.md)

-   :material-pipe: **pipeline**

    ---

    An end-to-end MLOps pipeline for healthcare tabular data — data versioning, experiment tracking, model serving, drift monitoring, and orchestration.

    [:octicons-arrow-right-24: Get started](pipeline/quickstart.md)

</div>

## Stack

| Concern | Tool |
|---|---|
| IaC generation | recipes CLI + Terraform |
| Data versioning | DVC + S3 |
| Experiment tracking | MLflow |
| Model serving | FastAPI + Docker → Lambda (ECR) |
| Monitoring | Evidently AI |
| Orchestration | Prefect |

AWS-native where necessary; open-source tooling throughout.
