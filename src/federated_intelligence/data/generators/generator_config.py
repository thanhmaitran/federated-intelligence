"""Configuration for the synthetic federated tabular generator."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

GENERATOR_VERSION = "1.0.0"
DATASET_NAME = "synthetic_federated_tabular"


class Scenario(StrEnum):
    IID = "iid"
    LABEL_SKEW = "label_skew"
    FEATURE_SKEW = "feature_skew"
    QUANTITY_SKEW = "quantity_skew"
    COMBINED = "combined_non_iid"

    @property
    def skews_labels(self) -> bool:
        return self in (Scenario.LABEL_SKEW, Scenario.COMBINED)

    @property
    def skews_features(self) -> bool:
        return self in (Scenario.FEATURE_SKEW, Scenario.COMBINED)

    @property
    def skews_quantity(self) -> bool:
        return self in (Scenario.QUANTITY_SKEW, Scenario.COMBINED)


class GeneratorConfig(BaseModel):
    """All knobs of the data-generating process (see docs/methodology)."""

    seed: int = 42
    num_clients: int = Field(50, ge=2)
    num_features: int = Field(24, ge=2)
    num_classes: int = 2
    min_samples_per_client: int = Field(300, ge=20)
    max_samples_per_client: int = Field(2000, ge=20)
    scenario: Scenario = Scenario.IID

    # Label skew: p_k ~ Beta(a, a); small a => strongly different class priors.
    label_beta: float = Field(0.5, gt=0)
    balanced_positive_rate: float = Field(0.5, gt=0, lt=1)
    # Feature skew: mu_k = Delta_k ~ N(0, feature_shift^2 I), Sigma_k = diag(scale_k^2).
    feature_shift: float = Field(1.0, ge=0)
    feature_scale_range: tuple[float, float] = (0.6, 1.6)
    # Concept perturbation shared by all scenarios: w_k = w_0 + N(0, concept_noise^2 I).
    concept_noise: float = Field(0.05, ge=0)
    # Quantity skew: n_k ~ LogNormal(0, sigma), rescaled into [min, max].
    quantity_sigma: float = Field(1.0, gt=0)

    label_noise: float = Field(0.0, ge=0, lt=0.5)
    weight_scale: float = Field(1.0, gt=0)
    global_test_size: int = Field(5000, ge=100)

    train_fraction: float = 0.70
    val_fraction: float = 0.15
    test_fraction: float = 0.15

    @model_validator(mode="after")
    def _check(self) -> GeneratorConfig:
        if self.num_classes != 2:
            raise ValueError("Only binary classification is supported (num_classes=2).")
        if self.max_samples_per_client < self.min_samples_per_client:
            raise ValueError("max_samples_per_client must be >= min_samples_per_client")
        total = self.train_fraction + self.val_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"split fractions must sum to 1, got {total}")
        return self
