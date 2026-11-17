"""Explicit, editable ANALYTICAL assumptions, not NEU's official grading policy."""
from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class Settings:
    # Minimum count for statistical inference; below threshold is provisional.
    min_section_size: int = 12
    min_peer_sections: int = 2
    min_prior_fraction: float = 0.35
    min_predicted_rows: int = 8
    # Research thresholds: evaluate on synthetically labelled anomalies.
    anomaly_flag_threshold: float = 0.49
    direction_min_shift: float = 0.65
    direction_min_z: float = 1.30
    review_q_threshold: float = 0.20
    max_adjustment: float = 1.50
    max_pred_interval: float = 4.00
    min_tci_courses: int = 4
    prior_strength_beta: float = 16.0
    # Inclusive lower bounds, in descending order; DEMONSTRATION ONLY.
    # The source data says passing starts at 4.5 but does not prescribe GPA 4 conversion.
    grade_scale: tuple = ((8.5, 4.0), (8.0, 3.5), (7.0, 3.0),
                          (6.5, 2.5), (5.5, 2.0), (5.0, 1.5),
                          (4.5, 1.0), (0.0, 0.0))

    def json(self):
        return asdict(self)

SETTINGS = Settings()
