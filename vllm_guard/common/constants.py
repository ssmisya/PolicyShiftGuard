from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

CANONICAL_DATASET_DIR = (
    REPO_ROOT / "data_curation" / "outputs" / "v2.7_withreason"
)
CANONICAL_DATASET_NAME = "adaptive_policy_v2.7_withreason"
CANONICAL_HF_DATASET_REPO = "PolicyShiftGuard/PolicyShiftBench"

ADAPTIVE_POLICY_EVAL_SPLITS = ("id_test", "ood_test", "sft", "sft_think", "rl")
ADAPTIVE_POLICY_CATEGORY_NAMES = {
    1: "Nudity, Sexual Content & Fetish",
    2: "Violence, Hate, Terrorism & Self-Harm",
    3: "Regulated Goods & Substances",
    4: "IP, Copyright & Brand Safety",
    5: "Cultural & Religious Sensitivity",
    6: "Privacy & PII",
    7: "Text-in-Image Safety",
}
