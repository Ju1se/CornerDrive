"""
FLPG Centralized Configuration
All system parameters for L1-L4 layers.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def _csv_env(name: str, default: str) -> list[str]:
    return [
        item.strip()
        for item in os.getenv(name, default).split(",")
        if item.strip()
    ]

# ============ NETWORK INFRASTRUCTURE ============
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1")
GANACHE_URL = os.getenv("GANACHE_URL", "http://localhost:8545")

# ============ L1: LINEAR DEFENSE ============
L1_HOST = os.getenv("L1_HOST", "127.0.0.1")
L1_PORT = int(os.getenv("L1_PORT", "8081"))
L1_SUSPECT_THRESHOLD = float(os.getenv("L1_SUSPECT_THRESHOLD", "0.3"))
L1_GEOMETRIC_MEDIAN_MAX_ITER = int(os.getenv("L1_GEOMETRIC_MEDIAN_MAX_ITER", "100"))
L1_GEOMETRIC_MEDIAN_EPS = float(os.getenv("L1_GEOMETRIC_MEDIAN_EPS", "1e-6"))
L1_BATCH_SIZE = int(os.getenv("L1_BATCH_SIZE", "10"))
L1_BATCH_TIMEOUT = float(os.getenv("L1_BATCH_TIMEOUT", "5.0"))
L1_RECHECK_PROBABILITY = float(os.getenv("L1_RECHECK_PROBABILITY", "0.0"))
L1_MAX_GRADIENT_DIM = int(os.getenv("L1_MAX_GRADIENT_DIM", "50000"))
L1_MAX_GRADIENT_ABS = float(os.getenv("L1_MAX_GRADIENT_ABS", "1000000"))

# ============ L2: DUAL-PURPOSE AUDIT ============
L2_FRAUD_THRESHOLD = float(os.getenv("L2_FRAUD_THRESHOLD", "0.05"))  # θ_tol
L2_RARITY_THRESHOLD = float(os.getenv("L2_RARITY_THRESHOLD", "-0.03"))  # θ_rare
L2_LEARNING_RATE = float(os.getenv("L2_LEARNING_RATE", "0.01"))  # η
L2_AUDIT_QUEUE = os.getenv("L2_AUDIT_QUEUE", "l2_audit_queue")

# ============ L3: GATEKEEPER ============
L3_DRIFT_THRESHOLD = float(os.getenv("L3_DRIFT_THRESHOLD", "0.05"))
L3_GOLDEN_DATASET_PATH = os.getenv("L3_GOLDEN_DATASET_PATH", "data/validation/golden")

# ============ L4: SETTLEMENT ============
L4_DASHBOARD_HOST = os.getenv("L4_DASHBOARD_HOST", "127.0.0.1")
L4_DASHBOARD_PORT = int(os.getenv("L4_DASHBOARD_PORT", "8082"))
L4_CONTRACT_ADDRESS = os.getenv("L4_CONTRACT_ADDRESS", "")
L4_ORACLE_PRIVATE_KEY = os.getenv("L4_ORACLE_PRIVATE_KEY", "")

# ============ SBT CREDIT SYSTEM ============
SBT_HONEST_REWARD = int(os.getenv("SBT_HONEST_REWARD", "1"))
SBT_RARITY_REWARD = int(os.getenv("SBT_RARITY_REWARD", "10"))
SBT_FRAUD_PENALTY = int(os.getenv("SBT_FRAUD_PENALTY", "-50"))

# Tier Thresholds
SBT_TIER_BRONZE = 0
SBT_TIER_SILVER = 100
SBT_TIER_GOLD = 500
SBT_TIER_PLATINUM = 1000

# Tier Multipliers (basis points, 100 = 1x)
SBT_MULTIPLIER_BRONZE = 100
SBT_MULTIPLIER_SILVER = 120
SBT_MULTIPLIER_GOLD = 150
SBT_MULTIPLIER_PLATINUM = 200

# ============ SECURITY ============
VALID_API_KEYS = _csv_env("VALID_API_KEYS", "change-me-local-dev-key")
RATE_LIMIT = os.getenv("RATE_LIMIT", "100/minute")
DEFAULT_CORS_ALLOWED_ORIGINS = (
    "http://localhost:3000,"
    "http://127.0.0.1:3000,"
    "http://localhost:8080,"
    "http://127.0.0.1:8080"
)
CORS_ALLOWED_ORIGINS = _csv_env("CORS_ALLOWED_ORIGINS", DEFAULT_CORS_ALLOWED_ORIGINS)
L1_CORS_ALLOWED_ORIGINS = _csv_env(
    "L1_CORS_ALLOWED_ORIGINS",
    ",".join(CORS_ALLOWED_ORIGINS),
)
L4_CORS_ALLOWED_ORIGINS = _csv_env(
    "L4_CORS_ALLOWED_ORIGINS",
    ",".join(CORS_ALLOWED_ORIGINS),
)

# ============ GLM LLM ASSISTANT ============
GLM_API_KEY = os.getenv("GLM_API_KEY", "")
GLM_BASE_URL = os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
GLM_MODEL = os.getenv("GLM_MODEL", "glm-4.7")
GLM_TIMEOUT = int(os.getenv("GLM_TIMEOUT", "5"))
GLM_MODE = os.getenv("GLM_MODE", "hybrid")  # hybrid, always, rule_only

# ============ LOGGING ============
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
