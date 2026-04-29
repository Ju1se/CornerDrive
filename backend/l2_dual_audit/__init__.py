"""
FLPG L2: Dual-Purpose Audit
The Sniper - Classifies gradients as FRAUD, RARITY, NOISE, or HONEST.

Core functionality:
- Dual-channel loss evaluation (main dataset + corner cases)
- Beneficial-rarity classification for corner-case updates
- Asynchronous batch processing with Celery
"""

from .classifier import Classification, AuditResult, DualChannelAuditor
from .worker import celery_app, audit_gradient, get_statistics

__version__ = "1.0.0"
__author__ = "FLPG Team"

__all__ = [
    "Classification",
    "AuditResult",
    "DualChannelAuditor",
    "celery_app",
    "audit_gradient",
    "get_statistics",
]
