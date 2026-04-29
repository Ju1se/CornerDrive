"""
FLPG L2: Dual-Purpose Audit
The Sniper - Classifies gradients as FRAUD, RARITY, NOISE, or HONEST.

Core functionality:
- Dual-channel loss evaluation (main dataset + corner cases)
- Beneficial-rarity classification for corner-case updates
- Asynchronous batch processing with Celery
"""

from .classifier import Classification, AuditResult, DualChannelAuditor

__version__ = "1.0.0"
__author__ = "FLPG Team"


def __getattr__(name):
    if name in {"celery_app", "audit_gradient", "get_statistics"}:
        from . import worker

        return getattr(worker, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "Classification",
    "AuditResult",
    "DualChannelAuditor",
    "celery_app",
    "audit_gradient",
    "get_statistics",
]
