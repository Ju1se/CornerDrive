"""
FLPG L4: On-Chain Settlement
Dashboard API and blockchain settlement management.

Core functionality:
- SBT credit system with tier multipliers
- Blockchain settlement via smart contracts
- Analytics and management dashboard
- Oracle service for L2→L4 settlement
"""

from .dashboard_api import (
    app,
    SystemStats,
    VehicleStats,
    TierDistribution,
    RecentAudit,
    HealthResponse,
    SettlementBatch,
)

__version__ = "1.0.0"
__author__ = "FLPG Team"

__all__ = [
    "app",
    "SystemStats",
    "VehicleStats",
    "TierDistribution",
    "RecentAudit",
    "HealthResponse",
    "SettlementBatch",
]