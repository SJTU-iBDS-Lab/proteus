"""Gray-box skill chain scanner based on AI-Infra-Guard (AIG).

Calls AIG's mcp-scan agent to audit individual skills, then aggregates
findings across a skill chain to determine if the chain is harmful.
"""

from .aig_scanner import scan_skill
from .chain_scanner import ChainScanResult, scan_chain

__all__ = ["ChainScanResult", "scan_chain", "scan_skill"]
