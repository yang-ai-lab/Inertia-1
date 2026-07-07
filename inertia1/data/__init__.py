"""Shared data pipeline for ts_ssl.

All methods consume a single canonical batch format:

  x: torch.FloatTensor of shape [B, C, T]

Optional metadata fields may be added later.
"""
