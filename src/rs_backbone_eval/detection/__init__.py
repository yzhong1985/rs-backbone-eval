"""Detection-side shared code: OBB geometry today, detector and AP next.

Nothing here may import torch at module level unless the module itself is
worker-only; ``geometry`` is numpy-only so the manifest builder and the CPU
evaluator can use it, which keeps the control plane free of torch (AGENTS.md).
"""
