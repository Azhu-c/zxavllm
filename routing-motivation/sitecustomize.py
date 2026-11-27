"""Opt-in import hooks. Active only in this experiment's Decoder children."""
import os

if os.environ.get("MOTIVATION_ROLE") == "decoder":
    from runtime import install_import_hooks
    install_import_hooks()
