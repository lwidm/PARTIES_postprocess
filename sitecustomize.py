# sitecustomize.py — project-local bootstrap for matplotlib backend
import os

on_anvil = os.getenv("MY_MACHINE", "") == "anvil"

if not on_anvil:
    try:
        import matplotlib

        matplotlib.use("Qt5Agg")
    except Exception:
        pass
