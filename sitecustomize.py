# sitecustomize.py — project-local bootstrap for matplotlib backend
import globals

print("sitecustomize ... \n")

if not globals.on_anvil:
    try:
        import matplotlib

        matplotlib.use("Qt5Agg")
    except Exception:
        pass
