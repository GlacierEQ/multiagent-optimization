from pathlib import Path
_source = Path(__file__).resolve().parents[1] / "recovery_manager.py"
exec(compile(_source.read_text(encoding="utf-8"), str(_source), "exec"), globals(), globals())
