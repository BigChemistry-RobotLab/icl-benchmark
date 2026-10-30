If you made any changes, run `uvx ty check && uv run ruff check .` at the end of the session. Fix any emerging errors. Use `uv run ruff check . --fix` if the errors can be corrected automatically.
Write comments following the Google-style docstring convention.
Use ASD-STE100 simplified technical English for comments.
Never add backwards-compatibility to features you add/update. If you change something, keep only the new version behaviour. Do not track versions.