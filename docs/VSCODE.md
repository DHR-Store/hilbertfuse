# Using HilbertFuse directly in VS Code (Windows)

## Open and prepare (once)
1. VS Code -> **File > Open Folder** -> choose the `hilbertfuse` folder.
2. **Terminal > New Terminal** (Ctrl+`). It opens PowerShell in the project folder.
3. First time only (skip if you already ran `scripts\setup_windows.ps1` or `start_windows.bat`):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   pip install -e ".[datasets]"
   ```
4. Every new terminal: `.\.venv\Scripts\Activate.ps1` (prompt shows `(.venv)`). If blocked: `Set-ExecutionPolicy -Scope Process Bypass`.
   Optional: **Ctrl+Shift+P > Python: Select Interpreter > .\.venv\Scripts\python.exe**; the included `.vscode\settings.json` then activates it automatically.

## Commands (type in the VS Code terminal)
```powershell
hilbertfuse predict  --model models\golub --input my.csv --out predictions.csv
hilbertfuse evaluate --model models\golub --test my_labeled.csv
hilbertfuse train --train my_data.csv --label-col label --out models\mymodel
hilbertfuse menu
```
Replace `my.csv`, `my_labeled.csv`, `my_data.csv` with the real path of your files (relative to the project folder, or a full path such as `C:\data\samples.csv`).

If `hilbertfuse` is "not recognized" (venv not active): use `.\.venv\Scripts\hilbertfuse.exe predict ...` or `python -m hilbertfuse.cli predict ...`.
In PowerShell a `.bat` in the current folder must be started with `.\`, for example `.\start_windows.bat predict ...` (it waits for a key press at the end; the direct commands above do not).

## Without typing: VS Code tasks
**Terminal > Run Task...** then pick one: *HilbertFuse: menu*, *predict*, *evaluate*, *train my model*, *honest accuracy (cross-validation)*. VS Code asks for the file names and model name in boxes at the top. (Tasks use `.venv`, so do the one-time setup first.)

## Check that it works (any real data set from the project)
```powershell
hilbertfuse predict --model models\khan --input datasets\khan.csv.gz --out predictions.csv
```
(This only checks the command runs; scoring a model on its own training data tells you nothing about accuracy. For honest accuracy use `hilbertfuse cv` or the numbers in `models\<name>\MODEL_CARD.md`.)
