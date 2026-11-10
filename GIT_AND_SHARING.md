# Publish on GitHub so anyone can `git clone` and run

## You (the owner): publish once
```powershell
cd C:\path\to\hilbertfuse
git init
git add .
git commit -m "HilbertFuse: models, code, docs"
git branch -M main
# create an EMPTY repository on github.com first (no README), then:
git remote add origin https://github.com/<your-user>/hilbertfuse.git
git push -u origin main
```
* The repository is about 25 MB (trained models + real data sets): well under GitHub's limits, no Git LFS needed.
* `.gitignore` already keeps your outputs (`runs/`, `predictions*.csv`) and a `data_private/` folder out of git.
  **Never commit real patient data.** Keep it in `data_private/` (ignored) or outside the folder.
* To update later: `git add . ; git commit -m "update" ; git push`.

## Anyone else (user): clone and run
```powershell
git clone https://github.com/<your-user>/hilbertfuse.git
cd hilbertfuse
.\start_windows.bat          # or just double-click start_windows.bat in the folder
```
First run installs Python packages (5-10 minutes, internet needed; Python 3.10-3.12 required). Afterwards it opens the menu: **2 = predict**, **3 = evaluate**, **4 = train my own model (quick)**, **7 = show all scores**.
Linux/macOS: `bash start.sh`.

Predictions run **offline on your own computer**; no data is sent anywhere by this tool (internet is only used once, for installing packages).

## Using patient or cancer data safely
* The file must have one row per sample and the **same gene columns** as the model (see `models\<name>\required_genes.txt`). The tool refuses files with the wrong genes.
* Output includes a confidence value and an UNCERTAIN flag. Treat low-confidence rows as "unknown".
* Results are research output, not a diagnosis. Keep patient files in `data_private\` so they cannot be committed by accident, and follow your institution's data-protection rules.
