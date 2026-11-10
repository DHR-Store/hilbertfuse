# Windows PowerShell setup + first real-data run.
# From the project folder:   powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
# Requires Python 3.10-3.12 from python.org (tick "Add python.exe to PATH").
function Run($exe, $argList) {
    & $exe @argList
    if ($LASTEXITCODE -ne 0) { throw "Command failed: $exe $argList" }
}
Run "python" @("-m", "venv", ".venv")
$py = ".\.venv\Scripts\python.exe"
Run $py @("-m", "pip", "install", "--upgrade", "pip")
# CPU build of PyTorch (small). NVIDIA GPU users: get the command for your CUDA version at https://pytorch.org
Run $py @("-m", "pip", "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu")
Run $py @("-m", "pip", "install", "-e", ".[dev,datasets]")
Run $py @("-m", "pytest", "-q", "tests")
Run $py @("-m", "hilbertfuse.cli", "real-demo", "--dataset", "khan", "--out", "runs\real_demo")
Write-Host "Done. Results are in runs\real_demo\khan"
