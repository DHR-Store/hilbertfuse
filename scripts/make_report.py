"""Build docs/RESULTS.md from a benchmark results.csv:  python scripts/make_report.py bench/results.csv docs/RESULTS.md"""
import sys
import pandas as pd
from hilbertfuse.benchmark import summarize

src, dst = sys.argv[1], sys.argv[2]
res = pd.read_csv(src)
res["method"] = res.method.replace({"new": "hfnet", "new_lr": "fuse"})   # legacy names from early runs
md = summarize(res, ".", reference="fuse")
wil_fuse = md[md.index("### Paired"):]
md2 = summarize(res, ".", reference="hfnet")
wil_hf = md2[md2.index("### Paired"):]
head = md[:md.index("### Paired")]
n = res.groupby(["dataset", "method"]).split.nunique().unstack()
with open(dst, "w") as f:
    f.write("# Benchmark results (real cancer datasets)\n\n")
    f.write(f"Folds per dataset: {int(n.max().max())}. Leak-free: all preprocessing re-fit inside each training fold.\n\n")
    f.write(head + "\n" + wil_fuse + "\n\n" + wil_hf + "\n")
print(open(dst).read())
