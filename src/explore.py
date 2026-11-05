"""Step 1 on Jammertest: see what is inside the extracted folder.
Run: python -m src.explore <extracted_folder>"""
import os, sys, collections

root = sys.argv[1]
ext, paths = collections.Counter(), []
for d, _, files in os.walk(root):
    for f in files:
        ext[os.path.splitext(f)[1].lower() or "(none)"] += 1
        paths.append(os.path.join(d, f))
print("File types:", dict(ext.most_common()))
print(f"Total files: {len(paths)}\nFirst 30 paths:")
for p in sorted(paths)[:30]:
    print(" ", p)
