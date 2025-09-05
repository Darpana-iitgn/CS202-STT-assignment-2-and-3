#!/usr/bin/env python3  # Shebang line for portability
import os  # Filesystem interactions
import re  # Regex for commit message patterns
import csv  # CSV export
import sys  # System-related functions
import subprocess  # Run git commands
from pathlib import Path  # Path handling
from typing import List, Tuple, Dict, Set  # Type hints
import pandas as pd  # DataFrames and CSV writing
from pydriller import Repository  # Commit traversal

# Keywords to detect bug-fix commits
BUG_KEYWORDS = ["fix", "bug", "issue", "resolve", "patch", "correct", "error", "defect", "fail", "broken", "hotfix", "fixes", "fixed", "bugfix", "bug-fix"]

# Blob size safety limit
MAX_BLOB_BYTES = 500_000

def run_git(repo_path: str, args: List[str]) -> Tuple[int, str, str]:
    """Run a git command and return (returncode, stdout, stderr)."""
    cmd = ["git", "-C", repo_path] + args
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out = proc.stdout.decode("utf-8", errors="replace")
    err = proc.stderr.decode("utf-8", errors="replace")
    return proc.returncode, out, err

def is_binary_blob(b: bytes) -> bool:
    """Heuristic to detect binary blobs."""
    if not b:
        return False
    if b.count(0) > 0:
        return True
    printable = set(range(32, 127)) | {9, 10, 13}
    non_printable = sum(1 for x in b if x not in printable)
    return non_printable / len(b) > 0.30

def safe_git_show_blob(repo_path: str, rev: str, path: str) -> str:
    """Safely return file content or empty string if invalid/binary/too large."""
    if not rev or not path:
        return ""
    rc, out, err = run_git(repo_path, ["show", f"{rev}:{path}"])
    if rc != 0:
        return ""
    lowerr = (out + " " + err).lower()
    if "missing" in lowerr or "fatal" in lowerr or "bad object" in lowerr or "not found" in lowerr:
        return ""
    data = out.encode("utf-8", errors="replace")
    if len(data) > MAX_BLOB_BYTES or is_binary_blob(data):
        return ""
    return out

def git_merge_files(repo_path: str, commit_hash: str) -> List[str]:
    """List files changed in a merge commit."""
    rc, out, err = run_git(repo_path, ["show", "-m", "--name-only", "--pretty=", commit_hash])
    if rc != 0:
        return []
    files = [ln.strip() for ln in out.splitlines() if ln.strip()]
    seen: Set[str] = set()
    unique: List[str] = []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return unique

def git_parents(repo_path: str, commit_hash: str) -> List[str]:
    """Return parent commit hashes."""
    rc, out, err = run_git(repo_path, ["rev-list", "--parents", "-n", "1", commit_hash])
    if rc != 0:
        return []
    parts = out.strip().split()
    return parts[1:] if len(parts) > 1 else []

def git_diff_for_file_against_parent(repo_path: str, parent: str, child: str, path: str) -> str:
    """Get unified diff between parent and child for one file."""
    if not parent:
        return ""
    rc, out, err = run_git(repo_path, ["diff", "-U0", parent, child, "--", path])
    if rc != 0:
        return ""
    lowerr = (out + " " + err).lower()
    if "missing" in lowerr or "fatal" in lowerr or "bad object" in lowerr:
        return ""
    return out

def is_bug_fixing_commit(commit_message: str) -> bool:
    """Decide whether commit message indicates a bug fix."""
    if not commit_message:
        return False
    msg = commit_message.lower()
    for kw in BUG_KEYWORDS:
        if kw in msg:
            return True
    if re.search(r'(fix(es|ed)?|close(s|d)?|resolve(s|d)?)\s*#\d+', msg):
        return True
    if re.search(r'(fix(es|ed)?|close(s|d)?|resolve(s|d)?)\s*(gh|issue)-\d+', msg):
        return True
    return False

def identify_and_store_bug_fixing_commits(repo_path: str, output_dir: str, commits_csv_filename: str = "bugfix_commits.csv", modified_files_csv_filename: str = "bugfix_modified_files.csv") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Mine bug-fixing commits and save commit/file CSVs."""
    print(f"--- Mining repository: {repo_path} ---")
    os.makedirs(output_dir, exist_ok=True)
    all_commit_rows: List[Dict[str, object]] = []
    all_file_rows: List[Dict[str, object]] = []
    try:
        for commit in Repository(repo_path).traverse_commits():
            try:
                if not is_bug_fixing_commit(commit.msg):
                    continue
                parent_hashes = [str(p) for p in commit.parents]
                is_merge = bool(commit.merge)
                if not is_merge:
                    modified_files_list = [mf.new_path if mf.new_path else mf.old_path for mf in commit.modified_files if (mf.new_path or mf.old_path)]
                    all_commit_rows.append({"Hash": commit.hash or "", "Message": commit.msg or "", "Hashes of parents": parent_hashes, "Is a merge commit?": is_merge, "List of modified files": modified_files_list})
                    for mf in commit.modified_files:
                        filename = mf.new_path if mf.new_path else mf.old_path
                        if not filename:
                            continue
                        all_file_rows.append({"Hash": commit.hash or "", "Message": commit.msg or "", "Filename": filename, "Source Code (before)": mf.source_code_before or "", "Source Code (current)": mf.source_code or "", "Diff": mf.diff or "", "LLM Inference (fix type)": "", "Rectified Message": ""})
                else:
                    files = git_merge_files(repo_path, commit.hash)
                    all_commit_rows.append({"Hash": commit.hash or "", "Message": commit.msg or "", "Hashes of parents": parent_hashes, "Is a merge commit?": True, "List of modified files": files})
                    if not files:
                        continue
                    first_parent = parent_hashes[0] if parent_hashes else None
                    for path in files:
                        before_blob = safe_git_show_blob(repo_path, first_parent, path) if first_parent else ""
                        current_blob = safe_git_show_blob(repo_path, commit.hash, path)
                        diffs_accum: List[str] = []
                        for p in parent_hashes:
                            d = git_diff_for_file_against_parent(repo_path, p, commit.hash, path)
                            if d.strip():
                                diffs_accum.append(d)
                        merged_diff = "\n".join(diffs_accum)
                        all_file_rows.append({"Hash": commit.hash or "", "Message": commit.msg or "", "Filename": path, "Source Code (before)": before_blob, "Source Code (current)": current_blob, "Diff": merged_diff, "LLM Inference (fix type)": "", "Rectified Message": ""})
            except Exception as ce:
                print(f"[WARN] Skipping commit due to error: {commit.hash if hasattr(commit, 'hash') else 'unknown'} -> {ce}")
    except Exception as e:
        print(f"[ERROR] Mining failed: {e}")
    commits_csv_path = os.path.join(output_dir, commits_csv_filename)
    files_csv_path = os.path.join(output_dir, modified_files_csv_filename)
    df_commits = pd.DataFrame(all_commit_rows)
    df_files = pd.DataFrame(all_file_rows)
    commit_cols = ["Hash", "Message", "Hashes of parents", "Is a merge commit?", "List of modified files"]
    file_cols = ["Hash", "Message", "Filename", "Source Code (before)", "Source Code (current)", "Diff", "LLM Inference (fix type)", "Rectified Message"]
    for c in commit_cols:
        if c not in df_commits.columns:
            df_commits[c] = ""
    for c in file_cols:
        if c not in df_files.columns:
            df_files[c] = ""
    if not df_commits.empty:
        df_commits["Hashes of parents"] = df_commits["Hashes of parents"].apply(lambda x: ";".join(map(str, x)) if isinstance(x, (list, tuple)) else (x or ""))
        df_commits["List of modified files"] = df_commits["List of modified files"].apply(lambda x: ";".join(map(str, x)) if isinstance(x, (list, tuple)) else (x or ""))
    df_commits = df_commits.fillna("").astype(str)
    df_files = df_files.fillna("").astype(str)
    df_commits.to_csv(commits_csv_path, index=False, quoting=csv.QUOTE_ALL, escapechar="\\", encoding="utf-8")
    df_files.to_csv(files_csv_path, index=False, quoting=csv.QUOTE_ALL, escapechar="\\", encoding="utf-8")
    print(f"✅ Bug-fixing commits saved to {commits_csv_path}")
    print(f"✅ Modified files data saved to {files_csv_path}")
    print(f"ℹ️ Commits: {len(df_commits)} | File records: {len(df_files)}")
    return df_commits, df_files

def main():
    """Entry point for script execution."""
    repo_path = "archivebox-full"
    output_dir = "output_files"
    identify_and_store_bug_fixing_commits(repo_path=repo_path, output_dir=output_dir, commits_csv_filename="bugfix_commits.csv", modified_files_csv_filename="modified_files.csv")

if __name__ == "__main__":
    main()

