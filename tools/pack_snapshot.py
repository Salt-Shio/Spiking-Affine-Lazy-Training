"""把不進 git 的本機檔案打成一個 zip,帶到另一台機器(那邊程式碼用 git clone)。

打包哪些檔案由 .gitignore 的段落決定:"# [打包]" 段的規則一定打包,"# [打包:dataset]" 段要加
--with-dataset,"# [垃圾]" 段不打包。符合規則的檔案由 git 判斷。
另外帶 Claude 的全域 CLAUDE.md、examples/ 跟這個專案的記憶,放在 zip 的 _claude_home/ 底下,
目錄結構跟 ~/.claude 相同。

用法:python tools/pack_snapshot.py [--with-dataset]
輸出:repo 根目錄的 snapshot_<時間>.zip。repo 的檔案解到 clone 的根目錄,_claude_home/ 的內容放進 ~/.claude。
"""
import argparse
import datetime
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GITIGNORE = REPO_ROOT / ".gitignore"
CLAUDE_HOME = Path.home() / ".claude"
CLAUDE_ARCHIVE_DIR = "_claude_home"

SECTION_PACK = "打包"
SECTION_PACK_DATASET = "打包:dataset"
SECTION_JUNK = "垃圾"
_SECTION_HEADER = re.compile(r"^#\s*\[(.+?)\]")


def gitignore_sections(text: str) -> dict[str, list[str]]:
    """.gitignore 的內容 -> {段名: 規則}。

    段落從 "# [段名]" 開始;其他 # 開頭的行跟空行略過。段名不是 SECTION_* 之一、
    或有規則不在任何段落底下時 raise ValueError。
    """
    known = (SECTION_PACK, SECTION_PACK_DATASET, SECTION_JUNK)
    sections: dict[str, list[str]] = {name: [] for name in known}
    current = None
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        header = _SECTION_HEADER.match(line)
        if header:
            current = header.group(1)
            if current not in known:
                raise ValueError(f".gitignore 第 {line_no} 行:不認得的段名 [{current}],"
                                 f"認得的是 {list(known)}")
            continue
        if not line or line.startswith("#"):
            continue
        if current is None:
            raise ValueError(f".gitignore 第 {line_no} 行的規則 {line!r} 不在任何段落底下")
        sections[current].append(line)
    return sections


def ignored_files(patterns: list[str]) -> list[str]:
    """repo 裡沒進 git、符合 patterns(.gitignore 語法)的檔案,回傳相對 repo 根目錄的路徑。"""
    if not patterns:
        return []
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".gitignore",
                                     delete=False) as f:
        f.write("\n".join(patterns) + "\n")
        exclude_path = f.name
    try:
        out = subprocess.run(
            ["git", "ls-files", "--others", "--ignored", f"--exclude-from={exclude_path}", "-z"],
            cwd=REPO_ROOT, check=True, capture_output=True).stdout
    finally:
        os.unlink(exclude_path)
    return sorted(p for p in out.decode("utf-8").split("\0") if p)


def claude_project_dir_name(repo_root: Path) -> str:
    """Claude Code 存這個專案記憶的資料夾名:路徑裡英數字以外的字元都換成 "-"。

    照這台機器 ~/.claude/projects/ 的實際資料夾名寫的,例如
    /home/salt/Projects/Spiking-Affine-Lazy-Training -> -home-salt-Projects-Spiking-Affine-Lazy-Training。
    """
    return re.sub(r"[^A-Za-z0-9]", "-", str(repo_root))


def claude_files() -> list[Path]:
    """要帶走的 Claude 檔案(絕對路徑):全域 CLAUDE.md、examples/、這個專案的 memory/。
    不存在的項目印出提示後略過。"""
    entries = [CLAUDE_HOME / "CLAUDE.md", CLAUDE_HOME / "examples",
               CLAUDE_HOME / "projects" / claude_project_dir_name(REPO_ROOT) / "memory"]
    files = []
    for entry in entries:
        if entry.is_file():
            files.append(entry)
        elif entry.is_dir():
            files.extend(sorted(p for p in entry.rglob("*") if p.is_file()))
        else:
            print(f"略過(不存在):{entry}")
    return files


def write_zip(zip_path: Path, repo_files: list[str], home_files: list[Path]) -> None:
    """repo 檔案放在 zip 根目錄,Claude 檔案放在 CLAUDE_ARCHIVE_DIR/ 底下(相對 ~/.claude)。"""
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in repo_files:
            zf.write(REPO_ROOT / rel, arcname=rel)
        for path in home_files:
            zf.write(path, arcname=f"{CLAUDE_ARCHIVE_DIR}/{path.relative_to(CLAUDE_HOME).as_posix()}")


def _summary(repo_files: list[str], home_files: list[Path]) -> list[str]:
    """每個頂層資料夾(experiments 細到 run)的檔案數,給使用者核對打包內容。"""
    counts: dict[str, int] = {}
    for rel in repo_files:
        parts = rel.split("/")
        key = "/".join(parts[:2]) if parts[0] == "experiments" and len(parts) > 2 else parts[0]
        counts[key] = counts.get(key, 0) + 1
    if home_files:
        counts[f"{CLAUDE_ARCHIVE_DIR}/"] = len(home_files)
    return [f"  {n:5d}  {key}" for key, n in sorted(counts.items())]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--with-dataset", action="store_true",
                        help=f"連 .gitignore [{SECTION_PACK_DATASET}] 段的檔案一起打包")
    args = parser.parse_args()

    try:
        sections = gitignore_sections(GITIGNORE.read_text(encoding="utf-8"))
    except ValueError as e:
        sys.exit(str(e))
    patterns = sections[SECTION_PACK] + (sections[SECTION_PACK_DATASET] if args.with_dataset else [])
    repo_files = ignored_files(patterns)
    home_files = claude_files()

    zip_path = REPO_ROOT / f"snapshot_{datetime.datetime.now():%Y%m%d_%H%M%S}.zip"
    write_zip(zip_path, repo_files, home_files)
    print("\n".join(_summary(repo_files, home_files)))
    print(f"寫入 {zip_path}({zip_path.stat().st_size / 2**20:.1f} MB)")


if __name__ == "__main__":
    main()
