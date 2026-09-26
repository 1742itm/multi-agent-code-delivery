"""历史任务清理脚本。

管理 workspace/ 下由 main.py 产生的历史任务：
- workspace/sandbox/<run_id>/   Agent 写出的代码文件
- workspace/outputs/<run_id>/   四份产出文档（run_id 形如 20260924_153012）

安全策略：
1. 默认是“只列出不删除”的预览模式，必须显式指定删除条件；
2. 只识别严格匹配 YYYYMMDD_HHMMSS 的任务目录，其他目录（如 default）
   一律跳过，绝不触碰 workspace 以外的任何路径；
3. 删除前展示完整清单并要求二次确认（--yes 可跳过交互，便于脚本调用）；
4. sandbox 与 outputs 两侧同名目录配对删除，结果逐个核验汇总。

用法：
    python clean_history.py                  # 列出全部历史任务（不删除）
    python clean_history.py --keep 3         # 保留最近 3 个，删除其余
    python clean_history.py --id 20260924_153012
    python clean_history.py --all            # 删除全部历史任务
    python clean_history.py --all --yes      # 删除全部且跳过确认
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SANDBOX_ROOT = PROJECT_ROOT / "workspace" / "sandbox"
OUTPUTS_ROOT = PROJECT_ROOT / "workspace" / "outputs"

# 只认 main.py 生成的时间戳目录：20260924_153012
_RUN_ID_RE = re.compile(r"^\d{8}_\d{6}$")


@dataclass
class RunRecord:
    """一次任务运行在 sandbox / outputs 两侧的配对记录。"""

    run_id: str
    sandbox_dir: Path | None
    outputs_dir: Path | None

    @property
    def exists_any(self) -> bool:
        return (self.sandbox_dir is not None) or (self.outputs_dir is not None)

    @property
    def doc_count(self) -> int:
        return len(list(self.outputs_dir.glob("*.md"))) if self.outputs_dir else 0

    @property
    def code_count(self) -> int:
        return len(list(self.sandbox_dir.rglob("*"))) if self.sandbox_dir else 0

    @property
    def size_bytes(self) -> int:
        total = 0
        for base in (self.sandbox_dir, self.outputs_dir):
            if base:
                for f in base.rglob("*"):
                    if f.is_file():
                        try:
                            total += f.stat().st_size
                        except OSError:
                            pass
        return total

    @property
    def timestamp(self) -> datetime:
        return datetime.strptime(self.run_id, "%Y%m%d_%H%M%S")


def _human_size(num: int) -> str:
    size = float(num)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f}{unit}" if unit != "B" else f"{int(size)}B"
        size /= 1024
    return f"{size:.1f}GB"


def scan_runs() -> tuple[list[RunRecord], list[str]]:
    """扫描两侧目录，返回（配对后的任务列表，被跳过的非任务目录名）。"""

    def task_dirs(root: Path) -> tuple[dict[str, Path], list[str]]:
        if not root.is_dir():
            return {}, []
        matched, skipped = {}, []
        for child in root.iterdir():
            if child.is_dir() and _RUN_ID_RE.match(child.name):
                matched[child.name] = child
            elif child.is_dir():
                skipped.append(f"{root.name}/{child.name}")
        return matched, skipped

    sandbox_dirs, skip1 = task_dirs(SANDBOX_ROOT)
    outputs_dirs, skip2 = task_dirs(OUTPUTS_ROOT)

    all_ids = sorted(set(sandbox_dirs) | set(outputs_dirs), reverse=True)
    records = [
        RunRecord(
            run_id=rid,
            sandbox_dir=sandbox_dirs.get(rid),
            outputs_dir=outputs_dirs.get(rid),
        )
        for rid in all_ids
    ]
    return records, sorted(skip1 + skip2)


def print_list(records: list[RunRecord], skipped: list[str]) -> None:
    print(f"沙箱代码目录：{SANDBOX_ROOT}")
    print(f"产出文档目录：{OUTPUTS_ROOT}\n")

    if not records:
        print("没有发现任何历史任务（workspace 尚不存在或为空）。")
    else:
        print(f"共 {len(records)} 个历史任务（按时间倒序）：")
        print("-" * 78)
        print(f"{'序号':<4}{'任务ID (run_id)':<20}{'文档':<5}{'代码文件':<8}{'大小':<10}时间")
        print("-" * 78)
        for idx, rec in enumerate(records, 1):
            locs = []
            if rec.sandbox_dir:
                locs.append("沙箱")
            if rec.outputs_dir:
                locs.append("文档")
            print(
                f"{idx:<4}{rec.run_id:<20}{rec.doc_count:<5}{rec.code_count:<8}"
                f"{_human_size(rec.size_bytes):<10}"
                f"{rec.timestamp.strftime('%Y-%m-%d %H:%M:%S')}  [{'+'.join(locs)}]"
            )
        print("-" * 78)

    if skipped:
        print("\n以下目录不是任务目录，已自动跳过（永不删除）：")
        for name in skipped:
            print(f"  - {name}")


def select_targets(
    records: list[RunRecord], args: argparse.Namespace
) -> list[RunRecord]:
    """根据命令行参数选出待删除任务。records 已按时间倒序。"""
    if args.all:
        return list(records)
    if args.keep is not None:
        return records[args.keep :]  # 倒序排列，前 N 个是最新的
    if args.id:
        index = {rec.run_id: rec for rec in records}
        missing = [rid for rid in args.id if rid not in index]
        if missing:
            print("错误：以下任务ID不存在：" + ", ".join(missing), file=sys.stderr)
            sys.exit(1)
        return [index[rid] for rid in args.id]
    return []


def confirm_delete(targets: list[RunRecord]) -> bool:
    total_size = sum(r.size_bytes for r in targets)
    print(f"\n即将永久删除 {len(targets)} 个历史任务（共 {_human_size(total_size)}）：")
    for rec in targets:
        print(f"  - {rec.run_id}")
    print("\n删除后不可恢复（sandbox 代码与 outputs 文档会一并移除）。")
    try:
        answer = input("确认删除？请输入 yes 继续，其他任意输入取消：").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n已取消。")
        return False
    return answer.lower() == "yes"


def delete_runs(targets: list[RunRecord]) -> tuple[int, list[str]]:
    """逐个删除并核验，返回（成功任务数，失败信息）。"""
    success, failures = 0, []
    for rec in targets:
        run_failed = False
        for label, path in (("sandbox", rec.sandbox_dir), ("outputs", rec.outputs_dir)):
            if path is None:
                continue
            try:
                shutil.rmtree(path)
                if path.exists():
                    failures.append(f"{rec.run_id}/{label}: 删除后目录仍存在")
                    run_failed = True
                else:
                    print(f"  已删除 {path.relative_to(PROJECT_ROOT)}")
            except OSError as exc:
                failures.append(f"{rec.run_id}/{label}: {exc}")
                run_failed = True
        if not run_failed:
            success += 1
    return success, failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="清理 workspace 下的历史任务（默认只列出，不删除）"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="删除全部历史任务")
    group.add_argument("--keep", type=int, metavar="N", help="保留最近 N 个任务，删除其余")
    group.add_argument("--id", nargs="+", metavar="RUN_ID", help="删除指定的一个或多个任务ID")
    parser.add_argument("--yes", action="store_true", help="跳过交互确认直接删除")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.keep is not None and args.keep < 0:
        print("错误：--keep 的值不能为负数（--keep 0 等同于 --all）。", file=sys.stderr)
        sys.exit(1)

    records, skipped = scan_runs()
    # 先做参数校验（如 --id 不存在会直接报错退出），再打印清单
    targets = select_targets(records, args)
    print_list(records, skipped)

    if not (args.all or args.keep is not None or args.id):
        # 预览模式
        print("\n当前为预览模式，未做任何删除。")
        print("删除请使用：--keep N（保留最近N个）/ --id 任务ID / --all，详见 --help。")
        return

    if not targets:
        print("\n没有符合条件的历史任务需要删除。")
        return

    if not args.yes and not confirm_delete(targets):
        print("已取消，未删除任何内容。")
        return

    print("\n开始删除：")
    success, failures = delete_runs(targets)
    print(f"\n完成：成功清理 {success} 个任务。")
    if failures:
        print(f"{len(failures)} 处删除失败：", file=sys.stderr)
        for msg in failures:
            print(f"  - {msg}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
