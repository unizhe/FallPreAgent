"""Small dispatcher which leaves model imports to the selected demonstration."""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="本周公开数据演示：人体动作 / 米制场景")
    parser.add_argument("demo", choices=("human", "scene", "five-actions", "joint", "evaluate-human", "doctor"))
    supplied = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(supplied[:1])
    remaining = supplied[1:]
    if args.demo == "human":
        from .human import main as selected
    elif args.demo == "scene":
        from .scene import main as selected
    elif args.demo == "evaluate-human":
        from .evaluate_human import main as selected
    elif args.demo == "five-actions":
        from .five_actions import main as selected
    elif args.demo == "joint":
        from .joint import main as selected
    else:
        from .doctor import main as selected
    selected(remaining)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(2)
