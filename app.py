"""BrandGuard CLI — review one piece of copy from the command line.

Usage:
    python app.py "Rise and train. Built for athletes who show up every day."
    echo "Crush the competition." | python app.py -
"""
from __future__ import annotations

import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agent import review_copy

load_dotenv()
console = Console()


def render(review) -> None:
    colour = {"APPROVE": "green", "APPROVE_WITH_EDITS": "yellow", "REJECT": "red"}[
        review.verdict
    ]
    console.print(
        Panel(
            f"[bold {colour}]{review.verdict}[/]   "
            f"safety = {review.safety_label} ({review.safety_confidence:.0%})",
            title="BrandGuard verdict",
        )
    )

    if review.issues:
        console.print("\n[bold]Issues:[/]")
        for i in review.issues:
            console.print(f"  • {i}")

    if review.cited_guidelines:
        t = Table(title="Cited guidelines", show_lines=False)
        t.add_column("ID", style="cyan")
        t.add_column("Reason")
        for c in review.cited_guidelines:
            t.add_row(c.guideline_id, c.reason)
        console.print(t)

    if review.suggested_rewrite:
        console.print(
            Panel(review.suggested_rewrite, title="Suggested rewrite", style="yellow")
        )


def main() -> None:
    if len(sys.argv) < 2:
        console.print("[red]Usage:[/] python app.py \"<copy>\"  |  python app.py -")
        sys.exit(1)

    copy = sys.stdin.read() if sys.argv[1] == "-" else sys.argv[1]
    console.print(Panel(copy.strip(), title="Input copy", style="white"))
    review = review_copy(copy.strip())
    render(review)


if __name__ == "__main__":
    main()
