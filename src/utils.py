"""
utils.py — Shared utilities used across the src package.

Functions:
  display_df(df)                           — show DataFrame in notebook or script mode
  section_header(step, title)              — print a consistent STEP N banner
  save_fig(fig, name, output_dir, dpi=150) — save figure as PNG + PDF then show
  format_date_axis(ax, fmt="%Y")           — apply YearLocator + DateFormatter to x-axis
"""

from pathlib import Path


def display_df(df):
    """Show a DataFrame — works in both IPython/Jupyter and plain script mode."""
    try:
        from IPython.display import display
        display(df)
    except Exception:
        print(df)


def section_header(step: int, title: str):
    """Print a consistent STEP N banner to stdout."""
    print(f"\n{'=' * 60}\nSTEP {step}: {title}\n{'=' * 60}")


def save_fig(fig, name: str, output_dir, dpi: int = 150):
    """Save *fig* as both PNG (at *dpi*) and PDF into *output_dir*, then plt.show()."""
    import matplotlib.pyplot as plt
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", bbox_inches="tight", dpi=dpi)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.show()


def format_date_axis(ax, fmt: str = "%Y"):
    """Apply YearLocator + DateFormatter to *ax* x-axis."""
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
