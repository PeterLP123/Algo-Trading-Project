"""
utils.py — Shared utilities used across the src package.

Functions:
  display_df(df)                           — show DataFrame in notebook or script mode
  section_header(step, title)              — print a consistent STEP N banner
  apply_template(fig)                      — apply the shared Plotly template
  save_plotly_fig(fig, name, output_dir)   — save figure as HTML + PNG + PDF
"""

from pathlib import Path

from src import config


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


def apply_template(fig):
    """Apply the shared Plotly theme to *fig* and return it."""
    fig.update_layout(template=config.PLOTLY_TEMPLATE)
    return fig


def save_plotly_fig(fig, name: str, output_dir, formats=None):
    """Save *fig* to all configured export formats in *output_dir*."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    formats = formats or config.EXPORT_FORMATS

    apply_template(fig)

    for fmt in formats:
        target = out / f"{name}.{fmt}"
        if fmt == "html":
            fig.write_html(target, include_plotlyjs="cdn", full_html=True)
        elif fmt in {"png", "pdf"}:
            fig.write_image(target, scale=2)
        else:
            raise ValueError(f"Unsupported plot export format: {fmt}")
