from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from PIL import Image


@dataclass
class Config:
    csv:      str = 'outliers.csv'   
    data_dir: str = 'data/wikiart'   
    top_n:    int = 20               
    out_dir:  str = 'plots'       

cfg = Config()


DARK   = '#0f0f0f'
LIGHT  = '#f5f0e8'
ACCENT = '#e05c2a'
GREY   = '#555555'

plt.rcParams.update({
    'figure.facecolor': DARK,
    'axes.facecolor':   DARK,
    'axes.edgecolor':   GREY,
    'axes.labelcolor':  LIGHT,
    'xtick.color':      LIGHT,
    'ytick.color':      LIGHT,
    'text.color':       LIGHT,
    'grid.color':       '#2a2a2a',
    'grid.linestyle':   '--',
    'font.family':      'monospace',
})



def plot_violin(df: pd.DataFrame, out_path: Path):
    order = (
        df.groupby('true_style')['outlier_score']
          .median()
          .sort_values(ascending=False)
          .index.tolist()
    )

    fig, ax = plt.subplots(figsize=(18, 7))
    fig.patch.set_facecolor(DARK)

    parts = ax.violinplot(
        [df.loc[df['true_style'] == s, 'outlier_score'].values for s in order],
        positions=range(len(order)),
        showmedians=True,
        showextrema=True,
        widths=0.75,
    )

    for pc in parts['bodies']:
        pc.set_facecolor(ACCENT)
        pc.set_alpha(0.55)
        pc.set_edgecolor(LIGHT)
        pc.set_linewidth(0.6)
    for key in ('cmedians', 'cmins', 'cmaxes', 'cbars'):
        parts[key].set_color(LIGHT)
        parts[key].set_linewidth(1.2)
    parts['cmedians'].set_color(ACCENT)
    parts['cmedians'].set_linewidth(2)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(
        [s.replace('_', ' ') for s in order],
        rotation=45, ha='right', fontsize=8
    )
    ax.set_ylabel('Outlier Score', fontsize=11)
    ax.set_title('Outlier Score Distribution by Style', fontsize=14, pad=16)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)

    # annotate median on each violin
    for i, s in enumerate(order):
        med = df.loc[df['true_style'] == s, 'outlier_score'].median()
        ax.text(i, med + 0.12, f'{med:.1f}', ha='center', va='bottom',
                fontsize=6.5, color=ACCENT)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[plot 2] saved → {out_path}')


def plot_confusion(df: pd.DataFrame, out_path: Path):
    all_styles = sorted(set(df['true_style'].tolist() + df['pred_style'].tolist()))

    matrix = pd.crosstab(
        df['true_style'], df['pred_style'],
        normalize='index'   # row-normalise → fraction of true class
    ).reindex(index=all_styles, columns=all_styles, fill_value=0)

    fig, ax = plt.subplots(figsize=(16, 14))
    fig.patch.set_facecolor(DARK)

    cmap = sns.color_palette('rocket', as_cmap=True)
    sns.heatmap(
        matrix,
        ax=ax,
        cmap=cmap,
        linewidths=0.3,
        linecolor='#1a1a1a',
        annot=False,
        cbar_kws={'shrink': 0.6, 'label': 'Fraction of true class'},
        square=True,
    )

    # highlight the diagonal
    n = len(all_styles)
    for i in range(n):
        ax.add_patch(
            mpatches.Rectangle(
                (i, i), 1, 1,
                fill=False, edgecolor='#44ff88', linewidth=1.2
            )
        )

    labels = [s.replace('_', '\n') for s in all_styles]
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha='right')
    ax.set_yticklabels(labels, fontsize=7, rotation=0)
    ax.set_xlabel('Predicted Style', fontsize=11, labelpad=10)
    ax.set_ylabel('True Style',      fontsize=11, labelpad=10)
    ax.set_title('Style Confusion Matrix\n(row-normalised, diagonal = correct)',
                 fontsize=13, pad=14)

    ax.collections[0].colorbar.ax.yaxis.label.set_color(LIGHT)
    ax.collections[0].colorbar.ax.tick_params(colors=LIGHT)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[plot 3] saved → {out_path}')


def _load_image(path: str, data_dir: Path, size: int = 160) -> Image.Image | None:
    """Try absolute path first, then search under data_dir by filename."""
    candidates = [
        Path(path),
        data_dir / Path(path).name,
        *data_dir.rglob(Path(path).name),
    ]
    for p in candidates:
        if p.exists():
            try:
                img = Image.open(p).convert('RGB')
                img.thumbnail((size, size), Image.LANCZOS)
                # pad to square
                canvas = Image.new('RGB', (size, size), (20, 20, 20))
                x = (size - img.width)  // 2
                y = (size - img.height) // 2
                canvas.paste(img, (x, y))
                return canvas
            except Exception:
                pass
    return None


def plot_image_grid(df: pd.DataFrame, data_dir: Path, top_n: int, out_path: Path):
    top = df.nlargest(top_n, 'outlier_score').reset_index(drop=True)

    ncols = 5
    nrows = int(np.ceil(top_n / ncols))
    thumb = 160
    pad   = 6

    fig_w = ncols * (thumb / 100) + (ncols + 1) * pad / 100
    fig_h = nrows * (thumb / 100 + 0.95) + 0.5  

    fig = plt.figure(figsize=(fig_w * 1.6, fig_h * 1.6), facecolor=DARK)
    fig.suptitle(f'Top-{top_n} Outlier Paintings', fontsize=13,
                 color=LIGHT, y=1.01)

    gs = gridspec.GridSpec(
        nrows, ncols,
        figure=fig,
        hspace=0.05,
        wspace=0.05,
    )

    for i, row in top.iterrows():
        r, c = divmod(i, ncols)
        ax = fig.add_subplot(gs[r, c])
        ax.set_facecolor(DARK)

        img = _load_image(row['path'], data_dir, size=thumb)
        if img is not None:
            ax.imshow(np.array(img))
        else:
            ax.text(0.5, 0.5, 'not found', transform=ax.transAxes,
                    ha='center', va='center', fontsize=7, color=GREY)

        # border colour: red if both wrong, orange if one wrong, green if both right
        n_wrong = int(row['style_wrong']) + int(row['artist_wrong'])
        border = {2: ACCENT, 1: '#e0a020', 0: '#44bb66'}[n_wrong]
        for spine in ax.spines.values():
            spine.set_edgecolor(border)
            spine.set_linewidth(2)

        ax.set_xticks([])
        ax.set_yticks([])

        # caption below the image
        artist = row['true_artist'].replace('-', ' ').title()
        style  = row['true_style'].replace('_', ' ')
        pred_s = row['pred_style'].replace('_', ' ')
        score  = row['outlier_score']

        caption = (
            f"#{i+1}  score {score:.1f}\n"
            f"{artist}\n"
            f"true: {style}\n"
            f"pred: {pred_s}"
        )
        ax.set_xlabel(caption, fontsize=5.5, color=LIGHT,
                      labelpad=3, linespacing=1.5)

    legend_elements = [
        mpatches.Patch(facecolor=ACCENT,    label='Both style & artist wrong'),
        mpatches.Patch(facecolor='#e0a020', label='One label wrong'),
        mpatches.Patch(facecolor='#44bb66', label='Both correct (centroid outlier)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=3, fontsize=8, framealpha=0.2,
               bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=DARK, bbox_extra_artists=[])
    plt.close(fig)
    print(f'[plot 4] saved → {out_path}')

def main():
    out_dir  = Path(cfg.out_dir)
    data_dir = Path(cfg.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'[*] Loading {cfg.csv}')
    df = pd.read_csv(cfg.csv)
    print(f'[*] {len(df):,} rows loaded')

    plot_violin(df,     out_dir / 'outlier_by_style.png')
    plot_confusion(df,  out_dir / 'style_confusion.png')
    plot_image_grid(df, data_dir, cfg.top_n, out_dir / f'outlier_grid_top{cfg.top_n}.png')

    print(f'\n[done]  all plots saved to {out_dir}/')


if __name__ == '__main__':
    main()