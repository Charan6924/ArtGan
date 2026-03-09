from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
    ax.set_xticklabels([s.replace('_', ' ') for s in order],
                       rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Outlier Score', fontsize=11)
    ax.set_title('Outlier Score Distribution by Style', fontsize=14, pad=16)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)

    for i, s in enumerate(order):
        med = df.loc[df['true_style'] == s, 'outlier_score'].median()
        ax.text(i, med + 0.005, f'{med:.3f}', ha='center', va='bottom',
                fontsize=6.5, color=ACCENT)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[plot 2] saved → {out_path}')


def plot_heatmap(df: pd.DataFrame, out_path: Path):
    # restrict to top styles/artists by mean outlier score to keep it readable
    top_styles  = df.groupby('true_style')['outlier_score'].mean().nlargest(15).index
    top_artists = df.groupby('true_artist')['outlier_score'].mean().nlargest(30).index

    subset = df[df['true_style'].isin(top_styles) & df['true_artist'].isin(top_artists)]

    if subset.empty:
        print('[plot 3] not enough data for heatmap, skipping')
        return

    matrix = subset.pivot_table(
        index='true_style', columns='true_artist',
        values='artist_dist', aggfunc='mean'
    )

    fig, ax = plt.subplots(figsize=(18, 7))
    fig.patch.set_facecolor(DARK)

    sns.heatmap(
        matrix,
        ax=ax,
        cmap=sns.color_palette('rocket', as_cmap=True),
        linewidths=0.2,
        linecolor='#1a1a1a',
        annot=False,
        cbar_kws={'shrink': 0.6, 'label': 'Mean artist centroid distance'},
    )

    ax.set_xticklabels([t.get_text().replace('-', ' ').title()
                        for t in ax.get_xticklabels()],
                       fontsize=6.5, rotation=45, ha='right')
    ax.set_yticklabels([t.get_text().replace('_', ' ')
                        for t in ax.get_yticklabels()],
                       fontsize=7.5, rotation=0)
    ax.set_xlabel('Artist',  fontsize=11, labelpad=10)
    ax.set_ylabel('Style',   fontsize=11, labelpad=10)
    ax.set_title('Mean Artist Centroid Distance\n(top outlier styles × artists)',
                 fontsize=13, pad=14)

    ax.collections[0].colorbar.ax.yaxis.label.set_color(LIGHT)
    ax.collections[0].colorbar.ax.tick_params(colors=LIGHT)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[plot 3] saved → {out_path}')


def _load_image(path: str, data_dir: Path, size: int = 160) -> Image.Image | None:
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
                canvas = Image.new('RGB', (size, size), (20, 20, 20))
                canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
                return canvas
            except Exception:
                pass
    return None


def plot_image_grid(df: pd.DataFrame, data_dir: Path, top_n: int, out_path: Path):
    top = df.nlargest(top_n, 'outlier_score').reset_index(drop=True)

    ncols = 5
    nrows = int(np.ceil(top_n / ncols))
    thumb = 160

    fig = plt.figure(figsize=(ncols * 2.8, nrows * 3.2), facecolor=DARK)
    fig.suptitle(f'Top-{top_n} Outlier Paintings', fontsize=13, color=LIGHT, y=1.01)

    gs = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.05, wspace=0.05)

    score_min = top['outlier_score'].min()
    score_max = top['outlier_score'].max()

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

        # border: red (high outlier) → green (lower outlier)
        t = (row['outlier_score'] - score_min) / (score_max - score_min + 1e-8)
        border = f'#{int(0xe0*t + 0x44*(1-t)):02x}{int(0x5c*t + 0xbb*(1-t)):02x}{int(0x2a*t + 0x66*(1-t)):02x}'
        for spine in ax.spines.values():
            spine.set_edgecolor(border)
            spine.set_linewidth(2.5)

        ax.set_xticks([])
        ax.set_yticks([])

        caption = (
            f"#{i+1}  score {row['outlier_score']:.3f}\n"
            f"{row['true_artist'].replace('-', ' ').title()}\n"
            f"{row['true_style'].replace('_', ' ')}"
        )
        ax.set_xlabel(caption, fontsize=5.5, color=LIGHT, labelpad=3, linespacing=1.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=DARK)
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
    plot_heatmap(df,    out_dir / 'outlier_heatmap.png')
    plot_image_grid(df, data_dir, cfg.top_n, out_dir / f'outlier_grid_top{cfg.top_n}.png')

    print(f'\n[done]  all plots saved to {out_dir}/')


if __name__ == '__main__':
    main()