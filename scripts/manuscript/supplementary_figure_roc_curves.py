"""Render held-out receiver-operating-characteristic curves for every finding."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from champs_pipeline.eval.display_data import display_inputs, evaluation_data
from champs_pipeline.figures import style
from champs_pipeline.figures.labels import FINDINGS
from champs_pipeline.figures.output import save_figure


HEIGHT_MM = 100


def draw(scored, summary, summary_method="sd"):
    """Draw one thin model-colour ROC curve per outer fold in every finding panel."""
    fig = plt.figure(figsize=style.figure_size(style.FULL_WIDTH_MM, HEIGHT_MM))
    # With 180 mm width these margins and spacings give 32 mm square panels.
    grid = fig.add_gridspec(2, 4, left=0.115, right=0.985, bottom=0.12, top=0.91,
                            wspace=0.32, hspace=0.50)
    summary = summary.set_index(["organ_group", "finding"])
    source = {}
    roc_axes = []
    lung_findings = list(FINDINGS["lung"].items())
    liver_findings = list(FINDINGS["liver"].items())
    panel_order = [
        [("lung", finding, title) for finding, title in lung_findings[:4]],
        [("lung", *lung_findings[4]),
         *(("liver", finding, title) for finding, title in liver_findings), None],
    ]
    for row, panels in enumerate(panel_order):
        for column, panel_spec in enumerate(panels):
            if panel_spec is None:
                continue
            organ, finding, title = panel_spec
            ax = fig.add_subplot(grid[row, column])
            roc_axes.append(ax)
            panel = scored[(scored["organ_group"] == organ) & (scored["finding"] == finding)]
            source[f"{organ}_{finding}"] = panel
            for _fold, group in panel.groupby("fold", sort=True):
                false_positive, true_positive, _ = roc_curve(group["label"], group["score"])
                line, = ax.plot(false_positive, true_positive, color=style.MODEL, alpha=0.55,
                                linewidth=style.LINE_WIDTH)
            ax.plot([0, 1], [0, 1], color=style.GREY, linestyle="--", linewidth=style.LINE_WIDTH)
            value = summary.loc[(organ, finding)]
            annotation = (f"AUROC\n{value['auroc_mean']:.3f} ± {value['auroc_sd']:.3f}"
                          if summary_method == "sd" else
                          f"AUROC\n{value['auroc_mean']:.3f}\n"
                          f"95% CI {value['auroc_ci_low']:.3f} to {value['auroc_ci_high']:.3f}")
            ax.text(0.96, 0.05, annotation, transform=ax.transAxes, ha="right", va="bottom")
            ax.set_title(title.replace(" of ", " of\n"))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_box_aspect(1)
            ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
            ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
            if row == 1:
                ax.set_xlabel("False-positive rate")
            else:
                ax.tick_params(axis="x", labelbottom=False)
            if column == 0:
                ax.set_ylabel("True-positive rate")
            else:
                ax.tick_params(axis="y", left=False, labelleft=False)
    return fig, source, roc_axes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--reference-slides", type=Path, required=True)
    ap.add_argument("--folds-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--summary", choices=["sd", "t"], default="sd")
    ap.add_argument("--draft", action="store_true",
                    help="render from predictions without a complete inference record; "
                         "the provenance records the display as a draft")
    args = ap.parse_args()

    values = evaluation_data(args.predictions, args.reference_slides, None, args.folds_dir,
                             args.summary, include_baseline=False, draft=args.draft)
    style.apply()
    figure, source, roc_axes = draw(values["scored"], values["model_summary"], args.summary)
    inputs = display_inputs(args.predictions, args.reference_slides, folds_dir=args.folds_dir,
                            draft=args.draft)
    save_figure(figure, args.out_dir, "supplementary_figure_roc_curves", source, inputs,
                draft=args.draft, data_axes=roc_axes)


if __name__ == "__main__":
    main()

