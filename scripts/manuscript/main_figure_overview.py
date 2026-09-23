"""Render the study overview: the data, the reference standards, the model and the evaluation.

Reads the cohort, Reference A, the report records, the case splits and the
training configuration, and writes the display ``main_figure_overview`` as
PDF and PNG with the source data and a provenance record. The figure has four
parts: a, the data sources and the study cohort; b, the two reference
standards; c, the model; d, the evaluation, drawn as branches from the model
scores to each design, its reference standard and its analysis.

Every number in the figure is read from the inputs: the cohort counts, the
number of tiles and the feature dimension of the example slide, the tile size
and magnification, the embedding size of the aggregator, the folds, the
held-out sites, the compared aggregators and encoders, the training fractions,
and the scale bar, which follows from the pixel size of the slide. The small
plots in part d are icons of the analyses and show no data.

The example slide of parts b and c is chosen by a fixed rule: a lung slide of
the study cohort, scanned at the central laboratory, that Reference A marks
positive for bronchopneumonia with the severity "severe" and a recorded
extent, and whose supporting record quotes the finding, the severity and the
extent. Among these the slide with the shortest quote is taken, and then the
first slide identifier. Its tiles are the most strongly stained tiles of the
slide (``figures.slides.stained_tiles``).
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from champs_pipeline.data_prep.cohort import LABEL_SOURCES, UNIT
from champs_pipeline.figures import flow, schematic, slides, style
from champs_pipeline.figures.flow import Box
from champs_pipeline.figures.labels import (AGGREGATORS, ENCODERS, FINDINGS, ORGANS, SITES,
                                           site_code)
from champs_pipeline.figures.output import save_figure

# Height of the drawing canvas; the saved figure is cropped to the drawn content.
CANVAS_HEIGHT_MM = 250
GAP_MM = 3
PANEL_GAP_MM = 7
HEADING_HEIGHT_MM = 6
LETTER_WIDTH_MM = 3.5
ARROW_MM = 4
ARROW_GAP_MM = 5.5
NEUTRAL_FILL = style.tint(style.GREY, 0.22)

DATA_WIDTH_MM = 70
REFERENCE_LEFT_MM = 76
REFERENCE_A_WIDTH_MM = 54
ICON_HEIGHT_MM = 10
REPORT_HEIGHT_MM = 16
SWATCH_MM = 2.5
SLIDE_GLYPH_WIDTH_MM = 14
SLIDE_GLYPH_HEIGHT_MM = 5.5
READERS_HEIGHT_MM = 6
GLYPH_COLUMN_MM = 20

THUMBNAIL_PIXELS = 1200
THUMBNAIL_WIDTH_MM = 32
SCALE_BAR_TISSUE_MM = 2
ZOOM_SOURCE_MM = 2.5
ZOOM_LENGTH_MM = 9
EXAMPLE_TILES = 4
TILE_MM = 11
TILE_OFFSET_MM = 1.8
ENCODER_WIDTH_MM = 16
ENCODER_HEIGHT_MM = 18
ENCODER_TAPER_MM = 4
CELL_MM = 1.6
MATRIX_ROWS = 8
MATRIX_COLUMNS = 3
AGGREGATOR_WIDTH_MM = 30

# Part d: branches from the model scores. Horizontal positions in millimetres.
MODEL_CHIP_WIDTH_MM = 20
TRUNK_X_MM = 24
DESIGN_LEFT_MM = 28
DESIGN_WIDTH_MM = 52
BRANCH_REFERENCE_LEFT_MM = 74
ICON_LEFT_MM = 101
ANALYSIS_LEFT_MM = 119
BRANCH_GAP_MM = 4
MARKER_MM = 5
MARKER_GAP_MM = 0.8
REFERENCE_CHIP_MM = 20
ICON_WIDTH_MM = 13
ICON_PLOT_HEIGHT_MM = 8
SPLIT_SHADES = ("#DADADA", "#595959")
NUMBER_WORDS = {2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight",
                9: "Nine", 10: "Ten"}

EXAMPLE_ORGAN = "lung"
EXAMPLE_FINDING = "bronchopneumonia"
EXAMPLE_SEVERITY = "severe"
EXAMPLE_SLIDE_SOURCE = "CPL"
CORRECTED = "elig"


def lower_first(text):
    return text[:1].lower() + text[1:]


def quotes_example(record):
    quote = record["finding_examples"].lower()
    return (EXAMPLE_FINDING in quote and record["severity"] in quote
            and record["extent"] in quote)


def example_slide(cohort_slides, reference, findings):
    """The example slide, its Reference A severity and extent, and its supporting quote."""
    labels = reference[(reference["variant"] == CORRECTED)
                       & (reference["finding"] == EXAMPLE_FINDING)
                       & (reference["label"] == 1)
                       & (reference["severity"] == EXAMPLE_SEVERITY)
                       & reference["extent"].notna()
                       & (reference["slide_source"] == EXAMPLE_SLIDE_SOURCE)]
    study = cohort_slides[cohort_slides["training"]
                          & (cohort_slides["organ_group"] == EXAMPLE_ORGAN)]
    candidates = labels.merge(study[["slide_id", "wsi_path", "feature_path_virchow2"]],
                              on="slide_id")
    records = findings[(findings["semantic_group"] == EXAMPLE_FINDING)
                       & (findings["kind"] == "condition")
                       & findings["text_source"].isin(LABEL_SOURCES[EXAMPLE_SLIDE_SOURCE])]
    candidates = candidates.merge(records[[*UNIT, "finding_examples"]], on=UNIT)
    candidates = candidates[candidates.apply(quotes_example, axis=1)]
    if candidates.empty:
        raise ValueError("no slide meets the rule for the example slide")
    candidates = candidates.assign(quote_length=candidates["finding_examples"].str.len())
    return candidates.sort_values(["quote_length", "slide_id"]).iloc[0]


def draw_heading(ax, left, top, letter, title):
    """Draw the bold part letter and the title of a part; return the top of its content."""
    ax.text(left, top, letter, fontweight="bold", ha="left", va="top")
    ax.text(left + LETTER_WIDTH_MM, top, title, ha="left", va="top")
    return top - HEADING_HEIGHT_MM


def labelled_arrow(ax, x, start_y, end_y, label):
    """A downward arrow with its label on the right."""
    flow.arrow(ax, (x, start_y), (x, end_y))
    ax.text(x + 1.5, (start_y + end_y) / 2, label, va="center")


def draw_data(ax, left, top, width, funnel, thumbnail):
    """Part a: the three data sources, the study cohort and the sites."""
    examined = funnel["report_arm"][-1]
    archive = funnel["slide_arm"][0]
    study = funnel["slide_arm"][-1]
    content_top = draw_heading(ax, left, top, "a", "Data and study cohort")

    column_width = (width - 2 * GAP_MM) / 3
    columns = [left + index * (column_width + GAP_MM) for index in range(3)]
    icon_top = content_top - 1
    page = schematic.document(ax, columns[0] + column_width / 2 - 4, icon_top, 8, ICON_HEIGHT_MM)
    schematic.text_lines(ax, page, [0.8, 0.55, 0.8, 0.45], first_offset=3.5, spacing=1.8)
    schematic.glass_slide(ax, columns[1] + column_width / 2 - 8, icon_top - 1.5, 16, 7,
                          picture=thumbnail)
    schematic.table_glyph(ax, columns[2] + column_width / 2 - 6, icon_top - 1, 12, 8)

    label_top = icon_top - ICON_HEIGHT_MM - 1.5
    labels = [
        ("Pathology reports", f"{examined['n']:,} cases"),
        ("Slide images", f"{archive['n']:,} slides"),
        ("Release tables", "Demographics, causes of death"),
    ]
    bottoms = []
    for column_left, (title, text) in zip(columns, labels):
        bottoms.append(flow.text_block(ax, column_left, label_top, column_width, text,
                                       centered=True, title=title))
    cohort = flow.box(ax, left, min(bottoms) - 2 * GAP_MM, width,
                      f"{study['n']:,} slides ({study['n_lung']:,} lung, "
                      f"{study['n_liver']:,} liver)\n{study['n_cases']:,} cases from "
                      f"{len(SITES)} sites", title="Study cohort", facecolor=NEUTRAL_FILL,
                      rounded=True)
    for column_left, bottom in zip(columns, bottoms):
        center = column_left + column_width / 2
        flow.arrow(ax, (center, bottom - 0.5), (center, cohort.top))
    return flow.text_block(ax, left, cohort.bottom - 1.5, width, ", ".join(SITES.values()),
                           centered=True)


def draw_reference_a(ax, left, top, width, example):
    """Reference A: a report quote, the extracted record and the label per slide."""
    colour = style.REFERENCE_A
    title_bottom = flow.text_block(ax, left, top, width,
                                   title="Report labels: from the reports")
    page = schematic.document(ax, left, title_bottom - 1.5, width, REPORT_HEIGHT_MM)
    schematic.text_lines(ax, page, [0.85, 0.6], first_offset=3.2, spacing=2.2)
    quote = f"“{example['finding_examples']}”"
    quote_y = page.top - 8.6
    highlight = Box(page.left + 1.4, quote_y + 2.1, flow.text_width_mm(ax, quote) + 1.6, 4.2)
    schematic.rectangle(ax, highlight, edgecolor="none", facecolor=style.tint(colour, 0.45),
                        zorder=1.5)
    ax.text(page.left + 2.2, quote_y, quote, va="center", zorder=3)
    schematic.text_lines(ax, page, [0.7], first_offset=13.2)

    finding = FINDINGS[EXAMPLE_ORGAN][EXAMPLE_FINDING]
    record = flow.box(ax, left, page.bottom - ARROW_GAP_MM, width,
                      f"Finding: {finding}\nSeverity: {example['severity']}\n"
                      f"Extent: {example['extent']}", title="Extracted record", centered=False,
                      edgecolor=colour, facecolor=style.tint(colour), rounded=True)
    labelled_arrow(ax, left + 6, page.bottom, record.top, "Structured extraction")

    legend_top = record.bottom - ARROW_GAP_MM
    labelled_arrow(ax, left + 6, record.bottom, legend_top, "Label rules")
    title_bottom = flow.text_block(ax, left, legend_top - 0.5, width,
                                   title="Label per slide and finding")
    swatch_top = title_bottom - 1.2
    x = schematic.swatch(ax, left, swatch_top, SWATCH_MM, "Positive", facecolor=colour,
                         edgecolor=colour)
    x = schematic.swatch(ax, x + 3, swatch_top, SWATCH_MM, "Negative", facecolor=style.WHITE)
    schematic.swatch(ax, x + 3, swatch_top, SWATCH_MM, "Masked", facecolor=style.WHITE,
                     edgecolor=style.GREY, hatch="////")
    return swatch_top - SWATCH_MM


def draw_reference_b(ax, left, top, width):
    """Reference B: a stratified sample of slides, blinded reads and a consensus."""
    colour = style.REFERENCE_B
    title_bottom = flow.text_block(ax, left, top, width, title="Blinded reads")
    text_left = left + GLYPH_COLUMN_MM
    text_width = width - GLYPH_COLUMN_MM

    stack_top = title_bottom - 1.5
    for index in range(3):
        schematic.glass_slide(ax, left + index * 1.2, stack_top - index * 1.2,
                              SLIDE_GLYPH_WIDTH_MM, SLIDE_GLYPH_HEIGHT_MM,
                              tissue=style.tint(style.GREY, 0.5), zorder=1.0 + index)
    sample_bottom = flow.text_block(ax, text_left, stack_top, text_width,
                                    "Lung slides sampled by model score and report status")
    stack_bottom = min(stack_top - 2 * 1.2 - SLIDE_GLYPH_HEIGHT_MM, sample_bottom)

    readers_top = stack_bottom - ARROW_GAP_MM
    readers = schematic.group_of_people(ax, left + 1, readers_top - READERS_HEIGHT_MM,
                                        READERS_HEIGHT_MM, colour)
    reads_bottom = flow.text_block(ax, text_left, readers_top, text_width,
                                   "Independent blinded reads by pathologists")
    flow.arrow(ax, (left + 8, stack_bottom), (left + 8, readers_top))

    card_top = min(readers.bottom, reads_bottom) - ARROW_GAP_MM
    card = flow.box(ax, left, card_top, width, "Presence and grade per finding",
                    title="Consensus", edgecolor=colour, facecolor=style.tint(colour),
                    rounded=True)
    flow.arrow(ax, (left + 8, min(readers.bottom, reads_bottom)), (left + 8, card.top))
    return card.bottom


def draw_references(ax, left, top, width, example):
    """Part b: Reference A from the report text, and Reference B from blinded reads."""
    content_top = draw_heading(ax, left, top, "b", "Reference standards")
    bottom_a = draw_reference_a(ax, left, content_top, REFERENCE_A_WIDTH_MM, example)
    b_left = left + REFERENCE_A_WIDTH_MM + 2 * GAP_MM
    bottom_b = draw_reference_b(ax, b_left, content_top, left + width - b_left)
    return min(bottom_a, bottom_b)


def draw_model(ax, left, top, width, picture, model):
    """Part c: slide, tiles, frozen encoder, features, aggregator and one score per finding."""
    content_top = draw_heading(ax, left, top, "c", "Model, one per organ")
    thumbnail = picture["thumbnail"]
    thumbnail_height = THUMBNAIL_WIDTH_MM * thumbnail.height / thumbnail.width
    slide_box = schematic.image(ax, thumbnail, left, content_top, THUMBNAIL_WIDTH_MM,
                                thumbnail_height)
    tissue_width_mm = picture["width_pixels"] * picture["microns_per_pixel"] / 1000
    bar_length = SCALE_BAR_TISSUE_MM * THUMBNAIL_WIDTH_MM / tissue_width_mm
    schematic.scale_bar(ax, left + 1, slide_box.bottom - 1.2, bar_length,
                        f"{SCALE_BAR_TISSUE_MM} mm")
    arrow_y = slide_box.middle_y

    front_x, front_y, _ = picture["tiles"][-1]
    source_center_x = left + front_x / picture["width_pixels"] * THUMBNAIL_WIDTH_MM
    source_center_y = content_top - front_y / picture["height_pixels"] * thumbnail_height
    source = Box(source_center_x - ZOOM_SOURCE_MM / 2, source_center_y + ZOOM_SOURCE_MM / 2,
                 ZOOM_SOURCE_MM, ZOOM_SOURCE_MM)
    schematic.rectangle(ax, source, zorder=3)
    stack_depth = (len(picture["tiles"]) - 1) * TILE_OFFSET_MM
    stack = schematic.tile_stack(ax, [tile for _, _, tile in picture["tiles"]],
                                 slide_box.right + ZOOM_LENGTH_MM,
                                 arrow_y + (TILE_MM + stack_depth) / 2, TILE_MM, TILE_OFFSET_MM)
    schematic.zoom(ax, source, stack)
    tiles_bottom = flow.text_block(ax, stack.left - 4, stack.bottom - 1.2, stack.width + 8,
                                   f"{model['tile_pixels']} px at {model['magnification']}×",
                                   centered=True, title=f"{model['tiles']:,} tiles")

    encoder = schematic.tapering_block(ax, stack.right + ARROW_MM,
                                       arrow_y + ENCODER_HEIGHT_MM / 2, ENCODER_WIDTH_MM,
                                       ENCODER_HEIGHT_MM, ENCODER_TAPER_MM, style.MODEL,
                                       ENCODERS[model["encoder"]])
    flow.arrow(ax, (stack.right + 0.5, arrow_y), (encoder.left, arrow_y))
    ax.text(encoder.center_x, encoder.top + 1.2, "Frozen encoder", ha="center", va="bottom")

    matrix = schematic.cells(ax, encoder.right + ARROW_MM, arrow_y + MATRIX_ROWS * CELL_MM / 2,
                             CELL_MM, MATRIX_ROWS, MATRIX_COLUMNS, style.MODEL)
    flow.arrow(ax, (encoder.right, arrow_y), (matrix.left, arrow_y))
    matrix_bottom = flow.text_block(ax, matrix.center_x - 12, matrix.bottom - 1.2, 24,
                                    f"{model['tiles']:,} × {model['feature_dimension']:,}",
                                    centered=True)

    aggregator = flow.box(ax, matrix.right + ARROW_MM, arrow_y, AGGREGATOR_WIDTH_MM,
                          "Attention pooling per finding", title=AGGREGATORS[model["aggregator"]],
                          middle_y=arrow_y, edgecolor=style.MODEL,
                          facecolor=style.tint(style.MODEL), rounded=True)
    flow.arrow(ax, (matrix.right, arrow_y), (aggregator.left, arrow_y))

    vector = schematic.cells(ax, aggregator.right + ARROW_MM,
                             arrow_y + MATRIX_ROWS * CELL_MM / 2, CELL_MM, MATRIX_ROWS, 1,
                             style.MODEL)
    flow.arrow(ax, (aggregator.right, arrow_y), (vector.left, arrow_y))
    vector_bottom = flow.text_block(ax, vector.center_x - 8, vector.bottom - 1.2, 16,
                                    f"{model['embedding_dimension']:,}", centered=True)

    output_left = vector.right + ARROW_MM
    output_width = left + width - output_left
    output = flow.box(ax, output_left, arrow_y, output_width,
                      "\n".join(FINDINGS[EXAMPLE_ORGAN].values()),
                      title=f"{ORGANS[EXAMPLE_ORGAN]} model: score per finding",
                      centered=False, middle_y=arrow_y, edgecolor=style.MODEL,
                      facecolor=style.tint(style.MODEL), rounded=True)
    flow.arrow(ax, (vector.right, arrow_y), (output.left, arrow_y))
    liver = " and ".join(lower_first(name) for name in FINDINGS["liver"].values())
    liver_bottom = flow.text_block(ax, output.left, output.bottom - 1.5, output_width,
                                   f"The {ORGANS['liver'].lower()} model scores {liver}")
    return min(slide_box.bottom - 5, tiles_bottom, matrix_bottom, vector_bottom, encoder.bottom,
               aggregator.bottom, liver_bottom)


def fold_glyph(n_folds):
    """A function that draws numbered folds with the first fold as the test fold."""
    labels = [str(fold) for fold in range(1, n_folds + 1)]

    def draw(ax, left, top):
        return schematic.numbered_squares(ax, left, top, MARKER_MM, MARKER_GAP_MM, labels, 0,
                                          SPLIT_SHADES)
    return draw


def site_glyph(sites):
    """A function that draws one marker per site with the first site held out."""
    def draw(ax, left, top):
        return schematic.labelled_circles(ax, left, top, MARKER_MM, MARKER_GAP_MM, sites, 0,
                                          SPLIT_SHADES)
    return draw


def reader_glyph(ax, left, top):
    """The reader glyph of the reading substudy."""
    height = MARKER_MM + 0.5
    return schematic.group_of_people(ax, left, top - height, height, style.REFERENCE_B)


def draw_branch(ax, left, top, width, branch):
    """One branch: the design with its glyph, the reference standard, the analysis and its icon.

    Returns the height at which the branch arrow enters the design, and the bottom of the branch.
    """
    design_left = left + DESIGN_LEFT_MM
    label_bottom = flow.text_block(ax, design_left, top, DESIGN_WIDTH_MM, branch["design"])
    glyph = branch["glyph"](ax, design_left, label_bottom - 1)
    arrow_y = glyph.middle_y

    reference_left = left + BRANCH_REFERENCE_LEFT_MM
    flow.arrow(ax, (glyph.right + 2, arrow_y), (reference_left, arrow_y))
    label, colour = branch["reference"]
    chip = flow.box(ax, reference_left, arrow_y, REFERENCE_CHIP_MM, label, edgecolor=colour,
                    facecolor=style.tint(colour), rounded=True, padding=1, middle_y=arrow_y)

    icon = branch["icon"](ax, left + ICON_LEFT_MM, arrow_y + ICON_PLOT_HEIGHT_MM / 2,
                          ICON_WIDTH_MM, ICON_PLOT_HEIGHT_MM, colour)
    analysis_left = left + ANALYSIS_LEFT_MM
    analysis = ax.text(analysis_left, arrow_y,
                       flow.wrap(ax, branch["analysis"], left + width - analysis_left),
                       va="center", linespacing=flow.LINE_SPACING)
    analysis_bottom = arrow_y - flow.text_height_mm(ax, analysis) / 2
    return arrow_y, min(glyph.bottom, chip.bottom, icon.bottom, analysis_bottom)


def draw_evaluation(ax, left, top, width, splits, matrix):
    """Part d: branches from the model scores to the designs, references and analyses."""
    content_top = draw_heading(ax, left, top, "d", "Evaluation")
    lung = splits[splits["organ"] == EXAMPLE_ORGAN]
    n_folds = lung.loc[lung["design"] == "fivefold", "fold"].nunique()
    held_out = lung[(lung["design"] == "loso_nested") & (lung["split"] == "test")]
    held_out_codes = set(held_out.groupby("fold")["site"].agg(lambda s: site_code(s.iloc[0])))
    if held_out_codes != set(SITES):
        raise ValueError(f"held-out sites differ from the study sites: {sorted(held_out_codes)}")
    sites = list(SITES)

    families = matrix["families"]
    encoders = [matrix["defaults"]["encoder"], *families["encoder_comparison"]["encoder"]]
    aggregators = families["aggregator_comparison"]["aggregator"]
    fractions = families["learning_curve"]["train_fraction"]

    branches = [
        {"design": f"{NUMBER_WORDS[n_folds]}-fold cross-validation",
         "glyph": fold_glyph(n_folds), "reference": ("Report labels", style.REFERENCE_A),
         "analysis": "AUROC and average precision", "icon": schematic.curve_icon},
        {"design": "Site-held-out validation",
         "glyph": site_glyph(sites), "reference": ("Report labels", style.REFERENCE_A),
         "analysis": "Performance at each held-out site", "icon": schematic.forest_icon},
        {"design": "Reading substudy of lung slides", "glyph": reader_glyph,
         "reference": ("Blinded reads", style.REFERENCE_B),
         # A non-breaking space keeps the name of the reference on one line.
         "analysis": "Agreement among the readers, and the report labels against the consensus",
         "icon": schematic.bar_icon},
    ]
    arrow_heights = []
    branch_top = content_top
    for branch in branches:
        arrow_y, bottom = draw_branch(ax, left, branch_top, width, branch)
        arrow_heights.append(arrow_y)
        branch_top = bottom - BRANCH_GAP_MM

    middle = arrow_heights[len(arrow_heights) // 2]
    chip = flow.box(ax, left, middle, MODEL_CHIP_WIDTH_MM, "Model scores", edgecolor=style.MODEL,
                    facecolor=style.tint(style.MODEL), rounded=True, padding=1, middle_y=middle)
    trunk_x = left + TRUNK_X_MM
    flow.line(ax, [(chip.right, middle), (trunk_x, middle)])
    flow.line(ax, [(trunk_x, arrow_heights[0]), (trunk_x, arrow_heights[-1])])
    for arrow_y in arrow_heights:
        flow.arrow(ax, (trunk_x, arrow_y), (left + DESIGN_LEFT_MM - 1, arrow_y))

    comparisons_bottom = flow.text_block(
        ax, left + DESIGN_LEFT_MM, branch_top + BRANCH_GAP_MM / 2, width - DESIGN_LEFT_MM,
        f"{len(aggregators)} aggregators, {len(encoders)} encoders, raw against corrected "
        f"labels, and {min(fractions):.0%} to {max(fractions):.0%} of the training cases",
        title=f"Comparisons in {NUMBER_WORDS[n_folds].lower()}-fold cross-validation")

    source = pd.DataFrame({
        "quantity": ["folds", "held-out sites", "aggregators", "encoders", "training fractions"],
        "value": [n_folds, ";".join(sites), ";".join(aggregators), ";".join(encoders),
                  ";".join(map(str, fractions))],
    })
    return comparisons_bottom, source


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort-dir", type=Path, required=True)
    ap.add_argument("--reference-slides", type=Path, required=True,
                    help="reference_a/reference_a_slide.parquet")
    ap.add_argument("--findings", type=Path, required=True, help="the report records")
    ap.add_argument("--case-splits", type=Path, required=True, help="folds/case_splits.csv")
    ap.add_argument("--training-configs", type=Path, required=True, help="configs/training")
    ap.add_argument("--slide-root", type=Path, required=True,
                    help="directory that holds the whole-slide image archive")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    funnel_path = args.cohort_dir / "funnel.json"
    slides_path = args.cohort_dir / "cohort_slides.csv"
    matrix_path = args.training_configs / "matrix.yaml"
    funnel = json.loads(funnel_path.read_text())
    cohort_slides = pd.read_csv(slides_path, low_memory=False,
                                usecols=["slide_id", "organ_group", "training", "wsi_path",
                                         "feature_path_virchow2"])
    reference = pd.read_parquet(args.reference_slides,
                                columns=["slide_id", *UNIT, "finding", "variant", "label",
                                         "severity", "extent"])
    findings = pd.read_parquet(args.findings,
                               columns=[*UNIT, "text_source", "semantic_group", "kind",
                                        "finding_examples"])
    splits = pd.read_csv(args.case_splits)
    matrix = yaml.safe_load(matrix_path.read_text())
    aggregator = matrix["defaults"]["aggregator"]
    aggregator_path = args.training_configs / "models" / f"{aggregator}.yaml"
    aggregator_config = yaml.safe_load(aggregator_path.read_text())

    example = example_slide(cohort_slides, reference, findings)
    slide_path = args.slide_root / example["wsi_path"]
    feature_path = Path(example["feature_path_virchow2"])
    width_pixels, height_pixels = slides.slide_dimensions(slide_path)
    picture = {
        "thumbnail": slides.thumbnail(slide_path, THUMBNAIL_PIXELS),
        "tiles": slides.stained_tiles(slide_path, feature_path, EXAMPLE_TILES),
        "width_pixels": width_pixels,
        "height_pixels": height_pixels,
        "microns_per_pixel": slides.microns_per_pixel(slide_path),
    }
    model = slides.feature_description(feature_path)
    model.update({
        "encoder": matrix["defaults"]["encoder"],
        "aggregator": aggregator,
        "embedding_dimension": aggregator_config["architecture"]["hidden_dim"],
    })

    style.apply()
    fig, ax = flow.canvas(style.FULL_WIDTH_MM, CANVAS_HEIGHT_MM)
    top = CANVAS_HEIGHT_MM - 1
    data_bottom = draw_data(ax, 0, top, DATA_WIDTH_MM, funnel, picture["thumbnail"])
    references_bottom = draw_references(ax, REFERENCE_LEFT_MM, top,
                                        style.FULL_WIDTH_MM - REFERENCE_LEFT_MM, example)
    model_top = min(data_bottom, references_bottom) - PANEL_GAP_MM
    model_bottom = draw_model(ax, 0, model_top, style.FULL_WIDTH_MM, picture, model)
    evaluation_bottom, evaluation_source = draw_evaluation(
        ax, 0, model_bottom - PANEL_GAP_MM, style.FULL_WIDTH_MM, splits, matrix)
    flow.crop(fig, ax, evaluation_bottom)

    study = funnel["slide_arm"][-1]
    source_data = {
        "a": pd.DataFrame({
            "quantity": ["sites", "cases with an examined report", "slides in the archive",
                         "study cohort slides", "study cohort lung slides",
                         "study cohort liver slides", "study cohort cases"],
            "value": [len(SITES), funnel["report_arm"][-1]["n"], funnel["slide_arm"][0]["n"],
                      study["n"], study["n_lung"], study["n_liver"], study["n_cases"]],
        }),
        "b": pd.DataFrame([{"slide_id": example["slide_id"], "finding": EXAMPLE_FINDING,
                            "severity": example["severity"], "extent": example["extent"],
                            "quote": example["finding_examples"]}]),
        "c": pd.DataFrame([{"slide_id": example["slide_id"], "tile_x": x, "tile_y": y,
                            "microns_per_pixel": picture["microns_per_pixel"], **model}
                           for x, y, _ in picture["tiles"]]),
        "d": evaluation_source,
    }
    inputs = [funnel_path, slides_path, args.reference_slides, args.findings,
              args.case_splits, matrix_path, aggregator_path, feature_path]
    save_figure(fig, args.out_dir, "main_figure_overview", source_data, inputs)


if __name__ == "__main__":
    main()
