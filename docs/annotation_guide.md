# Annotation Guide

Conventions for annotating HTR ground truth. This guide is the operational reference — the
schema (`1.0.0`), the granularity decision and the reasoning behind these rules live in
[ADR 004](ADRs/004_ground_truth_schema.md). Read it when a rule here looks arbitrary, or
before proposing a change to one.

## Workflow

Open the [annotation tool](https://huggingface.co/spaces/PellelNitram/xournalpp_htr_data_annotation_tool),
enter your `annotator_id` (always the same one), open the `.xopp`, classify every stroke,
export. Put the `.gt.json` next to its source
in `data/` of the `PellelNitram/xournalpp_htr_benchmark` dataset.

- Annotate the **copy in the dataset repo** — editing a source file after annotation
  invalidates its labels.
- Auto-save is browser-local only. Export every session.
- `R` rect select · `S` stroke select · `P` pan · `+`/`-` zoom · `F` fit

## Classes

Word level, never character level. `word`, `digit` and `mathematical_expression` require
`text`; the other seven forbid it.

| | |
|---|---|
| `word` | One word, incl. later strokes (`i` dots, `t` bars) |
| `digit` | Purely numeric |
| `mathematical_expression` | One whole formula, not per symbol |
| `arrow` `drawing` `diagram` | Pointers · doodles · graphs, flowcharts |
| `table` `separator` | Grid lines · rules, dividers, underlines, bullet markers |
| `correction` | Struck-through or scribbled-out content |
| `other` | Nothing else fits |

## Rules

**Transcribe what is written, not what was meant** — keep capitalisation, diacritics,
spelling mistakes.

| Case | Rule |
|---|---|
| Punctuation | Attach to adjacent word, include in `text` (`Hallo,`) |
| Hyphenation across lines | Two `word`s, hyphen kept (`Hand-`, `schrift`) |
| Digit grouping | `123` → one; `1  2  3` → three |
| `digit` vs `word` | `digit` only if purely numeric (`-1.234,50`); any letter → `word` |
| Math | LaTeX, no delimiters (`\frac{a}{b}`) |
| Operator between words | Both sides text → own `word` (`Wort := Begriff` → `Wort`, `:=`, `Begriff`) |
| Operator beside math | One side math → operator joins the `mathematical_expression` (`x^2 = ` + `Fläche`) |
| Overwritten | One `word` over all strokes, final reading; if unreadable → `correction` |
| Illegible | `other`, never a guessed `word` — a guess makes a correct prediction score as an error |
| Underline | `separator`, never part of the `word` |
| Bullet marker | `-` `*` `•` → `separator`, own annotation, never merged into the following word |
| Diagram labels | Text annotated normally, kept out of the `diagram` |

Avoid selections spanning layers — they split per layer on export.

## Quality control

- Annotate 3 documents, revisit these rules, *then* go bulk.
- Double-annotate ~10 pages under a second `annotator_id`, else a CER can't be separated
  from annotation disagreement.
- Evaluation-only: never train on benchmark data. Check for personal information before
  publishing.
