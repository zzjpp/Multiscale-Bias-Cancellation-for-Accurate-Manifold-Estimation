# Paper schematics

Figures 1 and 3 are explanatory drawings, not numerical experiment outputs. Editable TikZ sources and the selected PNG/PDF exports are retained.

Compile each standalone figure from its own directory with a LaTeX distribution that includes `standalone`, TikZ, `amsmath`, and `amssymb`:

```bash
cd illustrations/Figure1
pdflatex -interaction=nonstopmode -halt-on-error normal_correspondence.tex
```

```bash
cd illustrations/Figure3
pdflatex -interaction=nonstopmode -halt-on-error yy_normal_decomposition.tex
```

Each wrapper includes its local `figure_tikz.tex`; `figure_caption.tex` contains the accompanying caption. Figure 3 also retains `validate_geometry.py` and its construction checks. Rebuilding a schematic does not reproduce a numerical estimator experiment.
