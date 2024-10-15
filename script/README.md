# Quantitative evaluation for comparing motion magnification methods

Many motion magnification methods train their models using the training data proposed by ["Oh, Tae-Hyun, et al., "Learning-based video motion magnification"](https://arxiv.org/abs/1804.02684), ECCV, 2018", but the evaluation data for quantitative assessment presented in that paper has not been made publicly available.

Therefore, we release the evaluation dataset for quantitative comparison of motion magnification methods, strictly following the methods presented in that paper.

# Evaluation dataset information for traditional (generic) motion magnification methods

The structure of the traditional (generic) evaluation dataset is as follows:

    │  Traditional (generic) evaluation dataset
    │   ├── subpixel test
    │   │   ├── mode00: 0.040 (pixel)
    │   │   ├── mode01: 0.050 (pixel)
    │   │   ⋮
    │   │   ├── mode14: 1.000 (pixel)
    │   ├── noise005 test
    │   │   ├── mode00: 0.010 (level of additive noise)
    │   │   ├── mode01: 0.016 (level of additive noise)
    │   │   ⋮
    │   │   ├── mode20: 100.0 (level of additive noise)

