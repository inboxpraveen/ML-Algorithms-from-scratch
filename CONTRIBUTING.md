# Contributing

Thanks for helping. This repository teaches machine learning by showing how the algorithms
work, so the bar is correctness first, then clarity, then speed.

## Set up

You need Python 3.10 or newer and NumPy. `pytest` runs the checks.

```bash
git clone https://github.com/inboxpraveen/ML-Algorithms-from-scratch.git
cd ML-Algorithms-from-scratch

# with conda
conda create -n ml-scratch python=3.11 numpy pytest
conda activate ml-scratch

# or with venv
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install numpy pytest
```

Run one algorithm, then the whole suite:

```bash
python "6. Decision Trees/_6_decision_trees.py"
python -m pytest
```

The suite runs every file as a script, so it takes about a minute. To check a single
folder, use `python -m pytest -k "Decision"`.

## Ways to help

- Report a bug: open an issue with the algorithm, the command you ran and the output.
- Fix or clarify a guide: typos, a confusing step, a missing reference.
- Check an algorithm against the canonical version and report any difference.
- Propose a new algorithm: open an issue first so we can agree on scope.

## Rules for code and guides

1. Follow the existing style: clear, documented, educational.
2. Document every parameter with its type, default, effect and a typical range, and state
   the formula each method implements.
3. Put a plug-and-play demo at the bottom of the file under `if __name__ == "__main__":`.
   It must use NumPy only, seed with `np.random.seed(42)`, print ASCII only and finish in
   seconds.
4. Open the matching `.md` guide with a **Quick Start** whose expected output you copied
   from a real run.
5. Core code uses NumPy only. Other libraries are fine in guide examples that compare
   against a reference.
6. Validate against the canonical algorithm: scikit-learn or SciPy where an equivalent
   exists, otherwise a problem whose answer you can compute by hand.
7. Quote no number you have not measured. If your version simplifies the canonical
   algorithm, add a **Simplifications** section that says what differs and what it costs.
8. Add a row to the algorithm table in `README.md` and keep the folder naming
   (`N. Name/_N_name.py` and `N. Name/_N_name.md`).

Clarity is not an excuse for being wrong: an implementation that is clear and wrong teaches
the wrong thing.

## Pull requests

Keep each pull request to one algorithm or one topic. Run `python -m pytest` before you
open it, and describe what you checked the result against.

By contributing you agree that your work is released under the repository's
[GPL-3.0 license](LICENSE).
