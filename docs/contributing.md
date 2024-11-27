# Contributing

Find open issues [here](https://github.com/PellelNitram/xournalpp_htr/issues).


<div align="center">

<iframe width="560" height="315" src="https://www.youtube.com/embed/dQw4w9WgXcQ?si=3xMriRxJb8TdjVui" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

<br>

<i>(<a href="https://youtu.be/dQw4w9WgXcQ?utm_source=docs&utm_medium=docs&utm_campaign=docs">Click here to get to video on YouTube.</a>)</i>

</div>

## Summary

TODO: Add video here.

## In detail

TODO: Add text here.

## Code quality

We try to keep up code quality as high as practically possible. For that reason, the following steps are implemented:

- Testing. Xournal++ HTR uses `pytest` for implementing unit, regression and integration tests.
- Linting. Xournal++ HTR uses `ruff` for linting and code best practises. `ruff` is implemented as git pre-commit hook. Since `ruff` as pre-commit hook is configured externally with `pyproject.toml`, you can use the same settings in your IDE if you wish to speed up the process.
- Formatting. Xournal++ HTR uses `ruff-format` for consistent code formatting. `ruff-format` is implemented as git pre-commit hook. Since `ruff-format` as pre-commit hook is configured externally with `pyproject.toml`, you can use the same settings in your IDE if you wish to speed up the process.