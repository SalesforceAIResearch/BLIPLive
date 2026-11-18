# Contributing to BLIP-Live

Documentation corrections and focused fixes are welcome through GitHub pull requests.
Install with `pip install -e ".[dev]"` and run:

```bash
python -m unittest discover -s tests -v
ruff check src tests
ruff format --check src tests
```

Changes to scoring, frame sampling, prompts, or response parsing must preserve
the documented reference behavior, or explicitly introduce a new version.
Training and trained-checkpoint inference are planned for a future release.

## Contributor License Agreement

Contributors must sign the [Salesforce CLA](https://cla.salesforce.com/sign-cla)
before a pull request can be accepted. You only need to sign once to contribute
to Salesforce open-source projects. Contributions are licensed under the
[project license](LICENSE.txt).

Follow our [Code of Conduct](CODE_OF_CONDUCT.md). Report security concerns using
the process in [SECURITY.md](SECURITY.md).
