# Contributing to sEEGnal

Thank you for your interest in improving sEEGnal. Bug reports, feature ideas,
documentation suggestions, and other proposals are welcome.

All project documentation and public discussions must be written in English.

## Current contribution policy

sEEGnal accepts bug reports, feature proposals, documentation improvements,
and code contributions through GitHub Issues and pull requests. Substantial
changes should be discussed in an Issue before implementation.

## Contribution licensing

sEEGnal is distributed under the [BSD 3-Clause License](LICENSE). By submitting
a contribution, you agree that it may be distributed under that license and
confirm that you have the right to submit it. Contributions must not include
code, data, or other material whose terms are incompatible with BSD-3-Clause.

## Before opening an Issue

- Search the existing Issues to check whether the matter has already been
  reported or discussed.
- Read the main [README](README.md) and, when relevant, the
  [quickstart documentation](quickstart/README.md).
- Make sure the proposal concerns sEEGnal rather than a general question about
  MNE-Python, BIDS, Python, or EEG analysis.
- Do not upload EEG recordings, participant information, credentials, private
  paths, or any other sensitive or confidential data.

## Reporting a bug

Open a GitHub Issue and include enough information to reproduce and assess the
problem:

- a clear description of the observed behavior;
- the behavior you expected;
- the smallest sequence of steps that reproduces the problem;
- the affected sEEGnal stage or function;
- the sEEGnal version or Git commit;
- the Python version, operating system, and relevant dependency versions;
- the complete error message or traceback, after removing sensitive data; and
- whether the problem can be reproduced with the quickstart and
  non-sensitive data.

Use only non-sensitive data in a minimal reproducible example. Never upload
clinical EEG recordings or participant information to an Issue.

## Proposing a feature or other improvement

Open a GitHub Issue before implementing the change. Describe:

- the problem or use case;
- the proposed behavior;
- the part of sEEGnal that would be affected;
- possible alternatives or limitations; and
- any new dependency, file format, or compatibility requirement involved.

New functionality, dependencies, public API changes, and substantial design
decisions must be discussed with the maintainer or an authorized editor before
implementation.

## Review and discussion

The maintainer or an authorized editor will review each proposal. Discussion
may be required to define its scope, scientific assumptions, compatibility,
and validation requirements. Acceptance of an idea does not guarantee an
implementation date.

The current lead developer and maintainer is Federico Ramírez-Toraño.

## Pull request workflow

Contributors are expected to:

1. Open an Issue and wait until a maintainer or authorized editor approves the
   proposed implementation.
2. Fork the repository.
3. Create a branch named `fix/short-description` for a bug fix or
   `contribution/short-description` for any other contribution.
4. Make a focused change that addresses the approved Issue.
5. Test at least the part of the quickstart affected by the change.
6. Update the relevant docstrings and documentation.
7. Open a pull request from the contribution branch to the sEEGnal `main`
   branch and link the original Issue.
8. Address review comments. The change will be merged only after approval by
   the maintainer or an authorized editor.

Opening a pull request will not modify `main`. It only proposes that the
changes from the contribution branch be reviewed and, if accepted, merged.

## Code and documentation conventions

Contributions must follow these conventions:

- Write code, docstrings, documentation, Issues, and pull request descriptions
  in English.
- Use NumPy-style docstrings for public functions.
- Document parameters, return values, relevant units and array dimensions,
  exceptions, created files, and important object mutations when applicable.
- Qualify functions imported from another module with the module name. For
  example, use `mne_tools.prepare_eeg(...)` rather than importing and calling
  `prepare_eeg(...)` directly.
- Avoid opaque module aliases and wildcard imports.
- Keep changes consistent with the existing project style.
- Do not add or change dependencies without prior discussion and approval.
- Preserve backward compatibility unless an incompatible change has been
  explicitly discussed and approved.

## Testing expectations

At minimum, a contribution must successfully run the quickstart stage or
stages affected by the change. The core quickstart is launched from the
repository root with:

```console
python -m quickstart.run_sEEGnal
```

If a change cannot be exercised by the quickstart, its pull request must
explain how it was tested and provide a reproducible check where possible.

The package requires Python 3.11 or later. The currently documented formal
test environment is Python 3.13 on Windows; support for other declared Python
versions and operating systems should not be assumed to be formally validated
unless the contribution provides that evidence.
