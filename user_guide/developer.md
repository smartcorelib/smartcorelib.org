---
layout: manual
title:  "Developer's Guide"
---

# Developer's Guide

Ready to contribute, great! There are many ways you can improve *SmartCore*. Here are some ideas for you.

* [Improve *SmartCore*'s documentation.](#changes-to-documentation)
* [Report an issue in GitHub.](#how-to-report-an-issue)
* [Contribute new or improve existing code.](#contributing-code)
* [Review pull requests.](https://github.com/smartcorelib/smartcore/pulls)
* [You can contribute to this website by completing, improving, and correcting it.](#changes-to-documentation)
* [Propose new functionality.](#how-to-request-new-feature)

## How to report an issue?

If you found a bug or problem please do not hesitate to report it by [opening an issue](https://github.com/smartcorelib/smartcore/issues) in GitHub. Opening an issue is a simple matter of describing your problem or your idea and answering all follow-up questions from developer (if any). We only ask you to follow these simple recommendations when you open a new issue:

* Please verify that your issue is not being currently addressed by other issues or pull requests.
* When you are submitting a bug report, please clearly state your problem and demonstrate it with a simple example if you can. Please attach versions of SmartCore, Rust, your operational system to your report.

## How to request a new feature?

The best way to request a new feature is by [opening an issue](https://github.com/smartcorelib/smartcore/issues) in GitHub. When you submit your idea, please keep in mind these recommendations:

* If you are requesting new algorithm, please add references to papers describing this algorithm. If you have a particular implementation in mind, feel free to share references to it as well. If not, we will do our best to find the best implementation available ourselves.
* Please tell us why this feature is important to you.

## Contributing code

Writing new code or [fixing existing issues](https://github.com/smartcorelib/smartcore/issues) is a great way to contribute. In *SmartCore* all new code is peer-reviewed by core contributors. Before writing new code, please submit an issue describing your problem or feature and give us some time to look at your proposal and suggest changes (if any). 

New code should be submitted as a [pull request](https://github.com/smartcorelib/smartcore/pulls) in GitHub. We do not have any preferences as to [Forking](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request-from-a-fork) vs [Branching](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request) when it comes to submitting your pull request. Please request to merge your change into the *development* branch.

In order to get a PR approved and merged, the CI tests should be passing. New features should be covered by unit tests and documentation. Please make sure that your changes pass the CI tests by running:

1. `cargo test` to run thes tests.
2. `cargo fmt` to format the code according to the style guidelines.
3. `cargo clippy --all-features -- -Drust-2018-idioms -Dwarnings` for code linting.

## Changes to documentation

If you found a problem in documentation please do not hesitate to correct it and submit your proposed change as a [pull request](https://github.com/smartcorelib/smartcore/pulls) (PR) in GutHub. At this moment documentation is found in several places: [API](https://github.com/smartcorelib/smartcore), [website](https://github.com/smartcorelib/smartcorelib.org) and [examples](https://github.com/smartcorelib/smartcore-examples). Please submit your pull request to a corresponding repository. If your change is a minor correction (e.g. misspelling or grammar error) there is no need to open a separate issue describing what you've found, just correct it and submit your PR!

Another way to make a change in documentation is to [open an issue](https://github.com/smartcorelib/smartcore/issues) in GitHub.
