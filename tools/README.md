# Tools for hyperemble

This directory contains scripts and other files which are useful for
development, but are not part of the hyperemble package itself.

## Tools

- `travis_tools.sh`: setup script to be run at the beginning of the Travis build.
- `pre-commit`: pre-commit hook for running `make test` and `make
  style`. To activate, copy into the `.git/hooks` directory. This will
  then disallow commits which do not pass the tests and style
  check. If necessary, this can be bypassed using `git commit --no-verify`.
- `bash-and-git.sh`: contains various useful git/bash
  configurations. See comments in file for more details.
