#!/usr/bin/env bash

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )/.."

bundle exec jekyll build -d $BASE_DIR/_site -s $BASE_DIR
