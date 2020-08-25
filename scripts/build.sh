#!/usr/bin/env bash

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )/.."

pushd $1
cargo doc --no-deps --target-dir $BASE_DIR/api
popd

jekyll build -d $BASE_DIR/_site