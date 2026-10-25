#!/bin/sh
# PyTensor adds "-ld64" on macOS >= 15 to pick Apple's old linker. Newer clang
# no longer accepts it (reads it as "-l d64"), so drop it and call clang++.
for arg in "$@"; do
  shift
  [ "$arg" = "-ld64" ] && continue
  set -- "$@" "$arg"
done
exec /usr/bin/clang++ "$@"
