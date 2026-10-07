# Zig Example

`matrix_multiply.zig` uses the typed `zynum.blas.matrixMultiply` API.

Requires Zig 0.17.x. From the repository root:

```sh
zig build --build-file examples/zig/build.zig run
```

For a build specialized to the local CPU, including Apple M5:

```sh
zig build --build-file examples/zig/build.zig run -Dcpu=native --release=fast
```
