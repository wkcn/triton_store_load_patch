# Triton Load/Store Patch

There is a bug in the triton library that causes the load/store instructions to be interpreted incorrectly. This patch fixes the issue.

```
bash ./build_lib.sh
```

```
TRITON_INTERPRET=1
```
