"""pytest 根設定:把 repo 根目錄釘成 rootdir,並在 import jax 之前設好
JAX 的 GPU 記憶體選項(用多少配多少,不預先要一大塊;理由見 docs/問題紀錄.md
「決策:GPU 記憶體用 XLA_PYTHON_CLIENT_PREALLOCATE=false,不是別的選項」)。

`os.environ.setdefault` 而不是硬設——外部已經指定的話尊重外部。
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
