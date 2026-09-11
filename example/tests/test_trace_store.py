"""example/trace_store.py 的單元測試:pack_key/unpack_key 互為反函式、
layer_names 依首次出現順序去重。"""
from example.trace_store import layer_names, pack_key, unpack_key


def test_pack_unpack_roundtrip():
    key = pack_key("conv1", "spike_count")
    assert key == "conv1__spike_count"
    assert unpack_key(key) == ("conv1", "spike_count")


def test_layer_names_dedupes_and_skips_non_keys():
    files = ["epochs", "conv1__spike_count", "conv1__v_final",
             "conv2__spike_count", "out__idle_frac"]
    assert layer_names(files) == ["conv1", "conv2", "out"]


def test_layer_names_empty():
    assert layer_names([]) == []
    assert layer_names(["epochs"]) == []


TESTS = [
    test_pack_unpack_roundtrip,
    test_layer_names_dedupes_and_skips_non_keys,
    test_layer_names_empty,
]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\n全部 {len(TESTS)} 項通過")
