"""壓縮容量旋鈕動態放大機制(`example/train_conv_compressed.py`)的測試。沿用真實
N-MNIST 小規模資料(跟 configs/conv/compressed_smoke.yaml 同量級:
max_events=2000、train_size=16、val_size=8),不用合成資料;不碰
train_conv_compressed.py 本身,只呼叫它公開的函式/`train()` entrypoint。

step 4b 起,每個 conv 層有兩個會出界的容量:壓縮佇列長度 `L`、輸出 spike
上界 `max_out_spikes`。兩者同一套機制:偵測 -> 該層 `grown_to_fit` 放大 ->
退 checkpoint -> 重編譯續練。conv1 的 `L` 也走這套(不再像舊版那樣「conv1
出界直接 raise」)。放大公式:`new = ceil(max(observed, old) * grow_factor)`,
每個旋鈕各自一個 grow_factor。

**校準說明**:B/C/D 類測試要精準命中「第一個 epoch 就出界」「存過 checkpoint
之後才出界」「連續出界兩次」「最後一個 epoch 才出界」這些邊界,用到的
conv2_L_init/seed 是實際跑校準量出來的(固定 seed_train=0/train_size=16/
batch_size=4,用夠大的容量不截斷任何東西,記錄每個 (epoch,batch) 真正的
max_real_queue):

- seed=42:epoch0 四個 batch 的 conv2 max_real_queue 約 874/1050/1004/847,
  epoch0-3 全域最大約 1050,最大值出現在早期——這個 seed 沒有「晚期 epoch
  超過早期」的自然案例。
- seed=1:epoch0≈[933,943,1011,780]、epoch1≈[921,1016,952,836]——epoch1 的
  第 2 個 batch(≈1016)超過 epoch0 的最大值(1011),拿來測「checkpoint
  存過之後才出界」。

**這些觀察值有 ±數個單位的自然漂移**,所以測試**不硬編碼觀察值**:
conv2_L_init/seed 挑成能穩定觸發目標邊界(留了 margin),但「放大到多少」
一律用 `_parse_overflows` 從實際印出的 `[出界]` 訊息解析 observed/new,再驗證
`new == ceil(max(observed, old) * grow_factor)`——測的是機制,不是會漂的數字。

**比對兩次獨立 `train()` 結果為什麼用 `atol` 容差而不是逐位元相等**
(`docs/問題紀錄.md` 第八節):GPU 上 `jnp.sum` 等規約運算的浮點加法不滿足
結合律,XLA 可自由選規約順序——同一顆 seed、同一份 config,只要是「另一次
獨立的 `jax.jit` 編譯」,結果就有 float32 機器精度等級的雜訊(實測
w_conv1/w_conv2/w_fc 兩次獨立訓練後 max|diff| 落在 1e-7~5e-7,loss/val_acc
到小數點後 4 位一致)。沿用 `atol=1e-4`。反過來,checkpoint 存讀 round-trip、
整數計數(驅動出界判斷的 max_real_queue)、epoch 編號、容量欄位這些不牽涉
浮點規約重算的,一律維持逐位元/逐值精確比對。
"""
import contextlib
import csv
import io
import math
import os
import re
import shutil

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml

from salt_core.layers import LayerDiag
from example.models.conv_net import ConvNetCompressed, build_network
from example.paths import CONFIGS_DIR, EXPERIMENTS_DIR
from example.checkpoint import Checkpointer
from example.train_conv_compressed import train
from example.utils import TRACES_DIRNAME, TRAIN_DIRNAME

# 這份 e2e 測試會呼叫真正的 train(),每個案例吐一個 conv_compressed_*_<時間戳>
# 目錄。全部關進 experiments/TEST_TEMP,而且「一次只留最後一批」——模組載入
# (= 這一輪 pytest)開頭就把上一輪的整包清掉,不再無限累積。
_TEST_TEMP = os.path.join(EXPERIMENTS_DIR, "TEST_TEMP")
shutil.rmtree(_TEST_TEMP, ignore_errors=True)
os.makedirs(_TEST_TEMP, exist_ok=True)

_TMP_DIR = os.path.join(_TEST_TEMP, "_configs")
os.makedirs(_TMP_DIR, exist_ok=True)

_TRAIN_RESULT_TOL = 1e-4


def _assert_params_close(params_a, params_b, msg_prefix: str) -> None:
    # params 是對齊 layer list 的位置 tuple(一層一份權重陣列)。
    assert len(params_a) == len(params_b)
    for i, (a, b) in enumerate(zip(params_a, params_b)):
        max_diff = float(jnp.max(jnp.abs(a - b)))
        assert max_diff <= _TRAIN_RESULT_TOL, (
            f"{msg_prefix}:層[{i}] max|Δ|={max_diff:.3e} 超過容差 {_TRAIN_RESULT_TOL:.0e}")


def _base_cfg(run_name: str, seed: int, conv2_L_init: int, grow: float, epochs: int,
              conv1_L_init: int = 185, conv1_max_out_init: int = 8000,
              conv2_max_out_init: int = 35000, train_size: int = 16, val_size: int = 8) -> dict:
    """跟 compressed_smoke.yaml 同量級的骨架(config 是 model.layers list 形式,
    見 example/models/conv_net.py build_network),只留這份測試真正要調整的欄位當
    參數。`grow` 一次設定全部四個容量旋鈕的放大倍率(測試從沒需要它們互不
    相同)。`conv1_max_out_init` / `conv2_max_out_init` 預設給足(seed 1..42
    小規模不出界),E 類測試再覆寫成小值。"""
    def _conv(oc, L, max_out):
        return {"type": "conv", "oc": oc, "k": 3, "s": 2, "p": 1,
                "tau": 16.0, "v_th": 1.0, "alpha": 2.0, "chunk_size": 1,
                "L": L, "max_out_spikes": max_out, "init_k": 8.0 if oc == 8 else 64.0,
                "L_grow_factor": grow, "out_grow_factor": grow}
    return {
        "run_name": run_name,
        "model": {
            "decoder": "membrane_regression",
            "input_shape": [2, 34, 34],
            "layers": [
                _conv(8, conv1_L_init, conv1_max_out_init),
                _conv(16, conv2_L_init, conv2_max_out_init),
                {"type": "fc", "name": "out", "n_out": 10, "tau": 16.0,
                 "v_th": 1.0e9, "alpha": 2.0, "chunk_size": 512, "init_k": 5.0},
            ],
        },
        "data": {
            "max_events": 2000, "train_size": train_size, "val_size": val_size,
            "seed_train": 0, "seed_val": 0,
        },
        "train": {
            "lr": 1.0e-2, "epochs": epochs, "batch_size": 4, "seed": seed,
        },
    }


def _write_yaml(cfg: dict) -> str:
    path = os.path.join(_TMP_DIR, f"{cfg['run_name']}.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)
    return path


def _run_capture(config_path: str):
    """跑 `train(config_path)`,同時把訓練過程的 print(含 `[出界]` 訊息)整段
    接下來——B/C 類測試要驗證「出界訊息真的印了幾次、內容對不對」。"""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = train(config_path, exp_root=_TEST_TEMP)
    return result, buf.getvalue()


def _read_metrics_csv(exp_dir: str) -> list:
    with open(os.path.join(exp_dir, TRAIN_DIRNAME, "metrics.csv"), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


_OVERFLOW_BLOCK_RE = re.compile(r"\[出界\] epoch=(\d+) batch=(\d+): (.+)\n((?:  .+\n?)*)")
_KNOB_RE = re.compile(r"  (conv\d) (L|max_out|max_steps) (\d+)->(\d+)\(觀察 (\d+)\)")


def _parse_overflows(stdout: str) -> list[dict]:
    """從 stdout 抓出每一次 `[出界]` 事件,解析成結構化紀錄。每筆帶
    `epoch`/`batch`/`had_checkpoint` 跟一個 `knobs` list:每個被放大的旋鈕的
    (layer, knob, old, new, observed)。`[出界]` 那行只有 epoch/batch/是否退
    checkpoint,底下每個被放大的旋鈕各自縮排一行(見
    `train_conv_compressed._describe_growth`)。"""
    out = []
    for m in _OVERFLOW_BLOCK_RE.finditer(stdout):
        epoch, batch, where, knob_block = int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)
        knobs = [{"layer": k.group(1), "knob": k.group(2), "old": int(k.group(3)),
                  "new": int(k.group(4)), "observed": int(k.group(5))}
                 for k in _KNOB_RE.finditer(knob_block)]
        out.append({"epoch": epoch, "batch": batch, "knobs": knobs,
                    "had_checkpoint": "退回 checkpoint" in where})
    return out


def _assert_grow_formula(knob: dict, grow: float) -> None:
    assert knob["new"] == int(math.ceil(max(knob["observed"], knob["old"]) * grow)), \
        f"放大公式不符:{knob}(grow={grow})"


# ============================================================================
# A. 純函式層級
# ============================================================================

def test_checkpoint_roundtrip_preserves_all_fields():
    """合成小尺寸 params(對齊 layer list 的 weight tuple)+opt_state,存了再讀
    回來,逐位元比對每層權重、Adam opt_state(含 count/mu/nu)、shuffle_key、
    epoch。"""
    cfg = _base_cfg("roundtrip", seed=0, conv2_L_init=32, grow=1.5, epochs=1)
    net = ConvNetCompressed(build_network(cfg["model"]))
    params = net.init(jax.random.PRNGKey(123))
    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(params)

    fake_grad = jax.tree_util.tree_map(lambda w: jnp.ones_like(w) * 0.01, params)
    for _ in range(2):
        updates, opt_state = optimizer.update(fake_grad, opt_state, params)
        params = optax.apply_updates(params, updates)

    shuffle_key = jax.random.PRNGKey(999)
    epoch = 7
    ckpt = Checkpointer(os.path.join(_TMP_DIR, "roundtrip_test.npz"))
    ckpt.save(params=params, opt_state=opt_state, shuffle_key=shuffle_key, epoch=epoch)
    assert ckpt.exists() and ckpt.last_epoch == epoch

    template_params = net.init(jax.random.PRNGKey(0))
    template_opt_state = optimizer.init(template_params)
    loaded_params, loaded_opt_state, loaded_shuffle_key, loaded_epoch = ckpt.load(
        params_template=template_params, opt_state_template=template_opt_state)

    assert len(params) == len(loaded_params)
    for i, (orig, loaded) in enumerate(zip(params, loaded_params)):
        assert jnp.array_equal(orig, loaded), f"params[{i}] 讀回後不一致"

    orig_leaves, _ = jax.tree_util.tree_flatten(opt_state)
    loaded_leaves, _ = jax.tree_util.tree_flatten(loaded_opt_state)
    assert len(orig_leaves) == len(loaded_leaves)
    for i, (o, l) in enumerate(zip(orig_leaves, loaded_leaves)):
        assert jnp.array_equal(jnp.asarray(o), jnp.asarray(l)), f"opt_state leaf[{i}] 不一致"

    assert jnp.array_equal(shuffle_key, loaded_shuffle_key)
    assert loaded_epoch == epoch


def test_grown_to_fit_bumps_only_the_overflowing_knob():
    """`ConvLayer.grown_to_fit`:出界的旋鈕按公式放大,沒出界的旋鈕跟其他欄位
    原封不動;完全不出界時回傳自己(同一個物件)。這是動態放大機制的地基——
    如果 grown_to_fit 不小心動到別的欄位,重編譯後的模型就不再等價於「一開始
    就用大容量」。"""
    cfg = _base_cfg("grow_unit", seed=0, conv2_L_init=100, grow=1.5, epochs=1,
                     conv2_max_out_init=2000)
    _, conv2, _ = build_network(cfg["model"])

    same = conv2.grown_to_fit(LayerDiag(spike_count=jnp.array(0), firing_rate=jnp.array(0.0),
                                         max_real_queue=jnp.array(50), n_out_spikes=jnp.array(10),
                                         min_steps_needed=jnp.array(conv2.max_steps)))
    assert same is conv2, "沒出界應回傳自己"

    # 只有 L 出界:max_steps 沒有獨立超標,但 L 長大之後,這批用「舊、不夠大」
    # 的佇列算出的 min_steps_needed 已經不可信,安全網要求 max_steps 直接
    # 補到新 L(不是保留舊值,也不是信這批的 min_steps_needed)。
    grown = conv2.grown_to_fit(LayerDiag(spike_count=jnp.array(0), firing_rate=jnp.array(0.0),
                                          max_real_queue=jnp.array(777), n_out_spikes=jnp.array(10),
                                          min_steps_needed=jnp.array(0)))
    assert grown is not conv2
    assert grown.L == int(math.ceil(max(777, 100) * 1.5)) == 1166
    assert grown.max_out_spikes == conv2.max_out_spikes, "max_out 沒出界不該動"
    assert grown.max_steps == grown.L, "L 出界長大時,max_steps 安全網要補到新 L"

    # 只有 max_steps 自己的診斷出界(L / max_out 都沒事):max_steps 補到
    # ceil(min_steps_needed * max_steps_grow_factor)(跟 L/max_out 同一種留
    # 餘裕公式,不是精確值——見 shrink_max_steps 的防震盪設計),L / max_out
    # 原封不動。
    min_steps_needed = conv2.max_steps + 7
    grown_steps = conv2.grown_to_fit(LayerDiag(
        spike_count=jnp.array(0), firing_rate=jnp.array(0.0),
        max_real_queue=jnp.array(50), n_out_spikes=jnp.array(10),
        min_steps_needed=jnp.array(min_steps_needed)))
    assert grown_steps is not conv2
    assert grown_steps.L == conv2.L, "L 沒出界不該動"
    assert grown_steps.max_out_spikes == conv2.max_out_spikes, "max_out 沒出界不該動"
    assert grown_steps.max_steps == int(
        math.ceil(min_steps_needed * conv2.max_steps_grow_factor))

    for f in ("name", "ic", "oc", "h_out", "w_out", "k", "s", "p", "tau", "v_th",
              "alpha", "chunk_size", "init_k", "L_grow_factor", "out_grow_factor",
              "out_shrink_threshold", "max_steps_grow_factor", "max_steps_shrink_threshold"):
        assert getattr(grown, f) == getattr(conv2, f), f"{f} 不該被 grown_to_fit 改動"


# ============================================================================
# B. 狀態機邊界情況
# ============================================================================

def test_conv1_L_overflow_grows_not_raises():
    """故意設過小的 conv1_L_init,確認 conv1 的 L 跟其他旋鈕一樣被動態放大、
    訓練正常跑完(不再像舊版那樣直接 raise)。"""
    cfg = _base_cfg("b_conv1_L_grow", seed=42, conv2_L_init=5000, grow=1.5, epochs=2,
                     conv1_L_init=5)  # 真實 max_real_queue 落在 ~100+,5 保證第一個 batch 就出界
    (exp_dir, _, _, _, _, final_cfg), stdout = _run_capture(_write_yaml(cfg))

    ovs = _parse_overflows(stdout)
    assert ovs, "conv1_L_init=5 應該要觸發出界"
    conv1_L_knobs = [k for ov in ovs for k in ov["knobs"] if k["layer"] == "conv1" and k["knob"] == "L"]
    assert conv1_L_knobs, f"應看到 conv1 L 被放大,實際:{ovs}"
    for k in conv1_L_knobs:
        _assert_grow_formula(k, 1.5)
    assert final_cfg["final_capacity"]["conv1"]["L"] > 5
    rows = _read_metrics_csv(exp_dir)
    assert [int(r["epoch"]) for r in rows] == [0, 1], "最終應正常跑完 2 個 epoch"


def test_conv2_L_overflow_before_first_checkpoint_reinits_with_same_seed():
    """conv2_L_init 設到必定在 epoch0 batch0 就出界(校準:seed=42 epoch0 batch0
    的 max_real_queue≈874)。出界發生在還沒套用任何梯度更新之前,退回「訓練
    最初始狀態」應該跟「一開始就用夠大的 L 直接訓練」等價(容差比對,見檔案
    開頭)。"""
    l_init = 32
    overflow_cfg = _base_cfg("b_reinit_attempt", seed=42, conv2_L_init=l_init, grow=2.0, epochs=2)
    (_, _, params_a, _, _, cfg_a), stdout_a = _run_capture(_write_yaml(overflow_cfg))

    ovs = _parse_overflows(stdout_a)
    assert ovs, "conv2_L_init=32 應觸發出界"
    first = ovs[0]
    assert (first["epoch"], first["batch"]) == (0, 0)
    assert not first["had_checkpoint"], "這次出界落在還沒存過 checkpoint 的邊界"
    k = next(k for k in first["knobs"] if k["layer"] == "conv2" and k["knob"] == "L")
    _assert_grow_formula(k, 2.0)
    last_conv2_L = [kk for ov in ovs for kk in ov["knobs"]
                    if kk["layer"] == "conv2" and kk["knob"] == "L"][-1]
    assert cfg_a["final_capacity"]["conv2"]["L"] == last_conv2_L["new"]

    reference_cfg = _base_cfg("b_reinit_reference", seed=42, conv2_L_init=5000, grow=2.0, epochs=2)
    (_, _, params_b, _, _, _), stdout_b = _run_capture(_write_yaml(reference_cfg))
    assert not _parse_overflows(stdout_b), "reference 用 conv2_L=5000 應全程夠用"

    _assert_params_close(params_a, params_b, "出界重來 vs 直接用足夠大 L 訓練")


def test_conv2_L_overflow_mid_epoch_discards_partial_epoch_updates():
    """挑一個讓「第 2 個 batch」才出界的 conv2_L_init(校準:seed=42 epoch0
    四個 batch 約 874/1050/1004/847,L=900 讓 batch0 先正常更新一次,batch1
    才出界)。驗證重來後結果仍跟「直接用夠大 L」在容差內一致——batch0 那次
    更新若沒被正確丟棄,結果會差到遠超 float32 雜訊。"""
    l_init = 900
    overflow_cfg = _base_cfg("b_midepoch_attempt", seed=42, conv2_L_init=l_init, grow=2.0, epochs=2)
    (_, _, params_a, _, _, cfg_a), stdout_a = _run_capture(_write_yaml(overflow_cfg))

    ovs = _parse_overflows(stdout_a)
    assert len(ovs) == 1, f"預期剛好一次出界,實際 {len(ovs)}:{ovs}"
    ov = ovs[0]
    assert (ov["epoch"], ov["batch"]) == (0, 1)
    assert not ov["had_checkpoint"]
    k = next(k for k in ov["knobs"] if k["layer"] == "conv2" and k["knob"] == "L")
    _assert_grow_formula(k, 2.0)

    reference_cfg = _base_cfg("b_midepoch_reference", seed=42, conv2_L_init=5000, grow=2.0, epochs=2)
    (_, _, params_b, _, _, _), stdout_b = _run_capture(_write_yaml(reference_cfg))
    assert not _parse_overflows(stdout_b)

    _assert_params_close(params_a, params_b, "mid-epoch 出界重來 vs 直接用足夠大 L")


def test_conv2_L_overflow_after_checkpoint_resumes_from_disk_not_reinit():
    """先讓訓練正常跑完至少 1 個 epoch(存過 checkpoint),再讓後續某 batch
    出界。校準:seed=1 約 epoch0=[933,943,1011,780]、epoch1=[921,1016,952,836]
    ——conv2_L_init=1011 讓 epoch0 剛好完整跑完,epoch1 第 2 個 batch 才出界。

    驗證:如果「讀 checkpoint 續練」被誤植成「整個重新 init」,start_epoch 會
    錯誤變回 0,epoch0 被重複執行——metrics.csv 就會出現重複 epoch 或超行。"""
    epochs = 3
    cfg = _base_cfg("b_resume_from_checkpoint", seed=1, conv2_L_init=1011, grow=2.0, epochs=epochs)
    (exp_dir, _, _, _, _, final_cfg), stdout = _run_capture(_write_yaml(cfg))

    ovs = _parse_overflows(stdout)
    assert ovs, "應該至少出界一次"
    ov = ovs[0]
    assert ov["epoch"] >= 1, f"測的是「存過 checkpoint 之後才出界」,實際 epoch{ov['epoch']}"
    assert ov["had_checkpoint"]
    k = next(kk for kk in ov["knobs"] if kk["layer"] == "conv2" and kk["knob"] == "L")

    rows = _read_metrics_csv(exp_dir)
    epochs_seen = [int(r["epoch"]) for r in rows]
    assert epochs_seen == list(range(epochs)), (
        f"epoch 編號應該 0..{epochs - 1} 各一次,實際 {epochs_seen}——重複代表 resume 錯成重新 init")

    assert int(rows[0]["conv2_L"]) == 1011, "出界前(epoch0)的 row 記錄舊值"
    assert int(rows[ov["epoch"]]["conv2_L"]) == k["new"], "出界那個 epoch 續練完記錄新值"
    assert int(rows[-1]["conv2_L"]) == k["new"]
    assert final_cfg["final_capacity"]["conv2"]["L"] == k["new"]

    final_ckpt = np.load(os.path.join(exp_dir, TRAIN_DIRNAME, "checkpoint.npz"))
    assert int(final_ckpt["epoch"]) == epochs - 1


def test_conv2_L_overflow_multiple_times_eventually_converges():
    """conv2_L_init 設極小(1),搭配保守放大倍率(1.01——只這個測試用)。校準:
    seed=42 epoch0 約 874/1050/1004/847——L=1 出界放大到 ceil(874*1.01)≈883
    (仍小於後面的 batch),要再度出界放大到 ceil(~1050*1.01)≈1061 才夠。
    驗證最終能正常跑完、不再出界,且過程真的出現至少兩次出界。"""
    cfg = _base_cfg("b_multi_overflow", seed=42, conv2_L_init=1, grow=1.01, epochs=2)
    (exp_dir, _, _, _, _, _), stdout = _run_capture(_write_yaml(cfg))

    n_overflows = stdout.count("[出界]")
    assert n_overflows >= 2, f"L=1 配保守倍率應逼出至少兩次連續出界,實際 {n_overflows} 次"

    rows = _read_metrics_csv(exp_dir)
    assert [int(r["epoch"]) for r in rows] == [0, 1], "最終應正常跑完 2 個 epoch"
    assert all(not math.isnan(float(r["train_loss"])) for r in rows), "loss 不該 NaN"
    assert int(rows[0]["conv2_L"]) == int(rows[1]["conv2_L"]), "收斂後 conv2_L 不該再變"


def test_conv2_L_overflow_on_final_epoch_still_detected():
    """出界恰好發生在最後一個 epoch(seed=1,conv2_L_init=1011,epochs=2)。驗證
    不會被誤判成訓練正常結束——`overflowed` 旗標要跟 `epoch == epochs-1` 邊界
    正確交叉確認。"""
    epochs = 2
    cfg = _base_cfg("b_final_epoch_overflow", seed=1, conv2_L_init=1011, grow=2.0, epochs=epochs)
    (exp_dir, _, _, _, _, final_cfg), stdout = _run_capture(_write_yaml(cfg))

    ovs = _parse_overflows(stdout)
    assert ovs, "最後一個 epoch 出界的訊息不該被吃掉"
    assert ovs[0]["epoch"] == epochs - 1

    rows = _read_metrics_csv(exp_dir)
    assert [int(r["epoch"]) for r in rows] == list(range(epochs))
    assert final_cfg["best"]["epoch"] in range(epochs)


# ============================================================================
# C. 不變量測試
# ============================================================================

def test_sufficient_L_headroom_does_not_change_result():
    """兩個都足夠大、確定不出界的 conv2_L(1100 剛好夠、2200 是兩倍),各自
    完整跑同樣 epoch 數,比對最終 params 在容差內一致——「放大重來語義上等價
    於一開始就用夠大的值」的核心證據,也是 build_conv_queue_compressed「L 留寬
    不影響數值、只影響能不能塞下」在完整訓練迴圈下的再次確認。"""
    epochs = 3
    cfg_a = _base_cfg("c_headroom_just_enough", seed=42, conv2_L_init=1100, grow=2.0, epochs=epochs)
    cfg_b = _base_cfg("c_headroom_double", seed=42, conv2_L_init=2200, grow=2.0, epochs=epochs)

    (_, _, params_a, _, _, _), stdout_a = _run_capture(_write_yaml(cfg_a))
    (_, _, params_b, _, _, _), stdout_b = _run_capture(_write_yaml(cfg_b))

    assert not _parse_overflows(stdout_a), "conv2_L=1100 對 seed=42 小規模應全程夠用"
    assert not _parse_overflows(stdout_b)

    _assert_params_close(params_a, params_b, "conv2_L=1100 vs 2200(L 留多寬不該影響結果)")


def test_end_to_end_smoke_produces_expected_artifacts():
    """把手動驗證過的 compressed_smoke.yaml 流程收成自動化測試:正常結束、
    conv2_L 跟兩個 max_out 都確實被動態放大過至少一次、artifacts 都存在、
    數值沒有 NaN。"""
    smoke_path = os.path.join(CONFIGS_DIR, "conv", "compressed_smoke.yaml")
    (exp_dir, _, _, _, _, final_cfg), stdout = _run_capture(smoke_path)

    ovs = _parse_overflows(stdout)
    assert ovs, "compressed_smoke.yaml 的小起始值應該要觸發出界"
    grown_knobs = {(k["layer"], k["knob"]) for ov in ovs for k in ov["knobs"]}
    assert ("conv2", "L") in grown_knobs, "conv2_L_init=32 應觸發 L 出界"
    assert ("conv1", "max_out") in grown_knobs or ("conv2", "max_out") in grown_knobs, \
        "max_out 起始值刻意設小,應觸發輸出上界出界"

    for fname in ("checkpoint.npz", "run.yaml", "metrics.csv", "best_params.npz",
                  "params.npz"):
        assert os.path.isfile(os.path.join(exp_dir, TRAIN_DIRNAME, fname)), f"缺少 train/{fname}"

    # compressed_smoke.yaml 有開 probe_every -> traces/ 該有東西
    assert os.path.isfile(os.path.join(exp_dir, TRACES_DIRNAME, "summary.npz")), "缺少 traces/summary.npz"
    assert os.path.isfile(os.path.join(exp_dir, TRACES_DIRNAME, "full_epoch_000.npz")), "缺少 traces/full_epoch_000.npz"

    rows = _read_metrics_csv(exp_dir)
    assert len(rows) > 0
    for r in rows:
        loss = float(r["train_loss"])
        assert not math.isnan(loss) and not math.isinf(loss), f"epoch {r['epoch']} loss 異常:{loss}"
    assert final_cfg["final_capacity"]["conv2"]["L"] > 32
    assert final_cfg["final_capacity"]["conv2"]["max_out_spikes"] > 600
    assert final_cfg["final_capacity"]["conv1"]["max_out_spikes"] > 800
    assert int(rows[-1]["conv2_max_out"]) == final_cfg["final_capacity"]["conv2"]["max_out_spikes"]
    assert int(rows[-1]["conv1_max_out"]) == final_cfg["final_capacity"]["conv1"]["max_out_spikes"]


# ============================================================================
# D. 輸出正確性
# ============================================================================

def test_metrics_csv_capacity_columns_reflect_growth_after_overflow():
    """出界之後所有 row 的容量欄位要反映新值,不能停在舊值——沿用「先完整跑完
    1 個 epoch、之後才出界」的校準(seed=1,conv2_L_init=1011)。"""
    epochs = 3
    cfg = _base_cfg("d_capacity_column_growth", seed=1, conv2_L_init=1011, grow=2.0, epochs=epochs)
    (exp_dir, _, _, _, _, _), stdout = _run_capture(_write_yaml(cfg))

    rows = _read_metrics_csv(exp_dir)
    ovs = _parse_overflows(stdout)
    assert ovs, "應該出界一次"
    ov = ovs[0]
    assert ov["epoch"] >= 1
    k = next(kk for kk in ov["knobs"] if kk["layer"] == "conv2" and kk["knob"] == "L")
    assert int(rows[0]["conv2_L"]) == 1011, "出界前(epoch0)記錄舊值"
    assert int(rows[ov["epoch"]]["conv2_L"]) == k["new"], "出界那個 epoch 記錄新值"
    assert int(rows[-1]["conv2_L"]) == k["new"], "之後也是新值,不會又變回舊值"


def test_final_capacity_written_to_config_matches_last_used_value():
    """run 紀錄的 final_capacity[conv2] 應等於訓練結束當下實際用的值——用
    metrics.csv 最後一 row 當獨立比對基準。"""
    smoke_path = os.path.join(CONFIGS_DIR, "conv", "compressed_smoke.yaml")
    (exp_dir, _, _, _, _, final_cfg), _ = _run_capture(smoke_path)

    rows = _read_metrics_csv(exp_dir)
    assert final_cfg["final_capacity"]["conv2"]["L"] == int(rows[-1]["conv2_L"])
    assert final_cfg["final_capacity"]["conv2"]["max_out_spikes"] == int(rows[-1]["conv2_max_out"])


# ============================================================================
# E. 輸出 spike 上界(max_out_spikes)的動態偵測
# ============================================================================

def _buf_cfg(run_name: str, **buf_overrides) -> dict:
    """_base_cfg 的 E 類變體:conv2_L 給足(5000,隔離掉 L 出界),再把某個
    conv 層的 max_out_spikes 覆寫成會出界的小值。buf_overrides 用
    conv1_max_out_init / conv2_max_out_init 指定。"""
    cfg = _base_cfg(run_name, seed=42, conv2_L_init=5000, grow=1.5, epochs=2)
    idx = {"conv1_max_out_init": 0, "conv2_max_out_init": 1}
    for key, val in buf_overrides.items():
        cfg["model"]["layers"][idx[key]]["max_out_spikes"] = val
    return cfg


def test_conv_output_buffer_overflow_detected_and_grown():
    """conv2_max_out 設到必定在第一個 batch 就出界。驗證:偵測觸發、印出
    `[出界]`(帶 max_out 旋鈕)、放大後正常跑完 2 個 epoch、不再出界、
    metrics/final_cfg 反映新值、沒有 NaN。"""
    cfg = _buf_cfg("e_buf_overflow", conv2_max_out_init=500)
    (exp_dir, _, _, _, _, final_cfg), stdout = _run_capture(_write_yaml(cfg))

    ovs = _parse_overflows(stdout)
    assert ovs, "conv2_max_out=500 應觸發出界"
    max_out_knobs = [k for ov in ovs for k in ov["knobs"]
                     if k["layer"] == "conv2" and k["knob"] == "max_out"]
    assert max_out_knobs, f"應看到 conv2 max_out 被放大:{ovs}"
    for k in max_out_knobs:
        _assert_grow_formula(k, 1.5)
    assert not [k for ov in ovs for k in ov["knobs"] if k["knob"] == "L"], \
        "conv2_L=5000 應全程夠用,這裡只測 max_out"

    assert final_cfg["final_capacity"]["conv2"]["max_out_spikes"] > 500
    rows = _read_metrics_csv(exp_dir)
    assert [int(r["epoch"]) for r in rows] == [0, 1]
    for r in rows:
        assert not math.isnan(float(r["train_loss"]))
    assert int(rows[-1]["conv2_max_out"]) == final_cfg["final_capacity"]["conv2"]["max_out_spikes"]


def test_conv_output_buffer_grow_result_matches_generous_start():
    """max_out「小起始 -> 出界放大」跟「一開始就給夠」訓練結果在容差內一致
    ——跟 L headroom 不變量同一個道理。出界發生在還沒存 checkpoint 前,是
    乾淨的整個重來。"""
    small = _buf_cfg("e_buf_small", conv2_max_out_init=500)
    big = _buf_cfg("e_buf_big")  # max_out 沿用 _base_cfg 已足夠大的預設

    (_, _, params_small, _, _, _), stdout_small = _run_capture(_write_yaml(small))
    (_, _, params_big, _, _, _), stdout_big = _run_capture(_write_yaml(big))

    assert _parse_overflows(stdout_small), "小起始值應出界"
    assert not _parse_overflows(stdout_big), "_base_cfg 預設 max_out 對 seed=42 小規模應夠用"
    _assert_params_close(params_small, params_big, "max_out 出界放大 vs 一開始就給夠")


TESTS = [
    test_checkpoint_roundtrip_preserves_all_fields,
    test_grown_to_fit_bumps_only_the_overflowing_knob,
    test_conv1_L_overflow_grows_not_raises,
    test_conv2_L_overflow_before_first_checkpoint_reinits_with_same_seed,
    test_conv2_L_overflow_mid_epoch_discards_partial_epoch_updates,
    test_conv2_L_overflow_after_checkpoint_resumes_from_disk_not_reinit,
    test_conv2_L_overflow_multiple_times_eventually_converges,
    test_conv2_L_overflow_on_final_epoch_still_detected,
    test_sufficient_L_headroom_does_not_change_result,
    test_end_to_end_smoke_produces_expected_artifacts,
    test_metrics_csv_capacity_columns_reflect_growth_after_overflow,
    test_final_capacity_written_to_config_matches_last_used_value,
    test_conv_output_buffer_overflow_detected_and_grown,
    test_conv_output_buffer_grow_result_matches_generous_start,
]


if __name__ == "__main__":
    for test in TESTS:
        test()
        print(f"PASS: {test.__name__}")
    print(f"\n全部 {len(TESTS)} 項測試通過")
