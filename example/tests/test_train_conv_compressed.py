"""壓縮容量旋鈕動態放大機制(`example/train_conv_compressed.py`)的測試。沿用真實
N-MNIST 小規模資料(`_train_runs.base_cfg`:max_events=2000、train_size=16、
val_size=8),不用合成資料;不碰 train_conv_compressed.py 本身,只呼叫它公開
的函式/`train()` entrypoint。訓練寫在暫存目錄;「一開始就給夠容量」的對照組
共用 conftest.py 的 reference_run,整套測試只訓練一次。

step 4b 起,每個 conv 層有兩個會出界的容量:壓縮佇列長度 `L`、輸出 spike
上界 `max_out_spikes`。兩者同一套機制:偵測 -> 該層照 `GrowthPolicy` 放大 ->
退 checkpoint -> 重編譯續練。conv1 的 `L` 也走這套(不再像舊版那樣「conv1
出界直接 raise」)。放大公式:`new = ceil(max(observed, old) * grow_factor)`,
每個旋鈕各自一個 grow_factor。

**校準說明**:B 類測試要精準命中「第一個 epoch 就出界」「存過 checkpoint
之後才出界」「連續出界兩次」「最後一個 epoch 才出界」這些邊界,用到的
conv2_L_init/seed 是實際跑校準量出來的(固定 seed_train=0/train_size=16/
batch_size=4,用夠大的容量不截斷任何東西,記錄每個 (epoch,batch) 真正的
佇列需求 needed["L"]):

- seed=42:epoch0 四個 batch 的 conv2 佇列需求約 874/1050/1004/847,
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
整數計數(驅動出界判斷的佇列需求)、epoch 編號、容量欄位這些不牽涉
浮點規約重算的,一律維持逐位元/逐值精確比對。
"""
import csv
import math
import os
import re

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from example.tests._train_runs import base_cfg, run_capture
from example.train_conv_compressed import (_build_learning_rate, _build_optimizer,
                                            _cross_entropy_loss)
from example.utils import TRAIN_DIRNAME

_TRAIN_RESULT_TOL = 1e-4


@pytest.fixture(scope="module")
def run_root(tmp_path_factory):
    """這個檔案的訓練都寫在同一個暫存目錄。"""
    return tmp_path_factory.mktemp("train_runs")


def _assert_params_close(params_a, params_b, msg_prefix: str) -> None:
    # params 是對齊 layer list 的位置 tuple(一層一份權重陣列)。
    assert len(params_a) == len(params_b)
    for i, (a, b) in enumerate(zip(params_a, params_b)):
        max_diff = float(jnp.max(jnp.abs(a - b)))
        assert max_diff <= _TRAIN_RESULT_TOL, (
            f"{msg_prefix}:層[{i}] max|Δ|={max_diff:.3e} 超過容差 {_TRAIN_RESULT_TOL:.0e}")


def _read_metrics_csv(exp_dir: str) -> list:
    with open(os.path.join(exp_dir, TRAIN_DIRNAME, "metrics.csv"), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


_OVERFLOW_BLOCK_RE = re.compile(r"\[出界\] epoch=(\d+) batch=(\d+): (.+)\n((?:  .+\n?)*)")
_KNOB_RE = re.compile(r"  (conv\d) (L|max_out_spikes|max_steps) (\d+)->(\d+)\(觀察 (\d+)\)")


def _parse_overflows(stdout: str) -> list[dict]:
    """從 stdout 抓出每一次 `[出界]` 事件,解析成結構化紀錄。每筆帶
    `epoch`/`batch`/`had_checkpoint` 跟一個 `knobs` list:每個被放大的旋鈕的
    (layer, knob, old, new, observed)。`[出界]` 那行只有 epoch/batch/是否退
    checkpoint,底下每個被放大的旋鈕各自縮排一行(見
    `example.utils.describe_growth`)。"""
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


def test_optimizer_knobs_run_through_training(run_root):
    """weight_decay、grad_clip_norm、score_cap、lr_cosine_decay 全開,從 config 接進
    train(),跑完 2 個 epoch。opt_state 的存讀另外在 test_checkpoint.py 測。"""
    cfg = base_cfg("optimizer_knobs_smoke", seed=0, conv2_L_init=2000, grow=1.5, epochs=2)
    cfg["train"].update(weight_decay=1e-4, grad_clip_norm=10.0, score_cap=6.0,
                        lr_cosine_decay=True)
    (exp_dir, *_), _stdout = run_capture(cfg, run_root)
    rows = _read_metrics_csv(exp_dir)
    assert [int(r["epoch"]) for r in rows] == [0, 1]


def test_grad_clip_norm_none_matches_plain_adamw():
    """`train.grad_clip_norm` 不填(預設)時,`_build_optimizer` 要跟原本的
    `optax.adamw` 逐位元一致——「不設就是原樣通過」的慣例,跟
    `score_cap=None`/`weight_decay=0` 同一套。"""
    train_cfg = {"lr": 1e-2, "weight_decay": 0.0}
    data_cfg = {"train_size": 40}
    built = _build_optimizer(train_cfg, data_cfg, batch_size=4)
    plain = optax.adamw(1e-2, weight_decay=0.0)

    params = {"w": jnp.array([1.0, 2.0, 3.0])}
    grad = {"w": jnp.array([5.0, -3.0, 1.0])}
    upd_a, _ = built.update(grad, built.init(params), params)
    upd_b, _ = plain.update(grad, plain.init(params), params)
    assert jnp.array_equal(upd_a["w"], upd_b["w"])


def test_grad_clip_norm_changes_update_relative_to_unclipped():
    """驗證 `grad_clip_norm` 真的接進 optimizer chain、且順序在 `adamw` 前面
    (docs/math/梯度下降曲率穩定性推導.md §8.1)。Adam 對非零梯度的 update
    大小本身是比例縮放不變的(踩過的坑:單純把梯度縮小一點,Adam 會自己把
    update 正規化回同一個量級,測不出差異)——要把 `grad_clip_norm` 設到讓
    夾完的梯度掉到 Adam 內部 `eps`(1e-8)量級以下,`m/sqrt(v+eps)` 的行為
    才會明顯偏離未夾版本,才算真的驗證到 clip 有接進管線、不是被 Adam
    悄悄吃掉。"""
    params = {"w": jnp.array([1.0, 1.0, 1.0])}
    huge_grad = {"w": jnp.array([1e6, 1e6, 1e6])}

    unclipped = _build_optimizer({"lr": 1e-2}, {"train_size": 40}, batch_size=4)
    clipped = _build_optimizer({"lr": 1e-2, "grad_clip_norm": 1e-10},
                               {"train_size": 40}, batch_size=4)

    upd_u, _ = unclipped.update(huge_grad, unclipped.init(params), params)
    upd_c, _ = clipped.update(huge_grad, clipped.init(params), params)

    assert not jnp.allclose(upd_u["w"], upd_c["w"])


def test_score_cap_none_matches_plain_cross_entropy():
    """`score_cap=None`(預設)時,`_cross_entropy_loss` 要跟原始
    `optax.softmax_cross_entropy` 逐位元一致——這是「不設就是原樣通過」的
    基本保證。"""
    scores = jnp.array([[0.0, 3.0, -1.0]])
    labels_onehot = jax.nn.one_hot(jnp.array([1]), 3)

    capped = _cross_entropy_loss(scores, labels_onehot, None)
    plain = optax.softmax_cross_entropy(scores, labels_onehot)

    assert jnp.array_equal(capped, plain)


def test_score_cap_does_not_distort_scores_far_below_the_cap():
    """`score_cap` 只在分數逼近上限時才該生效(docs/math/梯度下降曲率穩定性
    推導.md §8.4)——分數遠小於 cap 時,`tanh(x/C) ≈ x/C`,loss 應該幾乎跟
    不設 cap 時一樣,不能連正常訓練階段的推力都跟著打折。"""
    scores = jnp.array([[0.0, 1.0, 0.0]])  # logit 差距只有 1,遠低於 cap
    labels_onehot = jax.nn.one_hot(jnp.array([1]), 3)

    uncapped = _cross_entropy_loss(scores, labels_onehot, None)
    capped = _cross_entropy_loss(scores, labels_onehot, 6.0)

    assert jnp.allclose(uncapped, capped, atol=1e-2)


def test_score_cap_saturates_extreme_scores_to_a_fixed_loss():
    """分數遠超過 cap 時,`tanh` 飽和到 ±1,不管原始分數多誇張,轉換後的分數
    都趨近同一個值——loss 因此不再隨原始分數繼續下降,不像沒有 cap 時
    logit 差距可以無界增長、loss 可以無界壓向 0。"""
    labels_onehot = jax.nn.one_hot(jnp.array([1]), 3)
    loss_1e6 = _cross_entropy_loss(jnp.array([[0.0, 1e6, 0.0]]), labels_onehot, 6.0)
    loss_1e9 = _cross_entropy_loss(jnp.array([[0.0, 1e9, 0.0]]), labels_onehot, 6.0)

    assert jnp.allclose(loss_1e6, loss_1e9, atol=1e-6)
    assert float(loss_1e6[0]) > 1e-4  # 有實質下限,不會被沖到趨近 0


def test_build_learning_rate_without_cosine_decay_returns_plain_float():
    """`train.lr_cosine_decay` 不填(預設)時,`_build_learning_rate` 原樣
    傳回 `train.lr` 這個純量,不包成 schedule——這是「不設就是原樣通過」的
    基本保證,跟 `score_cap=None`/`weight_decay=0` 同一個慣例。"""
    train_cfg = {"lr": 1e-2, "epochs": 10}
    data_cfg = {"train_size": 40}
    lr = _build_learning_rate(train_cfg, data_cfg, batch_size=4)
    assert lr == 1e-2


def test_build_learning_rate_cosine_decay_starts_high_ends_low():
    """`lr_cosine_decay=True` 時,回傳的是 schedule(呼叫得出值的函式),
    第 0 步等於 `train.lr`,退火到最後一步時降到接近 `alpha` 那個下限比例
    (docs/math/梯度下降曲率穩定性推導.md 第 5 節:讓 $2/\\eta$ 隨訓練進行
    升高)。"""
    train_cfg = {"lr": 1e-2, "epochs": 10, "lr_cosine_decay": True, "lr_cosine_alpha": 0.0}
    data_cfg = {"train_size": 40}
    schedule = _build_learning_rate(train_cfg, data_cfg, batch_size=4)
    total_steps = train_cfg["epochs"] * (data_cfg["train_size"] // 4)  # 10 * 10 = 100

    assert float(schedule(0)) == pytest.approx(1e-2, rel=1e-3)
    assert float(schedule(total_steps - 1)) < 1e-2 * 0.01  # alpha=0,退火到接近 0


# ============================================================================
# B. 狀態機邊界情況
# ============================================================================

def test_conv1_L_overflow_grows_not_raises(run_root):
    """故意設過小的 conv1_L_init,確認 conv1 的 L 跟其他旋鈕一樣被動態放大、
    訓練正常跑完(不再像舊版那樣直接 raise)。"""
    cfg = base_cfg("b_conv1_L_grow", seed=42, conv2_L_init=5000, grow=1.5, epochs=2,
                     conv1_L_init=5)  # 真實佇列需求落在 ~100+,5 保證第一個 batch 就出界
    (exp_dir, _, _, _, _, final_cfg), stdout = run_capture(cfg, run_root)

    ovs = _parse_overflows(stdout)
    assert ovs, "conv1_L_init=5 應該要觸發出界"
    conv1_L_knobs = [k for ov in ovs for k in ov["knobs"] if k["layer"] == "conv1" and k["knob"] == "L"]
    assert conv1_L_knobs, f"應看到 conv1 L 被放大,實際:{ovs}"
    for k in conv1_L_knobs:
        _assert_grow_formula(k, 1.5)
    assert final_cfg["final_capacity"]["conv1"]["L"] > 5
    rows = _read_metrics_csv(exp_dir)
    assert [int(r["epoch"]) for r in rows] == [0, 1], "最終應正常跑完 2 個 epoch"


def test_conv2_L_overflow_before_first_checkpoint_reinits_with_same_seed(run_root, reference_run):
    """conv2_L_init 設到必定在 epoch0 batch0 就出界(校準:seed=42 epoch0 batch0
    的佇列需求≈874)。出界發生在還沒套用任何梯度更新之前,退回「訓練
    最初始狀態」應該跟「一開始就用夠大的 L 直接訓練」等價(容差比對,見檔案
    開頭)。"""
    l_init = 32
    overflow_cfg = base_cfg("b_reinit_attempt", seed=42, conv2_L_init=l_init, grow=2.0, epochs=2)
    (_, _, params_a, _, _, cfg_a), stdout_a = run_capture(overflow_cfg, run_root)

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

    (_, _, params_b, _, _, _), stdout_b = reference_run
    assert not _parse_overflows(stdout_b), "reference 用 conv2_L=5000 應全程夠用"

    _assert_params_close(params_a, params_b, "出界重來 vs 直接用足夠大 L 訓練")


def test_conv2_L_overflow_mid_epoch_discards_partial_epoch_updates(run_root, reference_run):
    """挑一個讓「第 2 個 batch」才出界的 conv2_L_init(校準:seed=42 epoch0
    四個 batch 約 874/1050/1004/847,L=900 讓 batch0 先正常更新一次,batch1
    才出界)。驗證重來後結果仍跟「直接用夠大 L」在容差內一致——batch0 那次
    更新若沒被正確丟棄,結果會差到遠超 float32 雜訊。"""
    l_init = 900
    overflow_cfg = base_cfg("b_midepoch_attempt", seed=42, conv2_L_init=l_init, grow=2.0, epochs=2)
    (_, _, params_a, _, _, cfg_a), stdout_a = run_capture(overflow_cfg, run_root)

    ovs = _parse_overflows(stdout_a)
    assert len(ovs) == 1, f"預期剛好一次出界,實際 {len(ovs)}:{ovs}"
    ov = ovs[0]
    assert (ov["epoch"], ov["batch"]) == (0, 1)
    assert not ov["had_checkpoint"]
    k = next(k for k in ov["knobs"] if k["layer"] == "conv2" and k["knob"] == "L")
    _assert_grow_formula(k, 2.0)

    (_, _, params_b, _, _, _), stdout_b = reference_run
    assert not _parse_overflows(stdout_b)

    _assert_params_close(params_a, params_b, "mid-epoch 出界重來 vs 直接用足夠大 L")


def test_conv2_L_overflow_after_checkpoint_resumes_from_disk_not_reinit(run_root):
    """先讓訓練正常跑完至少 1 個 epoch(存過 checkpoint),再讓後續某 batch
    出界。校準:seed=1 約 epoch0=[933,943,1011,780]、epoch1=[921,1016,952,836]
    ——conv2_L_init=1011 讓 epoch0 剛好完整跑完,epoch1 第 2 個 batch 才出界。

    驗證:如果「讀 checkpoint 續練」被誤植成「整個重新 init」,start_epoch 會
    錯誤變回 0,epoch0 被重複執行——metrics.csv 就會出現重複 epoch 或超行。"""
    epochs = 3
    cfg = base_cfg("b_resume_from_checkpoint", seed=1, conv2_L_init=1011, grow=2.0, epochs=epochs)
    (exp_dir, _, _, _, _, final_cfg), stdout = run_capture(cfg, run_root)

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

    assert int(rows[0]["conv2_max_event_queue"]) == 1011, "出界前(epoch0)的 row 記錄舊值"
    assert int(rows[ov["epoch"]]["conv2_max_event_queue"]) == k["new"], "出界那個 epoch 續練完記錄新值"
    assert int(rows[-1]["conv2_max_event_queue"]) == k["new"]
    assert final_cfg["final_capacity"]["conv2"]["L"] == k["new"]

    final_ckpt = np.load(os.path.join(exp_dir, TRAIN_DIRNAME, "checkpoint.npz"))
    assert int(final_ckpt["epoch"]) == epochs - 1


def test_conv2_L_overflow_multiple_times_eventually_converges(run_root):
    """conv2_L_init 設極小(1),搭配保守放大倍率(1.01——只這個測試用)。校準:
    seed=42 epoch0 約 874/1050/1004/847——L=1 出界放大到 ceil(874*1.01)≈883
    (仍小於後面的 batch),要再度出界放大到 ceil(~1050*1.01)≈1061 才夠。
    驗證最終能正常跑完、不再出界,且過程真的出現至少兩次出界。"""
    cfg = base_cfg("b_multi_overflow", seed=42, conv2_L_init=1, grow=1.01, epochs=2)
    (exp_dir, _, _, _, _, _), stdout = run_capture(cfg, run_root)

    n_overflows = stdout.count("[出界]")
    assert n_overflows >= 2, f"L=1 配保守倍率應逼出至少兩次連續出界,實際 {n_overflows} 次"

    rows = _read_metrics_csv(exp_dir)
    assert [int(r["epoch"]) for r in rows] == [0, 1], "最終應正常跑完 2 個 epoch"
    assert all(not math.isnan(float(r["train_loss"])) for r in rows), "loss 不該 NaN"
    assert int(rows[0]["conv2_max_event_queue"]) == int(rows[1]["conv2_max_event_queue"]), \
        "收斂後 conv2_max_event_queue 不該再變"


def test_conv2_L_overflow_on_final_epoch_still_detected(run_root):
    """出界恰好發生在最後一個 epoch(seed=1,conv2_L_init=1011,epochs=2)。驗證
    不會被誤判成訓練正常結束——`overflowed` 旗標要跟 `epoch == epochs-1` 邊界
    正確交叉確認。"""
    epochs = 2
    cfg = base_cfg("b_final_epoch_overflow", seed=1, conv2_L_init=1011, grow=2.0, epochs=epochs)
    (exp_dir, _, _, _, _, final_cfg), stdout = run_capture(cfg, run_root)

    ovs = _parse_overflows(stdout)
    assert ovs, "最後一個 epoch 出界的訊息不該被吃掉"
    assert ovs[0]["epoch"] == epochs - 1

    rows = _read_metrics_csv(exp_dir)
    assert [int(r["epoch"]) for r in rows] == list(range(epochs))
    assert final_cfg["best"]["epoch"] in range(epochs)


# ============================================================================
# C. 輸出 spike 上界(max_out_spikes)的動態偵測
# ============================================================================

def _buf_cfg(run_name: str, **buf_overrides) -> dict:
    """base_cfg 的 C 類變體:設定跟參考訓練相同(conv2_L=5000 隔離掉 L 出界),
    只把某個 conv 層的 max_out_spikes 覆寫成會出界的小值。buf_overrides 用
    conv1_max_out_init / conv2_max_out_init 指定。"""
    cfg = base_cfg(run_name, seed=42, conv2_L_init=5000, grow=2.0, epochs=2)
    idx = {"conv1_max_out_init": 0, "conv2_max_out_init": 1}
    for key, val in buf_overrides.items():
        cfg["model"]["layers"][idx[key]]["max_out_spikes"] = val
    return cfg


def test_conv_output_buffer_overflow_grows_and_matches_generous_start(run_root, reference_run):
    """conv2_max_out 設到必定在第一個 batch 就出界。驗證:偵測觸發、印出
    `[出界]`(帶 max_out 旋鈕)、放大後正常跑完 2 個 epoch、不再出界、
    metrics/final_cfg 反映新值、沒有 NaN;出界發生在還沒存 checkpoint 前,
    是乾淨的整個重來,結果要跟一開始就給夠(參考訓練)在容差內一致。"""
    cfg = _buf_cfg("c_buf_overflow", conv2_max_out_init=500)
    (exp_dir, _, final_params, _, _, final_cfg), stdout = run_capture(cfg, run_root)

    ovs = _parse_overflows(stdout)
    assert ovs, "conv2_max_out=500 應觸發出界"
    max_out_knobs = [k for ov in ovs for k in ov["knobs"]
                     if k["layer"] == "conv2" and k["knob"] == "max_out_spikes"]
    assert max_out_knobs, f"應看到 conv2 max_out 被放大:{ovs}"
    for k in max_out_knobs:
        _assert_grow_formula(k, 2.0)
    assert not [k for ov in ovs for k in ov["knobs"] if k["knob"] == "L"], \
        "conv2_L=5000 應全程夠用,這裡只測 max_out"

    assert final_cfg["final_capacity"]["conv2"]["max_out_spikes"] > 500
    rows = _read_metrics_csv(exp_dir)
    assert [int(r["epoch"]) for r in rows] == [0, 1]
    for r in rows:
        assert not math.isnan(float(r["train_loss"]))
    assert int(rows[-1]["conv2_max_layer_spikes"]) == final_cfg["final_capacity"]["conv2"]["max_out_spikes"]

    (_, _, params_ref, _, _, _), stdout_ref = reference_run
    assert not _parse_overflows(stdout_ref), "參考訓練的 max_out 預設值對 seed=42 小規模應夠用"
    _assert_params_close(final_params, params_ref, "max_out 出界放大 vs 一開始就給夠")
