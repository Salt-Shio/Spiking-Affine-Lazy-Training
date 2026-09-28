"""訓練(example/training/)的測試:optimizer、loss 的純函式,跟訓練迴圈遇到容量出界時的處理。

出界測試用合成資料(_train_runs.synthetic_setup)。conv1 的佇列需求等於每筆樣本的事件數,
跟權重無關,所以出界落在哪個 epoch、哪個 batch,由事件數跟樣本順序精準決定。conv2 的需求
會隨權重變,只在第一個 batch(權重還是初始值)用 first_batch_needed 先量再設容量。

出界之後,迴圈要丟掉這個 batch 跟這個 epoch 已套用的更新,退回最後跑完的 epoch(沒有就從頭),
放大後重來。驗證方式:最終權重跟一開始就給夠容量的參考訓練一致。兩次獨立 jit 編譯的訓練有
float32 加總順序的雜訊(實測 1e-7 ~ 1e-6),權重用 atol 1e-4 比;epoch、容量、事件逐值比。
"""
import csv
import math
import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from example.checkpoint import Checkpointer
from example.models.conv_net import build_network
from example.tests._train_runs import (SYNTH_GROW, batch_needs, first_batch_needed, run_train,
                                       synthetic_cfg, synthetic_setup)
from example.paths import CONFIGS_DIR
from example.train import check_config_keys, train
from example.training.loss import cross_entropy_loss
from example.training.optim import build_learning_rate, build_optimizer
from example.training.run_dir import make_exp_dir
from example.utils import (TRAIN_DIRNAME, WEIGHTS_DIRNAME, load_config, split_raw_events,
                           weight_snapshot_path)
from salt_core.io import load_weights, network_from_dict

_TRAIN_RESULT_TOL = 1e-4


@pytest.fixture(scope="module")
def run_root(tmp_path_factory):
    """這個檔案的訓練都寫在同一個暫存目錄。"""
    return tmp_path_factory.mktemp("train_runs")


@pytest.fixture(scope="module")
def synth():
    return synthetic_setup()


@pytest.fixture(scope="module")
def synth_reference(synth, run_root):
    """合成資料、容量給足、3 epoch 的參考訓練。"""
    return run_train(synthetic_cfg("reference", synth.seed), synth.data, run_root)


def _assert_params_close(params_a, params_b, msg_prefix: str) -> None:
    assert len(params_a) == len(params_b)
    for i, (a, b) in enumerate(zip(params_a, params_b)):
        max_diff = float(jnp.max(jnp.abs(a - b)))
        assert max_diff <= _TRAIN_RESULT_TOL, (
            f"{msg_prefix}:層[{i}] max|Δ|={max_diff:.3e} 超過容差 {_TRAIN_RESULT_TOL:.0e}")


def _snapshot_params(result, epoch: int) -> tuple:
    _network, params = load_weights(
        weight_snapshot_path(os.path.join(result.exp_dir, WEIGHTS_DIRNAME), epoch))
    return params


def _read_metrics_csv(exp_dir: str) -> list:
    with open(os.path.join(exp_dir, TRAIN_DIRNAME, "metrics.csv"), newline="",
              encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _grow_events(result) -> list:
    return [e for e in result.run_record["capacity_events"] if e["kind"] == "grow"]


def _change(event: dict, layer: str, knob: str) -> dict:
    return next(c for c in event["changes"] if c["layer"] == layer and c["knob"] == knob)


def _final_capacity(result, layer_name: str):
    network = network_from_dict(result.run_record["network"])
    return next(layer.capacity for layer in network.layers if layer.name == layer_name)


def _assert_grown_by_formula(change: dict, grow: float) -> None:
    assert change["new"] == math.ceil(change["observed"] * grow), f"放大公式不符:{change}"


# ============================================================================
# A. optimizer、loss
# ============================================================================

def test_optimizer_knobs_run_through_training(synth, run_root):
    """weight_decay、grad_clip_norm、score_cap、lr_cosine_decay 全開,從 config 接進
    train(),跑完 2 個 epoch。opt_state 的存讀另外在 test_checkpoint.py 測。"""
    cfg = synthetic_cfg("optimizer_knobs", synth.seed, epochs=2)
    cfg["train"].update(weight_decay=1e-4, grad_clip_norm=10.0, score_cap=6.0,
                        lr_cosine_decay=True)
    result = run_train(cfg, synth.data, run_root)
    rows = _read_metrics_csv(result.exp_dir)
    assert [int(r["epoch"]) for r in rows] == [0, 1]


def test_grad_clip_norm_none_matches_plain_adamw():
    """`train.grad_clip_norm` 不填(預設)時,`build_optimizer` 要跟原本的
    `optax.adamw` 逐位元一致——「不設就是原樣通過」的慣例,跟
    `score_cap=None`/`weight_decay=0` 同一套。"""
    train_cfg = {"lr": 1e-2, "weight_decay": 0.0}
    built = build_optimizer(train_cfg, n_train=40, batch_size=4)
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

    unclipped = build_optimizer({"lr": 1e-2}, n_train=40, batch_size=4)
    clipped = build_optimizer({"lr": 1e-2, "grad_clip_norm": 1e-10}, n_train=40, batch_size=4)

    upd_u, _ = unclipped.update(huge_grad, unclipped.init(params), params)
    upd_c, _ = clipped.update(huge_grad, clipped.init(params), params)

    assert not jnp.allclose(upd_u["w"], upd_c["w"])


def test_score_cap_none_matches_plain_cross_entropy():
    """`score_cap=None`(預設)時,`cross_entropy_loss` 要跟原始
    `optax.softmax_cross_entropy` 逐位元一致——這是「不設就是原樣通過」的
    基本保證。"""
    scores = jnp.array([[0.0, 3.0, -1.0]])
    labels_onehot = jax.nn.one_hot(jnp.array([1]), 3)

    capped = cross_entropy_loss(scores, labels_onehot, None)
    plain = optax.softmax_cross_entropy(scores, labels_onehot)

    assert jnp.array_equal(capped, plain)


def test_score_cap_does_not_distort_scores_far_below_the_cap():
    """`score_cap` 只在分數逼近上限時才該生效(docs/math/梯度下降曲率穩定性
    推導.md §8.4)——分數遠小於 cap 時,`tanh(x/C) ≈ x/C`,loss 應該幾乎跟
    不設 cap 時一樣,不能連正常訓練階段的推力都跟著打折。"""
    scores = jnp.array([[0.0, 1.0, 0.0]])  # logit 差距只有 1,遠低於 cap
    labels_onehot = jax.nn.one_hot(jnp.array([1]), 3)

    uncapped = cross_entropy_loss(scores, labels_onehot, None)
    capped = cross_entropy_loss(scores, labels_onehot, 6.0)

    assert jnp.allclose(uncapped, capped, atol=1e-2)


def test_score_cap_saturates_extreme_scores_to_a_fixed_loss():
    """分數遠超過 cap 時,`tanh` 飽和到 ±1,不管原始分數多誇張,轉換後的分數
    都趨近同一個值——loss 因此不再隨原始分數繼續下降,不像沒有 cap 時
    logit 差距可以無界增長、loss 可以無界壓向 0。"""
    labels_onehot = jax.nn.one_hot(jnp.array([1]), 3)
    loss_1e6 = cross_entropy_loss(jnp.array([[0.0, 1e6, 0.0]]), labels_onehot, 6.0)
    loss_1e9 = cross_entropy_loss(jnp.array([[0.0, 1e9, 0.0]]), labels_onehot, 6.0)

    assert jnp.allclose(loss_1e6, loss_1e9, atol=1e-6)
    assert float(loss_1e6[0]) > 1e-4  # 有實質下限,不會被沖到趨近 0


def test_build_learning_rate_without_cosine_decay_returns_plain_float():
    """`train.lr_cosine_decay` 不填(預設)時,`build_learning_rate` 原樣
    傳回 `train.lr` 這個純量,不包成 schedule——這是「不設就是原樣通過」的
    基本保證,跟 `score_cap=None`/`weight_decay=0` 同一個慣例。"""
    train_cfg = {"lr": 1e-2, "epochs": 10}
    lr = build_learning_rate(train_cfg, n_train=40, batch_size=4)
    assert lr == 1e-2


def test_build_learning_rate_cosine_decay_starts_high_ends_low():
    """`lr_cosine_decay=True` 時,回傳的是 schedule(呼叫得出值的函式),
    第 0 步等於 `train.lr`,退火到最後一步時降到接近 `alpha` 那個下限比例
    (docs/math/梯度下降曲率穩定性推導.md 第 5 節:讓 $2/\\eta$ 隨訓練進行
    升高)。"""
    train_cfg = {"lr": 1e-2, "epochs": 10, "lr_cosine_decay": True, "lr_cosine_alpha": 0.0}
    schedule = build_learning_rate(train_cfg, n_train=40, batch_size=4)
    total_steps = train_cfg["epochs"] * (40 // 4)  # 10 * 10 = 100

    assert float(schedule(0)) == pytest.approx(1e-2, rel=1e-3)
    assert float(schedule(total_steps - 1)) < 1e-2 * 0.01  # alpha=0,退火到接近 0



# ============================================================================
# B. 合成資料的構造
# ============================================================================

def test_synthetic_conv1_queue_need_equals_event_count(synth):
    """所有事件落在同一個像素,涵蓋它的 conv1 神經元每顆都收到全部事件。"""
    network = build_network(synthetic_cfg("check", synth.seed)["model"])
    output = jax.jit(network.apply_batched)(network.init(jax.random.PRNGKey(0)),
                                            split_raw_events(synth.data.train))
    np.testing.assert_array_equal(np.asarray(output.diags[0].needed["max_queue_len"]),
                                  synth.needs)


def test_synthetic_epoch_zero_batch_needs_increase_and_largest_sample_waits_for_epoch_one(synth):
    """epoch 0 各 batch 的最大需求是 3, 5, 7, 9;需求 12 的那筆 epoch 0 沒用到,epoch 1 有。"""
    assert batch_needs(synth, 0) == [3, 5, 7, 9]
    assert max(batch_needs(synth, 1)) == 12


def test_synthetic_reference_never_overflows(synth_reference):
    assert synth_reference.run_record["capacity_events"] == []



# ============================================================================
# dormant 統計的層由 train.dormant_layers 決定
# ============================================================================

def test_no_dormant_layers_means_no_dormant_columns(synth_reference):
    rows = _read_metrics_csv(synth_reference.exp_dir)
    assert not [c for c in rows[0] if c.endswith("_dormant_frac")]


def test_unknown_dormant_layer_is_rejected(synth, run_root):
    cfg = synthetic_cfg("bad_dormant", synth.seed, epochs=1)
    cfg["train"]["dormant_layers"] = ["conv1", "nope"]
    with pytest.raises(ValueError, match="nope"):
        run_train(cfg, synth.data, run_root)


# ============================================================================
# config 的 key 檢查
# ============================================================================

def test_baseline_config_passes_key_check():
    check_config_keys(load_config(str(CONFIGS_DIR / "conv" / "baseline.yaml")))


@pytest.mark.parametrize("section", ["train", "data", "model"])
def test_unknown_config_key_is_rejected(section):
    cfg = load_config(str(CONFIGS_DIR / "conv" / "baseline.yaml"))
    cfg[section]["weight_decy"] = 1.0e-2
    with pytest.raises(ValueError, match=f"{section} .*weight_decy"):
        check_config_keys(cfg)


def test_old_shrink_knob_name_is_rejected(synth, run_root):
    """改名前的 key 不能被靜默忽略,train() 開訓前就要擋下。"""
    cfg = synthetic_cfg("old_knob", synth.seed, epochs=1)
    cfg["train"]["max_steps_reestimate_every"] = 1
    with pytest.raises(ValueError, match="max_steps_reestimate_every"):
        run_train(cfg, synth.data, run_root)


# ============================================================================
# C. 出界的四個邊界:沒 checkpoint 時從頭、epoch 中途、有 checkpoint 時退回、最後一個 epoch
# ============================================================================
# 出界落在 epoch 0 的測試只跑 1 個 epoch,跟參考訓練 epoch 0 的快照比(少跑的 epoch 只會多出
# 跟邊界無關的重來、多花編譯時間)。

def test_overflow_on_first_batch_restarts_from_scratch(synth, run_root, synth_reference):
    """conv1 佇列容量 1:第一個 batch(需求 3)就出界,還沒套用任何更新,從頭重來。"""
    cfg = synthetic_cfg("reinit", synth.seed, epochs=1, conv1={"max_queue_len": 1})
    result = run_train(cfg, synth.data, run_root)

    first = _grow_events(result)[0]
    assert (first["epoch"], first["batch"], first["resumed_from_epoch"]) == (0, 0, None)
    change = _change(first, "conv1", "max_queue_len")
    assert (change["old"], change["observed"]) == (1, batch_needs(synth, 0)[0])
    _assert_grown_by_formula(change, SYNTH_GROW)
    _assert_params_close(result.params, _snapshot_params(synth_reference, 0),
                         "從頭重來 vs 參考訓練")


def test_overflow_mid_epoch_discards_updates_already_applied(synth, run_root, synth_reference):
    """conv1 佇列容量 3:batch 0(需求 3)放得下、已經套用更新,batch 1(需求 5)出界。
    batch 0 的更新沒丟掉的話,最終權重會跟參考訓練差很多。"""
    cfg = synthetic_cfg("midepoch", synth.seed, epochs=1,
                        conv1={"max_queue_len": batch_needs(synth, 0)[0]})
    result = run_train(cfg, synth.data, run_root)

    first = _grow_events(result)[0]
    assert (first["epoch"], first["batch"], first["resumed_from_epoch"]) == (0, 1, None)
    assert _change(first, "conv1", "max_queue_len")["observed"] == batch_needs(synth, 0)[1]
    _assert_params_close(result.params, _snapshot_params(synth_reference, 0),
                         "epoch 中途出界 vs 參考訓練")


def _overflow_after_checkpoint_cfg(synth, run_name: str, epochs: int) -> dict:
    """conv1 佇列容量 = epoch 0 的最大需求 9:epoch 0 跑完,epoch 1 遇到需求 12 的樣本才出界。"""
    return synthetic_cfg(run_name, synth.seed, epochs=epochs,
                         conv1={"max_queue_len": max(batch_needs(synth, 0))})


def _batch_of_largest_sample(synth, epoch: int) -> int:
    return batch_needs(synth, epoch).index(int(synth.needs.max()))


def test_overflow_after_checkpoint_resumes_from_last_finished_epoch(synth, run_root,
                                                                    synth_reference):
    """epoch 1 出界時退回 epoch 0 的 checkpoint,不是從頭:metrics.csv 每個 epoch 只有一列。"""
    epochs = 3
    result = run_train(_overflow_after_checkpoint_cfg(synth, "resume", epochs), synth.data,
                       run_root)

    [event] = _grow_events(result)
    assert (event["epoch"], event["batch"], event["resumed_from_epoch"]) == (
        1, _batch_of_largest_sample(synth, 1), 0)
    change = _change(event, "conv1", "max_queue_len")
    assert (change["old"], change["observed"]) == (9, 12)

    rows = _read_metrics_csv(result.exp_dir)
    assert [int(r["epoch"]) for r in rows] == list(range(epochs))
    assert [int(r["conv1_max_event_queue"]) for r in rows] == [9, change["new"], change["new"]]
    assert _final_capacity(result, "conv1")["max_queue_len"] == change["new"]
    with np.load(os.path.join(result.exp_dir, TRAIN_DIRNAME, "checkpoint.npz")) as ckpt:
        assert int(ckpt["epoch"]) == epochs - 1
    _assert_params_close(result.params, synth_reference.params, "退回 checkpoint vs 參考訓練")


def test_overflow_on_final_epoch_is_still_handled(synth, run_root, synth_reference):
    """同一個設定只跑 2 epoch:出界落在最後一個 epoch,照樣退回重跑,不會被當成訓練結束。"""
    epochs = 2
    result = run_train(_overflow_after_checkpoint_cfg(synth, "final_epoch", epochs), synth.data,
                       run_root)

    [event] = _grow_events(result)
    assert (event["epoch"], event["resumed_from_epoch"]) == (epochs - 1, epochs - 2)
    rows = _read_metrics_csv(result.exp_dir)
    assert [int(r["epoch"]) for r in rows] == list(range(epochs))
    assert result.run_record["best"]["epoch"] in range(epochs)
    _assert_params_close(result.params, _snapshot_params(synth_reference, epochs - 1),
                         "最後一個 epoch 出界 vs 參考訓練同一個 epoch 的快照")


# ============================================================================
# D. 哪一層、哪個旋鈕出界
# ============================================================================

def test_only_conv2_overflowing_grows_only_conv2(synth, run_root, synth_reference):
    """conv2 佇列容量設成第一個 batch 用初始權重量到的需求減 1:(0, 0) 出界,只放大 conv2。"""
    needed = first_batch_needed(synthetic_cfg("measure", synth.seed),
                                synth)["conv2"]["max_queue_len"]
    assert needed >= 2
    cfg = synthetic_cfg("conv2_only", synth.seed, epochs=1, conv2={"max_queue_len": needed - 1})
    result = run_train(cfg, synth.data, run_root)

    first = _grow_events(result)[0]
    assert (first["epoch"], first["batch"], first["resumed_from_epoch"]) == (0, 0, None)
    assert {c["layer"] for c in first["changes"]} == {"conv2"}
    change = _change(first, "conv2", "max_queue_len")
    assert change["observed"] == needed
    _assert_grown_by_formula(change, SYNTH_GROW)
    _assert_params_close(result.params, _snapshot_params(synth_reference, 0), "conv2 出界 vs 參考訓練")


def test_both_convs_overflowing_grow_in_the_same_event(synth, run_root, synth_reference):
    """conv1 佇列容量 1 時,conv2 吃到的是被截過的輸入;conv2 的容量照同一個截斷情況量出的
    需求減 1。第一次出界時兩層在同一個事件裡一起放大。"""
    conv1 = {"max_queue_len": 1}
    needed = first_batch_needed(synthetic_cfg("measure", synth.seed, conv1=conv1),
                                synth)["conv2"]["max_queue_len"]
    assert needed >= 2
    cfg = synthetic_cfg("both", synth.seed, epochs=1, conv1=conv1,
                        conv2={"max_queue_len": needed - 1})
    result = run_train(cfg, synth.data, run_root)

    first = _grow_events(result)[0]
    assert (first["epoch"], first["batch"]) == (0, 0)
    assert _change(first, "conv1", "max_queue_len")["observed"] == batch_needs(synth, 0)[0]
    assert _change(first, "conv2", "max_queue_len")["observed"] == needed
    _assert_params_close(result.params, _snapshot_params(synth_reference, 0),
                         "兩層同時出界 vs 參考訓練")


def test_output_spike_overflow_grows_max_out_spikes(synth, run_root, synth_reference):
    """conv1 輸出 spike 容量設成第一個 batch 用初始權重量到的需求減 1:(0, 0) 出界。"""
    needed = first_batch_needed(synthetic_cfg("measure", synth.seed),
                                synth)["conv1"]["max_out_spikes"]
    assert needed >= 2
    cfg = synthetic_cfg("max_out", synth.seed, epochs=1, conv1={"max_out_spikes": needed - 1})
    result = run_train(cfg, synth.data, run_root)

    first = _grow_events(result)[0]
    assert (first["epoch"], first["batch"], first["resumed_from_epoch"]) == (0, 0, None)
    change = _change(first, "conv1", "max_out_spikes")
    assert change["observed"] == needed
    _assert_grown_by_formula(change, SYNTH_GROW)
    _assert_params_close(result.params, _snapshot_params(synth_reference, 0),
                         "輸出 spike 出界 vs 參考訓練")


@pytest.mark.parametrize("knob", ["max_out_spikes", "max_extra_steps"])
def test_hidden_fc_overflow_grows_only_hidden_fc(synth, run_root, synth_reference, knob):
    """隱藏 FC(會 fire、chunk_size > 1)的旋鈕設成第一個 batch 用初始權重量到的需求減 1:
    (0, 0) 出界,只放大隱藏 FC。"""
    needed = first_batch_needed(synthetic_cfg("measure", synth.seed), synth)["hidden"][knob]
    assert needed >= 1, "隱藏 FC 在第一個 batch 要真的 fire、fire 要多花步數,這個測試才有意義"
    cfg = synthetic_cfg(f"hidden_{knob}", synth.seed, epochs=1, hidden={knob: needed - 1})
    result = run_train(cfg, synth.data, run_root)

    first = _grow_events(result)[0]
    assert (first["epoch"], first["batch"], first["resumed_from_epoch"]) == (0, 0, None)
    assert {c["layer"] for c in first["changes"]} == {"hidden"}
    change = _change(first, "hidden", knob)
    assert change["observed"] == needed
    _assert_grown_by_formula(change, SYNTH_GROW)
    _assert_params_close(result.params, _snapshot_params(synth_reference, 0),
                         f"隱藏 FC {knob} 出界 vs 參考訓練")


def test_repeated_overflows_follow_every_new_record_need(synth, run_root):
    """conv1 佇列容量 1、倍率 1.01:每遇到比目前容量大的 batch 就出界一次,放大到 ceil(需求 * 1.01)。
    epoch 0 的需求 3, 5, 7, 9 各出界一次,epoch 1 的 12 再一次。"""
    grow, epochs = 1.01, 2
    cfg = synthetic_cfg("repeated", synth.seed, conv1={"max_queue_len": 1}, grow=grow,
                        epochs=epochs)
    result = run_train(cfg, synth.data, run_root)

    expected, capacity = [], 1
    for epoch in range(epochs):
        for batch, need in enumerate(batch_needs(synth, epoch)):
            if need > capacity:
                expected.append((epoch, batch, need))
                capacity = math.ceil(need * grow)
    events = _grow_events(result)
    assert [(e["epoch"], e["batch"], _change(e, "conv1", "max_queue_len")["observed"])
            for e in events] == expected
    rows = _read_metrics_csv(result.exp_dir)
    assert [int(r["epoch"]) for r in rows] == list(range(epochs))
    assert int(rows[-1]["conv1_max_event_queue"]) == capacity


# ============================================================================
# E. 跨行程續練
# ============================================================================

class _Crash(Exception):
    """模擬訓練行程在存完 checkpoint 之後當掉。"""


def test_resume_after_crash_continues_from_checkpoint(synth, run_root, synth_reference,
                                                       monkeypatch):
    """conv1 佇列容量 3:epoch 0 在 (0, 1) 出界一次;存完 epoch 0 的 checkpoint 就當掉。
    同一個目錄重新呼叫 train() 接著練:epoch 1 在需求 12 的樣本出界、退回 epoch 0。
    紀錄(metrics 列、事件)要接得上,權重跟參考訓練一致。"""
    cfg = synthetic_cfg("resume_after_crash", synth.seed,
                        conv1={"max_queue_len": batch_needs(synth, 0)[0]})
    exp_dir = make_exp_dir(cfg["run_name"], str(run_root))
    save = Checkpointer.save

    def save_then_crash(self, state):
        save(self, state)
        raise _Crash

    monkeypatch.setattr(Checkpointer, "save", save_then_crash)
    with pytest.raises(_Crash):
        train(cfg, synth.data, exp_dir)
    monkeypatch.undo()
    result = train(cfg, synth.data, exp_dir)

    assert [e["from_epoch"] for e in result.run_record["resumes"]] == [1]
    assert [(e["epoch"], e["batch"], e["resumed_from_epoch"]) for e in _grow_events(result)] == [
        (0, 1, None), (1, _batch_of_largest_sample(synth, 1), 0)]
    rows = _read_metrics_csv(result.exp_dir)
    reference_rows = _read_metrics_csv(synth_reference.exp_dir)
    assert [int(r["epoch"]) for r in rows] == [0, 1, 2]
    for row, reference_row in zip(rows, reference_rows):
        assert float(row["train_loss"]) == pytest.approx(float(reference_row["train_loss"]),
                                                         rel=_TRAIN_RESULT_TOL)
    assert result.run_record["best"]["epoch"] == synth_reference.run_record["best"]["epoch"]
    _assert_params_close(result.params, synth_reference.params, "當掉後接著練 vs 參考訓練")


def test_train_refuses_to_continue_a_finished_run(synth, synth_reference):
    cfg = synthetic_cfg("reference", synth.seed)
    with pytest.raises(ValueError, match="已經跑完"):
        train(cfg, synth.data, synth_reference.exp_dir)
