"""init_k 尺度推導的數值驗證(docs/math/初始權重尺度推導.md 步驟 1–5)。

這不是 pass/fail 單元測試,是一份「跑出數字、印成表」的驗證報告——確認推導的
代數沒推歪。對應文件的驗證計畫 V1–V5:

  V1  純 numpy 自由遞迴 -> 穩態變異數 σ_V^2 = σ_w^2 / (1-ρ)           步驟 1
  V2  真實 forward -> 感受野正規化 firing rate vs 單筆越界機率 p_SE     步驟 3
  V3  no-fire forward(v_th 設超大)-> 經驗 σ_V、P(V≥v_th)、
      E[slope | V≥v_th] 及其隨 init_k 的 1/k^2 衰減                     步驟 1 / 4
  V4  真實 forward + grad -> ‖∂L/∂W‖ 逐層對 init_k 的關係              步驟 4
  V5  綜合:firing_rate(V2)× slope_proxy(V3)是否解釋 grad_norm(V4);
      低 init_k 那側 slope 維持在 α/2、梯度不塌                        步驟 5

用法:
  python -m example.analysis.verify_init_k           # 跑全部 V1–V5,印表
  pytest example/analysis/verify_init_k.py           # 只跑 test_v1_*(純 numpy、快)
"""
import math
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import optax

from data.src.nmnist import NMNISTDataset
from salt_core.connectivity.conv import _axis_candidates, unravel_conv_source
from salt_core.stream import EventStream
from salt_core.capacity import GrowthPolicy, grown_to_fit, grown_to_fit_batch, reduce_over_batch
from example.models.conv_net import build_decoder, build_network
from example.paths import DATASET_ROOT, resolve_config
from example.utils import load_config, split_raw_events

# 掃描的 init_k;√3 = Lee 變異數保持、8/64 = 舊 firing-rate 準則。
INIT_KS = (math.sqrt(3.0), 3.0, 5.0, 8.0, 64.0)
# conv 幾何 fan_in:conv1 = ic·k² = 2·9、conv2 = 8·9。
FAN_IN = {"conv1": 18, "conv2": 72}
V_TH = 1.0
ALPHA = 2.0
# 容量只求放得下,放大倍率不影響量到的數值,用預設公式。
FIT_POLICY = GrowthPolicy()
CHUNK = 16


def _sigma_w(init_k: float, fan_in: int) -> float:
    """單筆權重標準差 σ_w = init_k / sqrt(3·fan_in)(U(-limit,limit),limit=init_k/√fan_in)。"""
    return init_k / math.sqrt(3.0 * fan_in)


def _p_se(init_k: float, fan_in: int, v_th: float = V_TH) -> float:
    """單筆事件從 V≈0 直接越過門檻的機率 = max(0, (1 - √fan_in/init_k)/2)。"""
    limit = init_k / math.sqrt(fan_in)
    return max(0.0, 0.5 * (1.0 - v_th / limit))


def _slope(u, alpha: float = ALPHA):
    """ATan surrogate 斜率,u = V - v_th。α=2 時 = 1/(1+(π u)²)。"""
    a = alpha / 2.0
    return a / (1.0 + (math.pi * a * np.asarray(u)) ** 2)


def receptive_field_tap_count(x: jax.Array, y: jax.Array, S: int, P: int,
                               H_out: int, W_out: int, K: int,
                               n_real_events: jax.Array | int) -> jax.Array:
    """每個空間輸出位置有幾筆真事件是它的合法 tap。

    只看事件座標、K/S/P、n_real_events,不看 channel 跟權重。
    回傳 (H_out*W_out,) int32。
    """
    x = jnp.asarray(x, dtype=jnp.int32)
    y = jnp.asarray(y, dtype=jnp.int32)
    n_events = x.shape[0]
    N = (K - 1) // S + 1
    o_y, valid_y, _ = _axis_candidates(y, K, S, P, N, H_out)  # (n_events, N)
    o_x, valid_x, _ = _axis_candidates(x, K, S, P, N, W_out)  # (n_events, N)
    valid_2d = valid_y[:, :, None] & valid_x[:, None, :]        # (n_events, N, N)
    is_real = jnp.arange(n_events) < jnp.asarray(n_real_events, dtype=jnp.int32)
    valid_2d = valid_2d & is_real[:, None, None]

    o_flat = o_y[:, :, None] * W_out + o_x[:, None, :]          # (n_events, N, N)
    o_flat = jnp.where(valid_2d, o_flat, H_out * W_out).reshape(-1)  # 不合法標成越界
    counts = jnp.zeros((H_out * W_out,), dtype=jnp.int32)
    return counts.at[o_flat].add(valid_2d.reshape(-1).astype(jnp.int32), mode='drop')


def conv_layer_receptive_field_firing_rate(spike_mask: jax.Array, x: jax.Array, y: jax.Array,
                                            S: int, P: int, H_out: int, W_out: int, K: int,
                                            OC: int, n_real_events: jax.Array | int
                                            ) -> jax.Array:
    """每顆神經元 spike 數 / 自己的感受野事件數,對感受野事件數 > 0 的神經元取平均。

    感受野事件數 0 的神經元這個樣本沒機會 fire,不算進平均。
    spike_mask: (OC*H_out*W_out, max_steps)。回傳純量。
    """
    spatial = receptive_field_tap_count(x, y, S, P, H_out, W_out, K, n_real_events)
    opportunity = jnp.tile(spatial, OC)  # (OC*H_out*W_out,)
    spike_count = jnp.sum(spike_mask, axis=1)
    has_opp = opportunity > 0
    rate_per_neuron = jnp.where(has_opp, spike_count / jnp.maximum(opportunity, 1), 0.0)
    return jnp.sum(rate_per_neuron) / jnp.maximum(jnp.sum(has_opp), 1)


def calibration_measure(layer, calib_stream_batch: EventStream, chunk: int = 16):
    """回傳 measure(weight) -> 純量:一批輸入流的感受野正規化 firing rate,對樣本取平均。

    layer: ConvLayer,容量要放得下這批輸入(見 _fit_layer)。
    分批 vmap,避免整批一次建壓縮佇列 OOM。
    """
    n = calib_stream_batch.event_times.shape[0]

    def measure(w: jax.Array) -> float:
        def one(s: EventStream):
            result = layer.forward(w, s).result
            x, y, _c = unravel_conv_source(s.event_source_idx, layer.h_in, layer.w_in)
            return conv_layer_receptive_field_firing_rate(
                result.spike_mask, x, y, layer.s, layer.p, layer.h_out, layer.w_out,
                layer.k, layer.oc, s.n_real_events)

        total = 0.0
        for lo in range(0, n, chunk):
            total += float(jnp.sum(jax.vmap(one)(_slice_stream(calib_stream_batch, lo,
                                                                min(lo + chunk, n)))))
        return total / n

    return measure


# ----------------------------------------------------------------------------
# V1:純 numpy,穩態膜電位變異數
# ----------------------------------------------------------------------------
def run_v1(tau: float = 16.0, n_chains: int = 20000, n_steps: int = 400,
           mean_gaps=(1.0, 2.0, 4.0, 10.0), seed: int = 0) -> list[dict]:
    """自由遞迴 V_k = a_k V_{k-1} + w_k,V_0 = 0,w_k ~ U(-limit, limit),
    間隔 Δ_k ~ Geometric(mean = mean_gap)。比對經驗 Var(V_∞) 跟
    σ_w²/(1-ρ),ρ = E[(1-1/τ)^{2Δ}] 用同一批 Δ 的樣本平均估。"""
    rng = np.random.default_rng(seed)
    decay = 1.0 - 1.0 / tau
    rows = []
    for mean_gap in mean_gaps:
        # Geometric on {1,2,...},平均 = 1/p  ->  p = 1/mean_gap。
        p = 1.0 / mean_gap
        deltas = rng.geometric(p, size=(n_steps, n_chains)).astype(np.float64)
        a = decay ** deltas                       # (n_steps, n_chains)
        rho_hat = float(np.mean(a ** 2))
        n_eff = 1.0 / (1.0 - rho_hat)
        for init_k in INIT_KS:
            limit = init_k / math.sqrt(FAN_IN["conv1"])
            sigma_w2 = limit ** 2 / 3.0
            w = rng.uniform(-limit, limit, size=(n_steps, n_chains))
            v = np.zeros(n_chains)
            for t in range(n_steps):
                v = a[t] * v + w[t]
            var_emp = float(np.var(v))
            var_pred = sigma_w2 / (1.0 - rho_hat)
            rows.append(dict(mean_gap=mean_gap, rho=rho_hat, n_eff=n_eff,
                             init_k=init_k, sigma_w=math.sqrt(sigma_w2),
                             var_emp=var_emp, var_pred=var_pred,
                             ratio=var_emp / var_pred))
    return rows


def _print_v1(rows: list[dict]) -> None:
    print("\n=== V1  穩態膜電位變異數(純 numpy,步驟 1)===")
    print(f"{'mean_gap':>8} {'rho':>7} {'N_eff':>7} {'init_k':>8} "
          f"{'sigma_w':>8} {'Var_emp':>10} {'Var_pred':>10} {'emp/pred':>9}")
    for r in rows:
        print(f"{r['mean_gap']:>8.1f} {r['rho']:>7.4f} {r['n_eff']:>7.2f} "
              f"{r['init_k']:>8.4f} {r['sigma_w']:>8.4f} {r['var_emp']:>10.4f} "
              f"{r['var_pred']:>10.4f} {r['ratio']:>9.4f}")
    worst = max(abs(r["ratio"] - 1.0) for r in rows)
    print(f"  最大相對偏差 |emp/pred - 1| = {worst:.4f}  (應 << 0.05)")


# ----------------------------------------------------------------------------
# forward 驗證共用:載資料、建 baseline 層、生權重、逐層 forward
# ----------------------------------------------------------------------------
def _load_layers_and_stream(n_samples: int):
    cfg = load_config(str(resolve_config("configs/conv/baseline.yaml")))
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    network = build_network(model_cfg)
    layers = list(network.layers)

    ds = NMNISTDataset(DATASET_ROOT, max_events=data_cfg["max_events"])
    split = ds.build_split(seed=data_cfg["seed_train"],
                           n_samples=max(n_samples, data_cfg["train_size"]),
                           which="train")
    sl = jax.tree_util.tree_map(lambda a: a[:n_samples], split)
    stream = jax.vmap(network.input_stream)(split_raw_events(sl))
    return model_cfg, layers, split, sl, stream


def _slice_stream(sb, lo: int, hi: int):
    return type(sb)(*(f[lo:hi] for f in sb))


def _sub_batched(fn, stream_batch, chunk: int):
    """對 stream_batch 分 chunk 做 jax.vmap(fn),沿 axis 0 串回。"""
    n = stream_batch.event_times.shape[0]
    parts = [jax.vmap(fn)(_slice_stream(stream_batch, lo, min(lo + chunk, n)))
             for lo in range(0, n, chunk)]
    return jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *parts)


def _fit_layer(layer, w, stream_batch, chunk: int):
    """容量放大到放得下這批輸入為止。回傳 (放得下的層, 這層的逐筆輸出流)。

    config 的容量是訓練用的小起始值,不放大的話 forward 會被截斷。
    """
    while True:
        out, diag = _sub_batched(lambda s: _out_and_diag(layer.forward(w, s)),
                                 stream_batch, chunk)
        [grown] = grown_to_fit([layer], {layer.name: FIT_POLICY}, [reduce_over_batch(diag)])
        if grown is layer:
            return layer, out
        layer = grown


def _out_and_diag(output):
    return output.stream, output.diag


def _weights_for(layers, init_ks: dict, key0: int = 0):
    """每層用固定 key、只換 init_k 生權重(對齊 uniform_init 的設計意圖)。"""
    keys = jax.random.split(jax.random.PRNGKey(key0), len(layers))
    ws = []
    for layer, k in zip(layers, keys):
        ik = init_ks.get(layer.name, layer.init_k)
        ws.append(replace(layer, init_k=float(ik)).init_weight(k))
    return tuple(ws)


# ----------------------------------------------------------------------------
# V2:感受野正規化 firing rate vs p_SE
# ----------------------------------------------------------------------------
def run_v2(n_samples: int = 128) -> list[dict]:
    _mc, layers, _split, _sl, stream = _load_layers_and_stream(n_samples)
    conv1, conv2 = layers[0], layers[1]
    rows = []
    for init_k in INIT_KS:
        ws = _weights_for(layers, {"conv1": init_k, "conv2": init_k})
        conv1_fit, conv1_out = _fit_layer(conv1, ws[0], stream, CHUNK)
        conv2_fit, _ = _fit_layer(conv2, ws[1], conv1_out, CHUNK)
        fr1 = float(calibration_measure(conv1_fit, stream, CHUNK)(ws[0]))
        fr2 = float(calibration_measure(conv2_fit, conv1_out, CHUNK)(ws[1]))
        rows.append(dict(init_k=init_k,
                         fr1=fr1, p_se1=_p_se(init_k, FAN_IN["conv1"]),
                         fr2=fr2, p_se2=_p_se(init_k, FAN_IN["conv2"])))
    return rows


def _print_v2(rows: list[dict]) -> None:
    print("\n=== V2  感受野正規化 firing rate vs 單筆越界機率 p_SE(步驟 3)===")
    print(f"{'init_k':>8} | {'conv1 fr':>9} {'conv1 p_SE':>11} | "
          f"{'conv2 fr':>9} {'conv2 p_SE':>11}")
    for r in rows:
        print(f"{r['init_k']:>8.4f} | {r['fr1']:>9.4f} {r['p_se1']:>11.4f} | "
              f"{r['fr2']:>9.4f} {r['p_se2']:>11.4f}")
    print("  預期:limit<v_th(conv2 小 init_k)時 p_SE=0、fr 僅來自累積;"
          "limit>v_th 時 fr 隨 p_SE 一起上升")


# ----------------------------------------------------------------------------
# V3:no-fire 膜電位分布 -> σ_V、P(V≥v_th)、E[slope | V≥v_th]
# ----------------------------------------------------------------------------
def _vfinal_stats(layer, w, stream_batch, chunk: int) -> np.ndarray:
    layer, _ = _fit_layer(layer, w, stream_batch, chunk)
    vf = _sub_batched(lambda s: layer.forward(w, s).result.v_final, stream_batch, chunk)
    return np.asarray(vf).reshape(-1)


def run_v3(n_samples: int = 128) -> list[dict]:
    _mc, layers, _split, _sl, stream = _load_layers_and_stream(n_samples)
    conv1, conv2 = layers[0], layers[1]
    nofire1 = replace(conv1, v_th=1e12)
    nofire2 = replace(conv2, v_th=1e12)
    rows = []
    for init_k in INIT_KS:
        ws = _weights_for(layers, {"conv1": init_k, "conv2": init_k})
        v1 = _vfinal_stats(nofire1, ws[0], stream, CHUNK)
        _, conv1_out = _fit_layer(conv1, ws[0], stream, CHUNK)
        v2 = _vfinal_stats(nofire2, ws[1], conv1_out, CHUNK)
        rows.append(dict(init_k=init_k,
                         **_one_layer_v3("conv1", init_k, v1),
                         **{f"c2_{k}": v for k, v in _one_layer_v3("conv2", init_k, v2).items()}))
    return rows


def _one_layer_v3(name: str, init_k: float, v: np.ndarray) -> dict:
    sigma_emp = float(np.std(v))
    sigma_w = _sigma_w(init_k, FAN_IN[name])
    n_eff_impl = (sigma_emp / sigma_w) ** 2 if sigma_w > 0 else float("nan")
    over = v[v >= V_TH] - V_TH
    p_cross = float(np.mean(v >= V_TH))
    slope_mean = float(np.mean(_slope(over))) if over.size else float("nan")
    return dict(sigma_emp=sigma_emp, sigma_w=sigma_w, n_eff_impl=n_eff_impl,
               p_cross=p_cross, slope_mean=slope_mean)


def _print_v3(rows: list[dict]) -> None:
    print("\n=== V3  no-fire 膜電位:σ_V、P(V≥v_th)、E[slope|V≥v_th](步驟 1 / 4)===")
    for tag, pfx in (("conv1", ""), ("conv2", "c2_")):
        print(f"  [{tag}]  {'init_k':>8} {'σ_emp':>8} {'σ_w':>8} {'N_eff*':>7} "
              f"{'P(V≥1)':>8} {'E[slope]':>9}")
        for r in rows:
            print(f"        {r['init_k']:>8.4f} {r[pfx+'sigma_emp']:>8.4f} "
                  f"{r[pfx+'sigma_w']:>8.4f} {r[pfx+'n_eff_impl']:>7.2f} "
                  f"{r[pfx+'p_cross']:>8.4f} {r[pfx+'slope_mean']:>9.5f}")
    print("  N_eff* = (σ_emp/σ_w)²,應在個位數且大致穩定;E[slope] 應隨 init_k 快速下降(~1/k²)")


# ----------------------------------------------------------------------------
# V4:‖∂L/∂W‖ 逐層對 init_k
# ----------------------------------------------------------------------------
def _fit_network(network, params, raw):
    """整個網路的容量放大到放得下這批樣本為止,回傳放得下的網路。"""
    layers = network.layers
    policies = {layer.name: FIT_POLICY for layer in layers if layer.capacity is not None}
    while True:
        diags = network.replace_layers(layers).apply_batched(params, raw).diags
        grown = grown_to_fit_batch(layers, policies, diags)
        if grown is layers:
            return network.replace_layers(layers)
        layers = grown


def run_v4(n_samples: int = 16) -> list[dict]:
    model_cfg, layers, _split, sl, _stream = _load_layers_and_stream(n_samples)
    decoder = build_decoder(model_cfg, layers)
    labels_oh = sl.labels_onehot
    raw = split_raw_events(sl)
    network = build_network(model_cfg)

    def loss_fn(params, net):
        scores, _ = jax.vmap(decoder.decode)(net.apply_batched(params, raw).last)
        return jnp.mean(optax.softmax_cross_entropy(scores, labels_oh))

    rows = []
    for init_k in INIT_KS:
        params = _weights_for(layers, {"conv1": init_k, "conv2": init_k})
        net = _fit_network(network, params, raw)
        loss, grad = jax.value_and_grad(loss_fn)(params, net)
        gn = {layer.name: float(jnp.linalg.norm(g)) for layer, g in zip(layers, grad)}
        rows.append(dict(init_k=init_k, loss=float(loss), **gn))
    return rows


def _print_v4(rows: list[dict], layer_names: list[str]) -> None:
    print("\n=== V4  ‖∂L/∂W‖ 逐層 vs init_k(步驟 4)===")
    head = f"{'init_k':>8} {'loss':>8} " + " ".join(f"{n:>12}" for n in layer_names)
    print(head)
    for r in rows:
        print(f"{r['init_k']:>8.4f} {r['loss']:>8.4f} "
              + " ".join(f"{r[n]:>12.3e}" for n in layer_names))
    print("  預期:conv2 的 ‖grad‖ 在 init_k 8/64 明顯小於 5(飽和區梯度塌)")


# ----------------------------------------------------------------------------
# V5:綜合 —— firing_rate × slope_proxy 是否解釋 grad_norm
# ----------------------------------------------------------------------------
def run_v5(v2_rows, v3_rows, v4_rows) -> list[dict]:
    rows = []
    for r2, r3, r4 in zip(v2_rows, v3_rows, v4_rows):
        assert r2["init_k"] == r3["init_k"] == r4["init_k"]
        fr = r2["fr2"]
        slope = r3["c2_slope_mean"]
        proxy = fr * slope
        rows.append(dict(init_k=r2["init_k"], fr2=fr, slope_proxy=slope,
                         fr_x_slope=proxy, grad_conv2=r4["conv2"]))
    # 以 init_k=5 當基準,看 grad 的相對變化跟 proxy 的相對變化是否同向。
    base = next(r for r in rows if abs(r["init_k"] - 5.0) < 1e-6)
    for r in rows:
        r["grad_rel"] = r["grad_conv2"] / base["grad_conv2"]
        r["proxy_rel"] = r["fr_x_slope"] / base["fr_x_slope"] if base["fr_x_slope"] else float("nan")
    return rows


def _print_v5(rows: list[dict]) -> None:
    print("\n=== V5  綜合:firing_rate × slope_proxy vs grad_norm(conv2,步驟 5)===")
    print(f"{'init_k':>8} {'fr2':>8} {'slope':>9} {'fr×slope':>10} "
          f"{'‖grad‖':>11} {'grad/基準':>10} {'proxy/基準':>11}")
    for r in rows:
        print(f"{r['init_k']:>8.4f} {r['fr2']:>8.4f} {r['slope_proxy']:>9.5f} "
              f"{r['fr_x_slope']:>10.5f} {r['grad_conv2']:>11.3e} "
              f"{r['grad_rel']:>10.3f} {r['proxy_rel']:>11.3f}")
    print("  預期:grad/基準 與 proxy/基準 同向;init_k < 5 那側 slope 接近 α/2、grad 不塌")


# ----------------------------------------------------------------------------
# pytest 可見的快速檢查(只有 V1,純 numpy)
# ----------------------------------------------------------------------------
def test_v1_stationary_variance():
    rows = run_v1(n_chains=4000, n_steps=300)
    worst = max(abs(r["ratio"] - 1.0) for r in rows)
    assert worst < 0.08, f"穩態變異數經驗/理論偏差過大:{worst:.4f}"


def test_v1_sigma_v_linear_in_init_k():
    """σ_V ∝ init_k:同一個 mean_gap 下,√Var_emp 對 init_k 應接近正比。"""
    rows = [r for r in run_v1(n_chains=4000, n_steps=300, mean_gaps=(4.0,))]
    ks = np.array([r["init_k"] for r in rows])
    sig = np.array([math.sqrt(r["var_emp"]) for r in rows])
    ratio = sig / ks
    assert np.max(ratio) / np.min(ratio) - 1.0 < 0.08, f"σ_V/init_k 不夠穩定:{ratio}"


def main() -> None:
    v1 = run_v1()
    _print_v1(v1)

    v2 = run_v2()
    _print_v2(v2)

    v3 = run_v3()
    _print_v3(v3)

    _mc, layers, *_ = _load_layers_and_stream(2)
    layer_names = [l.name for l in layers]
    v4 = run_v4()
    _print_v4(v4, layer_names)

    v5 = run_v5(v2, v3, v4)
    _print_v5(v5)


if __name__ == "__main__":
    main()
