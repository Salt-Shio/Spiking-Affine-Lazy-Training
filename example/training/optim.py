"""optimizer 組裝:AdamW,選配餘弦退火跟梯度裁剪。

各選項的理由見 docs/math/梯度下降曲率穩定性推導.md。
"""
import optax


def build_learning_rate(train_cfg: dict, n_train: int, batch_size: int):
    """train_cfg["lr"];train_cfg 的 lr_cosine_decay 開啟時回傳餘弦退火 schedule。

    退火總步數 = epochs * (n_train // batch_size),用規劃的 epoch 數算。
    出界退回 checkpoint 時,schedule 的進度跟著 opt_state 一起退回。
    lr_cosine_alpha:退火到最後剩 lr 的幾倍,預設 0。
    """
    if not train_cfg.get("lr_cosine_decay", False):
        return train_cfg["lr"]
    total_steps = train_cfg["epochs"] * (n_train // batch_size)
    return optax.cosine_decay_schedule(
        init_value=train_cfg["lr"], decay_steps=total_steps,
        alpha=train_cfg.get("lr_cosine_alpha", 0.0))


def build_optimizer(train_cfg: dict, n_train: int, batch_size: int):
    """AdamW(weight_decay 預設 0);train_cfg 有 grad_clip_norm 時,梯度先照整體範數裁剪再交給 AdamW。"""
    lr = build_learning_rate(train_cfg, n_train, batch_size)
    adamw = optax.adamw(lr, weight_decay=train_cfg.get("weight_decay", 0.0))
    grad_clip_norm = train_cfg.get("grad_clip_norm")
    if grad_clip_norm is None:
        return adamw
    return optax.chain(optax.clip_by_global_norm(grad_clip_norm), adamw)
