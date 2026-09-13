"""viz/channel_grid.py 的單元測試(合成陣列,不需要訓練 run)。"""
import numpy as np

import matplotlib
matplotlib.use("Agg")

from viz.channel_grid import ImageGridPlot, unflatten_channels


def test_unflatten_channels_matches_channel_major_order():
    oc, h, w = 2, 3, 4
    flat = np.arange(oc * h * w)

    out = unflatten_channels(flat, oc, h, w)

    assert out.shape == (oc, h, w)
    # channel-major:flat_index = c*h*w + y*w + x,第 0 個 channel 應該是
    # 0..(h*w-1) 這段連續數字 reshape 回 (h, w)。
    assert np.array_equal(out[0], np.arange(h * w).reshape(h, w))
    assert np.array_equal(out[1], np.arange(h * w, 2 * h * w).reshape(h, w))


def test_unflatten_channels_wrong_length_raises():
    flat = np.zeros(10)
    try:
        unflatten_channels(flat, oc=2, h=3, w=4)   # 2*3*4=24 != 10
    except ValueError:
        pass
    else:
        raise AssertionError("預期長度對不上要拋 ValueError")


def _grid_axes(fig):
    """`fig.axes` 混了 `colorbar()` 額外加的 axes,只留網格本體那些。"""
    return [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]


def test_render_creates_one_axes_per_image_with_titles():
    images = [np.random.rand(5, 5) for _ in range(4)]
    titles = ["a", "b", "c", "d"]

    fig = ImageGridPlot(ncols=2).render(images, titles=titles)

    visible = [ax for ax in _grid_axes(fig) if ax.get_visible()]
    assert len(visible) == 4
    assert {ax.get_title() for ax in visible} == set(titles)


def test_render_without_titles_defaults_to_blank():
    images = [np.random.rand(3, 3) for _ in range(3)]

    fig = ImageGridPlot(ncols=2).render(images)

    grid = _grid_axes(fig)
    visible = [ax for ax in grid if ax.get_visible()]
    hidden = [ax for ax in grid if not ax.get_visible()]
    assert len(visible) == 3
    assert len(hidden) == 1
    assert all(ax.get_title() == "" for ax in visible)


def test_render_mismatched_titles_length_raises():
    images = [np.random.rand(2, 2) for _ in range(2)]
    try:
        ImageGridPlot().render(images, titles=["only-one"])
    except ValueError:
        pass
    else:
        raise AssertionError("預期 titles 長度不對要拋 ValueError")


def test_render_empty_images_raises():
    try:
        ImageGridPlot().render([])
    except ValueError:
        pass
    else:
        raise AssertionError("預期 render([]) 要拋 ValueError")


def test_render_bool_image_uses_two_value_colorbar():
    img = np.array([[True, False], [False, True]])

    fig = ImageGridPlot(ncols=1).render([img])

    cbar_axes = [ax for ax in fig.axes if ax.get_label() == "<colorbar>"]
    assert len(cbar_axes) == 1
    assert [t.get_text() for t in cbar_axes[0].get_yticklabels()] == ["False", "True"]


def test_render_float_image_with_only_zero_one_values_uses_two_value_colorbar():
    # chunk_size=1 時 s_value 的 forward 數值恆為 0.0/1.0(見
    # example/notebooks/plot_full_epoch_animation.ipynb 的說明)——判斷依值域,
    # 不依 dtype,float 陣列只要值域是 {0,1} 也要走離散色階。
    img = np.array([[0.0, 1.0], [1.0, 0.0]])

    fig = ImageGridPlot(ncols=1).render([img])

    cbar_axes = [ax for ax in fig.axes if ax.get_label() == "<colorbar>"]
    assert [t.get_text() for t in cbar_axes[0].get_yticklabels()] == ["False", "True"]


def test_render_continuous_image_keeps_default_colorbar():
    img = np.array([[0.1, 0.5], [0.9, 1.0]])

    fig = ImageGridPlot(ncols=1).render([img])

    cbar_axes = [ax for ax in fig.axes if ax.get_label() == "<colorbar>"]
    assert [t.get_text() for t in cbar_axes[0].get_yticklabels()] != ["False", "True"]


TESTS = [
    test_unflatten_channels_matches_channel_major_order,
    test_unflatten_channels_wrong_length_raises,
    test_render_creates_one_axes_per_image_with_titles,
    test_render_without_titles_defaults_to_blank,
    test_render_mismatched_titles_length_raises,
    test_render_empty_images_raises,
    test_render_bool_image_uses_two_value_colorbar,
    test_render_float_image_with_only_zero_one_values_uses_two_value_colorbar,
    test_render_continuous_image_keeps_default_colorbar,
]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\n全部 {len(TESTS)} 項通過")
