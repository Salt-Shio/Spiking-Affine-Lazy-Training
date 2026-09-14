"""viz/channel_grid.py 的單元測試(合成陣列,不需要訓練 run)。"""
import numpy as np

import matplotlib
matplotlib.use("Agg")

from matplotlib.animation import FuncAnimation
from matplotlib.colors import to_rgba

from viz.channel_grid import ChannelGridAnimation, ImageGridPlot, unflatten_channels

# 跟 viz/channel_grid.py 的 _PAD_COLOR 對齊——這裡故意寫死字面值而不是 import
# 私有常數,測的是「呼叫端看得到的顏色」這個外部行為,不是內部實作細節。
_PAD_COLOR_RGBA = to_rgba("#ff00ff")


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


def test_render_discrete_true_uses_two_value_colorbar():
    img = np.array([[True, False], [False, True]])

    fig = ImageGridPlot(ncols=1).render([img], discrete=True)

    cbar_axes = [ax for ax in fig.axes if ax.get_label() == "<colorbar>"]
    assert len(cbar_axes) == 1
    assert [t.get_text() for t in cbar_axes[0].get_yticklabels()] == ["False", "True"]


def test_render_without_discrete_keeps_default_colorbar_even_for_bool():
    # discrete 沒講就是連續值,不看 dtype 猜——即使是 bool 陣列也一樣。
    img = np.array([[True, False], [False, True]])

    fig = ImageGridPlot(ncols=1).render([img])

    cbar_axes = [ax for ax in fig.axes if ax.get_label() == "<colorbar>"]
    assert [t.get_text() for t in cbar_axes[0].get_yticklabels()] != ["False", "True"]


def test_render_discrete_list_applies_per_image():
    images = [np.array([[True, False]]), np.array([[0.1, 0.9]])]

    fig = ImageGridPlot(ncols=2).render(images, discrete=[True, False])

    cbar_axes = [ax for ax in fig.axes if ax.get_label() == "<colorbar>"]
    assert len(cbar_axes) == 2
    labelled = [[t.get_text() for t in cb.get_yticklabels()] for cb in cbar_axes]
    assert ["False", "True"] in labelled
    assert any(l != ["False", "True"] for l in labelled)


def test_render_discrete_mismatched_length_raises():
    images = [np.zeros((2, 2)) for _ in range(2)]
    try:
        ImageGridPlot().render(images, discrete=[True])
    except ValueError:
        pass
    else:
        raise AssertionError("預期 discrete 長度不對要拋 ValueError")


def test_render_without_extent_strips_ticks():
    fig = ImageGridPlot(ncols=1).render([np.zeros((3, 3))])

    ax = fig.axes[0]
    assert len(ax.get_xticks()) == 0
    assert len(ax.get_yticks()) == 0


def test_render_with_extent_keeps_real_axis_coordinates():
    img = np.zeros((5, 21))  # 模擬「5 顆神經元 x 21 欄時間窗」

    fig = ImageGridPlot(ncols=1).render(
        [img], extents=(-10.0, 10.0, 104.0, 99.0), xlabel="ms offset", ylabel="neuron index")

    ax = fig.axes[0]
    assert len(ax.get_xticks()) > 0
    assert ax.get_xlabel() == "ms offset"
    assert ax.get_ylabel() == "neuron index"
    im = ax.images[0]
    assert im.get_extent() == [-10.0, 10.0, 104.0, 99.0]


def test_animation_extent_stays_fixed_across_frame_updates():
    frames = _animation_frames()

    anim = ChannelGridAnimation(ncols=2).build(
        frames, discrete=[False, True], extents=(0.0, 2.0, 1.0, 0.0))
    anim._func(2)

    for ax in _grid_axes(anim._fig):
        assert ax.images[0].get_extent() == [0.0, 2.0, 1.0, 0.0]


def test_render_nan_pixel_uses_pad_color_for_continuous_image():
    img = np.array([[0.1, np.nan], [0.9, 0.5]])

    fig = ImageGridPlot(ncols=1).render([img])

    im = fig.axes[0].images[0]
    assert im.cmap(np.nan) == _PAD_COLOR_RGBA


def test_render_nan_pixel_uses_pad_color_for_discrete_image():
    img = np.array([[1.0, np.nan], [0.0, 1.0]])

    fig = ImageGridPlot(ncols=1).render([img], discrete=True)

    im = fig.axes[0].images[0]
    assert im.cmap(np.nan) == _PAD_COLOR_RGBA


def _animation_frames():
    # image 0(連續值):每個 frame 內全部像素同一個值,frame 0/1/2 分別是
    # 0/1/2,方便驗證色階固定範圍跟 set_data 是否真的逐 frame 更新。
    # image 1(離散值):spike_mask 風格的 True/False 棋盤。
    frames = np.array([
        [[[0.0, 0.0], [0.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]],
        [[[1.0, 1.0], [1.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]],
        [[[2.0, 2.0], [2.0, 2.0]], [[1.0, 1.0], [0.0, 0.0]]],
    ])
    return frames


def test_animation_nan_region_uses_pad_color():
    # 模擬把小尺寸的層 padding 進大畫布:image 0 右下角補 NaN。
    frames = _animation_frames()
    frames[:, 0, 1, 1] = np.nan

    anim = ChannelGridAnimation(ncols=2).build(frames, discrete=[False, True])

    im = [ax.images[0] for ax in _grid_axes(anim._fig)][0]
    assert im.cmap(np.nan) == _PAD_COLOR_RGBA


def test_animation_build_returns_func_animation_with_expected_axes():
    anim = ChannelGridAnimation(ncols=2).build(_animation_frames(), discrete=[False, True])

    assert isinstance(anim, FuncAnimation)
    grid = _grid_axes(anim._fig)
    assert len([ax for ax in grid if ax.get_visible()]) == 2


def test_animation_continuous_color_scale_fixed_across_all_frames():
    frames = _animation_frames()

    anim = ChannelGridAnimation(ncols=2).build(frames, discrete=[False, True])

    continuous_im = [im for im in anim._fig.axes[0].images][0]
    assert continuous_im.get_clim() == (0.0, 2.0)


def test_animation_update_sets_image_data_for_requested_frame():
    frames = _animation_frames()

    anim = ChannelGridAnimation(ncols=2).build(frames, discrete=[False, True])
    anim._func(2)

    updated = [ax.images[0].get_array() for ax in _grid_axes(anim._fig)]
    assert np.array_equal(updated[0], frames[2, 0])
    assert np.array_equal(updated[1], frames[2, 1])


def test_animation_discrete_uses_two_value_colorbar():
    anim = ChannelGridAnimation(ncols=2).build(_animation_frames(), discrete=[False, True])

    cbar_axes = [ax for ax in anim._fig.axes if ax.get_label() == "<colorbar>"]
    labelled = [[t.get_text() for t in cb.get_yticklabels()] for cb in cbar_axes]
    assert ["False", "True"] in labelled


def test_animation_frame_labels_update_suptitle():
    frames = _animation_frames()
    labels = ["t=0ms", "t=1ms", "t=2ms"]

    anim = ChannelGridAnimation(ncols=2).build(frames, discrete=[False, True], frame_labels=labels)
    assert anim._fig._suptitle.get_text() == "t=0ms"
    anim._func(1)
    assert anim._fig._suptitle.get_text() == "t=1ms"


def test_animation_wrong_ndim_raises():
    try:
        ChannelGridAnimation().build(np.zeros((3, 2, 2)))
    except ValueError:
        pass
    else:
        raise AssertionError("預期 frames 不是 4 維要拋 ValueError")


def test_animation_empty_frames_raises():
    try:
        ChannelGridAnimation().build(np.zeros((0, 2, 3, 3)))
    except ValueError:
        pass
    else:
        raise AssertionError("預期 n_frames=0 要拋 ValueError")


def test_animation_mismatched_frame_labels_length_raises():
    frames = _animation_frames()
    try:
        ChannelGridAnimation().build(frames, frame_labels=["only-one"])
    except ValueError:
        pass
    else:
        raise AssertionError("預期 frame_labels 長度不對要拋 ValueError")


TESTS = [
    test_unflatten_channels_matches_channel_major_order,
    test_unflatten_channels_wrong_length_raises,
    test_render_creates_one_axes_per_image_with_titles,
    test_render_without_titles_defaults_to_blank,
    test_render_mismatched_titles_length_raises,
    test_render_empty_images_raises,
    test_render_discrete_true_uses_two_value_colorbar,
    test_render_without_discrete_keeps_default_colorbar_even_for_bool,
    test_render_discrete_list_applies_per_image,
    test_render_discrete_mismatched_length_raises,
    test_render_without_extent_strips_ticks,
    test_render_with_extent_keeps_real_axis_coordinates,
    test_animation_extent_stays_fixed_across_frame_updates,
    test_render_nan_pixel_uses_pad_color_for_continuous_image,
    test_render_nan_pixel_uses_pad_color_for_discrete_image,
    test_animation_nan_region_uses_pad_color,
    test_animation_build_returns_func_animation_with_expected_axes,
    test_animation_continuous_color_scale_fixed_across_all_frames,
    test_animation_update_sets_image_data_for_requested_frame,
    test_animation_discrete_uses_two_value_colorbar,
    test_animation_frame_labels_update_suptitle,
    test_animation_wrong_ndim_raises,
    test_animation_empty_frames_raises,
    test_animation_mismatched_frame_labels_length_raises,
]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\n全部 {len(TESTS)} 項通過")
