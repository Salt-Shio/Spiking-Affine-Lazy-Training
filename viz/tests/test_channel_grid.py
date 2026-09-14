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
    panels = [_FakePanel(np.zeros((3, 2, 2)), extent=(0.0, 2.0, 1.0, 0.0)),
              _FakePanel(np.zeros((3, 2, 2)), extent=(0.0, 2.0, 1.0, 0.0))]

    anim = ChannelGridAnimation(ncols=2).build(panels)
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


class _FakePanel:
    """`AnimatedPanel` 介面的最小測試替身——不代表 conv 或 FC 的任何真實語意,
    純粹用來驗證 `ChannelGridAnimation` 這個共用機制本身的行為。"""

    def __init__(self, frames, discrete: bool = False, title: str | None = None,
                 extent: tuple | None = None, xlabel: str | None = None,
                 ylabel: str | None = None, position: tuple | None = None,
                 size_ratio: tuple | None = None):
        self._frames = np.asarray(frames)  # (n_frames, H, W)
        self.discrete = discrete
        self.title = title
        self.extent = extent
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.position = position
        self.size_ratio = size_ratio
        self.n_frames = self._frames.shape[0]
        self.value_range = None if discrete else (
            float(np.nanmin(self._frames)), float(np.nanmax(self._frames)))

    def frame(self, t: int) -> np.ndarray:
        return self._frames[t]


def _two_fake_panels():
    # panel 0(連續值):每個 frame 內全部像素同一個值,frame 0/1/2 分別是
    # 0/1/2,方便驗證色階固定範圍跟 set_data 是否真的逐 frame 更新。
    # panel 1(離散值):spike_mask 風格的 True/False 棋盤。
    continuous = np.array([[[0.0, 0.0], [0.0, 0.0]],
                            [[1.0, 1.0], [1.0, 1.0]],
                            [[2.0, 2.0], [2.0, 2.0]]])
    discrete = np.array([[[1.0, 0.0], [0.0, 1.0]],
                          [[0.0, 1.0], [1.0, 0.0]],
                          [[1.0, 1.0], [0.0, 0.0]]])
    return [_FakePanel(continuous, discrete=False), _FakePanel(discrete, discrete=True)]


def test_animation_nan_region_uses_pad_color():
    # 模擬 panel 自己的圖裡有一塊沒有真實資料(例如 FC 窗口邊界):右下角補 NaN。
    panels = _two_fake_panels()
    panels[0]._frames[:, 1, 1] = np.nan

    anim = ChannelGridAnimation(ncols=2).build(panels)

    im = [ax.images[0] for ax in _grid_axes(anim._fig)][0]
    assert im.cmap(np.nan) == _PAD_COLOR_RGBA


def test_animation_build_returns_func_animation_with_expected_axes():
    anim = ChannelGridAnimation(ncols=2).build(_two_fake_panels())

    assert isinstance(anim, FuncAnimation)
    grid = _grid_axes(anim._fig)
    assert len([ax for ax in grid if ax.get_visible()]) == 2


def test_animation_continuous_color_scale_fixed_across_all_frames():
    anim = ChannelGridAnimation(ncols=2).build(_two_fake_panels())

    continuous_im = anim._fig.axes[0].images[0]
    assert continuous_im.get_clim() == (0.0, 2.0)


def test_animation_update_sets_image_data_for_requested_frame():
    panels = _two_fake_panels()

    anim = ChannelGridAnimation(ncols=2).build(panels)
    anim._func(2)

    updated = [ax.images[0].get_array() for ax in _grid_axes(anim._fig)]
    assert np.array_equal(updated[0], panels[0].frame(2))
    assert np.array_equal(updated[1], panels[1].frame(2))


def test_animation_discrete_uses_two_value_colorbar():
    anim = ChannelGridAnimation(ncols=2).build(_two_fake_panels())

    cbar_axes = [ax for ax in anim._fig.axes if ax.get_label() == "<colorbar>"]
    labelled = [[t.get_text() for t in cb.get_yticklabels()] for cb in cbar_axes]
    assert ["False", "True"] in labelled


def test_animation_frame_labels_update_suptitle():
    labels = ["t=0ms", "t=1ms", "t=2ms"]

    anim = ChannelGridAnimation(ncols=2).build(_two_fake_panels(), frame_labels=labels)
    assert anim._fig._suptitle.get_text() == "t=0ms"
    anim._func(1)
    assert anim._fig._suptitle.get_text() == "t=1ms"


def test_animation_different_shaped_panels_keep_own_box_aspect():
    # conv 接近正方形、FC 窗口又寬又扁——混在同一組動畫時,每個 panel 的
    # imshow 應該維持自己真正的長寬比例,不是被同一格的框拉伸成一樣的形狀。
    panels = [_FakePanel(np.zeros((3, 10, 10))), _FakePanel(np.zeros((3, 5, 20)))]

    anim = ChannelGridAnimation(ncols=2).build(panels)

    aspects = [ax.get_box_aspect() for ax in _grid_axes(anim._fig)]
    assert aspects[0] == 10 / 10
    assert aspects[1] == 5 / 20


def test_animation_manual_size_ratio_also_overrides_box_aspect():
    # 明講 size_ratio 之後,連 imshow 的框形狀都要照 size_ratio 走(拉寬/壓扁
    # 是呼叫端自己選的),不能被資料的真實長寬比(10/21)卡住。
    panel = _FakePanel(np.zeros((3, 10, 21)), size_ratio=(3.0, 0.5))

    anim = ChannelGridAnimation(ncols=1).build([panel])

    ax = _grid_axes(anim._fig)[0]
    assert ax.get_box_aspect() == 0.5 / 3.0


def test_animation_manual_position_overrides_auto_placement():
    # 3 個 panel、ncols=2:自動排版本來會是 (0,0)(0,1)(1,0)。把最後一個明講
    # 放到 (0,1),前面自動排的那個(panel 1)應該讓開、被擠到 (1,0)。
    panels = [_FakePanel(np.zeros((3, 4, 4))),
              _FakePanel(np.zeros((3, 4, 4))),
              _FakePanel(np.zeros((3, 4, 4)), position=(0, 1))]

    anim = ChannelGridAnimation(ncols=2).build(panels)

    axes = _grid_axes(anim._fig)
    cells = [(list(ax.get_subplotspec().rowspan)[0], list(ax.get_subplotspec().colspan)[0])
             for ax in axes]
    assert cells[2] == (0, 1)  # 明講的那個真的在 (0,1)
    assert cells[1] == (1, 0)  # 自動排的第 2 個被擠到剩下的格子,不是原本的 (0,1)


def test_animation_manual_size_ratio_overrides_auto_aspect():
    # 不設 size_ratio 時,方形圖(aspect=1)的比重是 (1,1);明講 size_ratio
    # 之後,不管圖片形狀是什麼,比重直接照明講的值。
    panels = [_FakePanel(np.zeros((3, 4, 4)), size_ratio=(2.0, 0.5))]

    anim = ChannelGridAnimation(ncols=1).build(panels)

    ax = _grid_axes(anim._fig)[0]
    assert ax.get_gridspec().get_width_ratios() == [2.0]
    assert ax.get_gridspec().get_height_ratios() == [0.5]


def test_animation_stacked_fc_panels_get_position_and_size_override():
    # 對應真實情境:2 個 conv 面板維持自動排版(row 0),2 個 FC 面板改成
    # 上下並排(同一欄、兩列)、寬度拉長高度縮小。
    conv_a = _FakePanel(np.zeros((3, 10, 10)))
    conv_b = _FakePanel(np.zeros((3, 10, 10)))
    fc_a = _FakePanel(np.zeros((3, 10, 21)), position=(1, 0), size_ratio=(3.0, 0.6))
    fc_b = _FakePanel(np.zeros((3, 10, 21)), position=(2, 0), size_ratio=(3.0, 0.6))

    anim = ChannelGridAnimation(ncols=2).build([conv_a, conv_b, fc_a, fc_b])

    axes = _grid_axes(anim._fig)
    cells = [(list(ax.get_subplotspec().rowspan)[0], list(ax.get_subplotspec().colspan)[0])
             for ax in axes]
    assert cells[0] == (0, 0) and cells[1] == (0, 1)  # conv 兩個維持自動排版
    assert cells[2] == (1, 0) and cells[3] == (2, 0)  # FC 兩個上下並排在同一欄
    gs = axes[0].get_gridspec()
    assert gs.get_width_ratios()[0] == 3.0   # FC 那欄的寬度採用明講的倍率
    assert gs.get_height_ratios()[1:] == [0.6, 0.6]  # FC 那兩列的高度採用明講的倍率


def test_animation_duplicate_position_raises():
    panels = [_FakePanel(np.zeros((3, 4, 4)), position=(0, 0)),
              _FakePanel(np.zeros((3, 4, 4)), position=(0, 0))]
    try:
        ChannelGridAnimation(ncols=2).build(panels)
    except ValueError:
        pass
    else:
        raise AssertionError("預期兩個 panel 搶同一個 position 要拋 ValueError")


def test_animation_spanning_panel_does_not_inflate_spanned_columns():
    # 對應「conv 兩個緊靠、FC 橫跨兩欄上下並排」的真實情境:FC 橫跨欄 0~1
    # 時,不該逼任一欄單獨變寬——欄寬維持 conv 自己算出來的 1.0,不會因為
    # FC 的 size_ratio 寬倍率(2.0)被拉大。
    conv_a = _FakePanel(np.zeros((3, 10, 10)))
    conv_b = _FakePanel(np.zeros((3, 10, 10)))
    fc = _FakePanel(np.zeros((3, 10, 21)), position=(1, slice(0, 2)), size_ratio=(2.0, 0.5))

    anim = ChannelGridAnimation(ncols=2).build([conv_a, conv_b, fc])

    gs = _grid_axes(anim._fig)[0].get_gridspec()
    assert gs.get_width_ratios() == [1.0, 1.0]   # 沒有被 FC 的寬倍率拉大
    assert gs.get_height_ratios() == [1.0, 0.5]  # FC 自己那一列還是採用明講的高倍率


def test_animation_spanning_panel_placed_across_requested_columns():
    conv_a = _FakePanel(np.zeros((3, 10, 10)))
    conv_b = _FakePanel(np.zeros((3, 10, 10)))
    fc = _FakePanel(np.zeros((3, 10, 21)), position=(1, slice(0, 2)))

    anim = ChannelGridAnimation(ncols=2).build([conv_a, conv_b, fc])

    fc_ax = _grid_axes(anim._fig)[2]
    spec = fc_ax.get_subplotspec()
    assert list(spec.rowspan) == [1]
    assert list(spec.colspan) == [0, 1]


def test_animation_spanning_position_conflict_raises():
    panels = [_FakePanel(np.zeros((3, 4, 4)), position=(0, 0)),
              _FakePanel(np.zeros((3, 4, 4)), position=(0, slice(0, 2)))]
    try:
        ChannelGridAnimation(ncols=2).build(panels)
    except ValueError:
        pass
    else:
        raise AssertionError("預期橫跨範圍跟另一個 panel 的格子重疊要拋 ValueError")


def test_animation_empty_panels_raises():
    try:
        ChannelGridAnimation().build([])
    except ValueError:
        pass
    else:
        raise AssertionError("預期 panels 是空的要拋 ValueError")


def test_animation_mismatched_n_frames_raises():
    panels = [_FakePanel(np.zeros((3, 2, 2))), _FakePanel(np.zeros((4, 2, 2)))]
    try:
        ChannelGridAnimation().build(panels)
    except ValueError:
        pass
    else:
        raise AssertionError("預期 panel 之間 n_frames 不一致要拋 ValueError")


def test_animation_mismatched_frame_labels_length_raises():
    try:
        ChannelGridAnimation().build(_two_fake_panels(), frame_labels=["only-one"])
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
    test_animation_different_shaped_panels_keep_own_box_aspect,
    test_animation_manual_size_ratio_also_overrides_box_aspect,
    test_animation_manual_position_overrides_auto_placement,
    test_animation_manual_size_ratio_overrides_auto_aspect,
    test_animation_stacked_fc_panels_get_position_and_size_override,
    test_animation_duplicate_position_raises,
    test_animation_spanning_panel_does_not_inflate_spanned_columns,
    test_animation_spanning_panel_placed_across_requested_columns,
    test_animation_spanning_position_conflict_raises,
    test_animation_empty_panels_raises,
    test_animation_mismatched_n_frames_raises,
    test_animation_mismatched_frame_labels_length_raises,
]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\n全部 {len(TESTS)} 項通過")
