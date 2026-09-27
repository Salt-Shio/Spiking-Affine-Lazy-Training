"""viz/channel_grid.py 的單元測試(合成陣列,不需要訓練 run)。"""
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import to_rgb, to_rgba

from viz.channel_grid import ChannelGridAnimation, ColorOverlayPanel, ImageGridPlot, unflatten_channels

# 跟 viz/channel_grid.py 的 _PAD_COLOR/_DISCRETE_OFF_COLOR 對齊——這裡故意
# 寫死字面值而不是 import 私有常數,測的是「呼叫端看得到的顏色」這個外部
# 行為,不是內部實作細節。
_PAD_COLOR_RGBA = to_rgba("#ff00ff")
_DISCRETE_OFF_RGB = to_rgb("#d9d9d9")


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
                 ylabel: str | None = None):
        self._frames = np.asarray(frames)  # (n_frames, H, W)
        self.discrete = discrete
        self.title = title
        self.extent = extent
        self.xlabel = xlabel
        self.ylabel = ylabel
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


def _build_two_panel_row():
    panels = _two_fake_panels()
    anim = ChannelGridAnimation().add_row(*panels).build()
    return anim, panels


def test_animation_extent_stays_fixed_across_frame_updates():
    panels = [_FakePanel(np.zeros((3, 2, 2)), extent=(0.0, 2.0, 1.0, 0.0)),
              _FakePanel(np.zeros((3, 2, 2)), extent=(0.0, 2.0, 1.0, 0.0))]

    anim = ChannelGridAnimation().add_row(*panels).build()
    anim._func(2)

    for ax in _grid_axes(anim._fig):
        assert ax.images[0].get_extent() == [0.0, 2.0, 1.0, 0.0]


def test_animation_nan_region_uses_pad_color():
    # 模擬 panel 自己的圖裡有一塊沒有真實資料(例如 FC 窗口邊界):右下角補 NaN。
    panels = _two_fake_panels()
    panels[0]._frames[:, 1, 1] = np.nan

    anim = ChannelGridAnimation().add_row(*panels).build()

    im = [ax.images[0] for ax in _grid_axes(anim._fig)][0]
    assert im.cmap(np.nan) == _PAD_COLOR_RGBA


def test_animation_build_returns_func_animation_with_expected_axes():
    anim, _ = _build_two_panel_row()

    assert isinstance(anim, FuncAnimation)
    grid = _grid_axes(anim._fig)
    assert len([ax for ax in grid if ax.get_visible()]) == 2


def test_animation_continuous_color_scale_fixed_across_all_frames():
    anim, _ = _build_two_panel_row()

    continuous_im = anim._fig.axes[0].images[0]
    assert continuous_im.get_clim() == (0.0, 2.0)


def test_animation_update_sets_image_data_for_requested_frame():
    anim, panels = _build_two_panel_row()
    anim._func(2)

    updated = [ax.images[0].get_array() for ax in _grid_axes(anim._fig)]
    assert np.array_equal(updated[0], panels[0].frame(2))
    assert np.array_equal(updated[1], panels[1].frame(2))


def test_animation_discrete_uses_two_value_colorbar():
    anim, _ = _build_two_panel_row()

    cbar_axes = [ax for ax in anim._fig.axes if ax.get_label() == "<colorbar>"]
    labelled = [[t.get_text() for t in cb.get_yticklabels()] for cb in cbar_axes]
    assert ["False", "True"] in labelled


def test_animation_frame_labels_update_suptitle():
    labels = ["t=0ms", "t=1ms", "t=2ms"]

    anim = ChannelGridAnimation().add_row(*_two_fake_panels()).build(frame_labels=labels)
    assert anim._fig._suptitle.get_text() == "t=0ms"
    anim._func(1)
    assert anim._fig._suptitle.get_text() == "t=1ms"


def test_animation_panels_in_same_row_keep_own_box_aspect():
    # conv 接近正方形、FC 窗口又寬又扁——同一排混在一起時,每個 panel 的
    # imshow 應該維持自己真正的長寬比例,不是被同一排硬拉伸成一樣的形狀。
    panels = [_FakePanel(np.zeros((3, 10, 10))), _FakePanel(np.zeros((3, 5, 20)))]

    anim = ChannelGridAnimation().add_row(*panels).build()

    aspects = [ax.get_box_aspect() for ax in _grid_axes(anim._fig)]
    assert aspects[0] == 10 / 10
    assert aspects[1] == 5 / 20


def test_animation_panel_with_extent_skips_box_aspect_constraint():
    # 給了 extent 的 panel(座標軸已經決定顯示比例)不套 box_aspect,不會被
    # 圖片本身的像素形狀(10/21)卡住。
    panel = _FakePanel(np.zeros((3, 10, 21)), extent=(-10.0, 10.0, 10.0, 0.0))

    anim = ChannelGridAnimation().add_row(panel).build()

    ax = _grid_axes(anim._fig)[0]
    assert ax.get_box_aspect() is None


def test_animation_rows_are_independent_of_each_other():
    # 對應真實情境:第一排放 2 個 conv(自動等寬),第二、三排各自放 1 個 FC
    # (各自撐滿整排的寬度)。改第二、三排放幾個,不該牽動第一排 conv 的形狀。
    conv_a = _FakePanel(np.zeros((3, 17, 17)))
    conv_b = _FakePanel(np.zeros((3, 17, 17)))
    fc_a = _FakePanel(np.zeros((3, 10, 21)))
    fc_b = _FakePanel(np.zeros((3, 10, 21)))

    anim = (ChannelGridAnimation()
            .add_row(conv_a, conv_b)
            .add_row(fc_a)
            .add_row(fc_b)
            .build())

    axes = _grid_axes(anim._fig)
    assert len(axes) == 4
    # conv 兩個維持自己的正方形比例,不受後面兩排的存在影響。
    assert axes[0].get_box_aspect() == 1.0
    assert axes[1].get_box_aspect() == 1.0
    # FC 兩排各自只有一個 panel,寬度撐滿自己那一排(不會被 conv 那排的兩欄
    # 佔的欄寬套住)。
    _, _, conv_width, _ = axes[0].get_position().bounds
    _, _, fc_width, _ = axes[2].get_position().bounds
    assert fc_width > conv_width


def test_animation_row_height_weight_scales_row_relative_to_others():
    # 兩排各一個 panel,第二排 height=0.5:第二排的實際物理高度應該接近第一
    # 排的一半。用 window_extent(畫布的絕對像素座標)量,不是用
    # get_position()——後者對「axes 在 subfigure 裡面」回傳的是相對那個
    # subfigure 自己的座標,不是相對整張畫布,兩排會量出一樣的比例,測不出
    # height 這個參數有沒有真的生效。
    tall = _FakePanel(np.zeros((3, 4, 4)))
    short = _FakePanel(np.zeros((3, 4, 4)))

    anim = ChannelGridAnimation().add_row(tall).add_row(short, height=0.5).build()

    anim._fig.canvas.draw()
    renderer = anim._fig.canvas.get_renderer()
    axes = _grid_axes(anim._fig)
    tall_height = axes[0].get_window_extent(renderer).height
    short_height = axes[1].get_window_extent(renderer).height
    ratio = short_height / tall_height
    assert 0.4 < ratio < 0.6


def test_animation_widths_scale_columns_within_same_row():
    # 同一排兩個 panel,widths=[2, 1]:第一個的實際物理寬度應該是第二個的
    # 兩倍。兩個都給 extent(不套 box_aspect),避免長寬比例限制干擾寬度本身
    # 的比重量測。
    a = _FakePanel(np.zeros((3, 4, 4)), extent=(0.0, 4.0, 4.0, 0.0))
    b = _FakePanel(np.zeros((3, 4, 4)), extent=(0.0, 4.0, 4.0, 0.0))

    anim = ChannelGridAnimation().add_row(a, b, widths=[2, 1]).build()

    anim._fig.canvas.draw()
    renderer = anim._fig.canvas.get_renderer()
    axes = _grid_axes(anim._fig)
    width_a = axes[0].get_window_extent(renderer).width
    width_b = axes[1].get_window_extent(renderer).width
    assert abs(width_a / width_b - 2.0) < 1e-6


def test_add_row_mismatched_widths_length_raises():
    try:
        ChannelGridAnimation().add_row(_FakePanel(np.zeros((3, 2, 2))), widths=[1, 2])
    except ValueError:
        pass
    else:
        raise AssertionError("預期 widths 長度跟 panel 數量不一致要拋 ValueError")


def test_color_overlay_panel_combines_colors_at_each_pixel():
    # 2x2 圖,panel_a 只有 (0,0) 是 1、panel_b 只有 (1,1) 是 1。紅/藍疊起來,
    # 兩個像素應該各自變成純紅/純藍;完全沒有 channel 亮的像素該是底色
    # (跟離散圖 False 用的同一個灰色),不是加法算出來的黑色。
    panel_a = _FakePanel(np.array([[[1.0, 0.0], [0.0, 0.0]]]))
    panel_b = _FakePanel(np.array([[[0.0, 0.0], [0.0, 1.0]]]))

    overlay = ColorOverlayPanel([panel_a, panel_b], colors=[(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)])
    frame = overlay.frame(0)

    assert frame.shape == (2, 2, 3)
    assert np.array_equal(frame[0, 0], [1.0, 0.0, 0.0])
    assert np.array_equal(frame[1, 1], [0.0, 0.0, 1.0])
    assert np.allclose(frame[0, 1], _DISCRETE_OFF_RGB)


def test_color_overlay_panel_custom_background_overrides_default():
    panel_a = _FakePanel(np.array([[[0.0]]]))

    overlay = ColorOverlayPanel([panel_a], colors=[(1.0, 0.0, 0.0)], background=(1.0, 1.0, 1.0))

    assert np.array_equal(overlay.frame(0)[0, 0], [1.0, 1.0, 1.0])


def test_color_overlay_panel_overlapping_pixels_clip_to_one():
    # 同一個像素兩個 panel 都是 1,紅+紅疊起來不能超過 1.0。
    panel_a = _FakePanel(np.array([[[1.0]]]))
    panel_b = _FakePanel(np.array([[[1.0]]]))

    overlay = ColorOverlayPanel([panel_a, panel_b], colors=[(1.0, 0.0, 0.0), (1.0, 0.0, 0.0)])

    assert np.array_equal(overlay.frame(0)[0, 0], [1.0, 0.0, 0.0])


def test_color_overlay_panel_mismatched_panels_and_colors_length_raises():
    try:
        ColorOverlayPanel([_FakePanel(np.zeros((1, 2, 2)))], colors=[(1, 0, 0), (0, 0, 1)])
    except ValueError:
        pass
    else:
        raise AssertionError("預期 panels/colors 數量不一致要拋 ValueError")


def test_color_overlay_panel_empty_panels_raises():
    try:
        ColorOverlayPanel([], colors=[])
    except ValueError:
        pass
    else:
        raise AssertionError("預期 panels 是空的要拋 ValueError")


def test_color_overlay_panel_mismatched_n_frames_raises():
    panel_a = _FakePanel(np.zeros((3, 2, 2)))
    panel_b = _FakePanel(np.zeros((4, 2, 2)))
    try:
        ColorOverlayPanel([panel_a, panel_b], colors=[(1, 0, 0), (0, 0, 1)])
    except ValueError:
        pass
    else:
        raise AssertionError("預期 panel 之間 n_frames 不一致要拋 ValueError")


def test_color_overlay_panel_mismatched_frame_shapes_raises():
    panel_a = _FakePanel(np.zeros((1, 2, 2)))
    panel_b = _FakePanel(np.zeros((1, 3, 3)))
    overlay = ColorOverlayPanel([panel_a, panel_b], colors=[(1, 0, 0), (0, 0, 1)])
    try:
        overlay.frame(0)
    except ValueError:
        pass
    else:
        raise AssertionError("預期被疊的 panel 圖形狀不一致要拋 ValueError")


def test_animation_rgb_panel_renders_without_colorbar():
    panel_a = _FakePanel(np.zeros((2, 3, 3)))
    panel_b = _FakePanel(np.zeros((2, 3, 3)))
    overlay = ColorOverlayPanel([panel_a, panel_b], colors=[(1, 0, 0), (0, 0, 1)])

    anim = ChannelGridAnimation().add_row(overlay).build()

    cbar_axes = [ax for ax in anim._fig.axes if ax.get_label() == "<colorbar>"]
    assert len(cbar_axes) == 0
    ax = _grid_axes(anim._fig)[0]
    assert ax.get_box_aspect() == 3 / 3
    assert ax.images[0].get_array().shape == (3, 3, 3)


def test_add_row_without_panels_raises():
    try:
        ChannelGridAnimation().add_row()
    except ValueError:
        pass
    else:
        raise AssertionError("預期 add_row() 沒給任何 panel 要拋 ValueError")


def test_animation_no_rows_raises():
    try:
        ChannelGridAnimation().build()
    except ValueError:
        pass
    else:
        raise AssertionError("預期沒有任何一排 panel 就 build() 要拋 ValueError")


def test_animation_mismatched_n_frames_raises():
    panels = [_FakePanel(np.zeros((3, 2, 2))), _FakePanel(np.zeros((4, 2, 2)))]
    try:
        ChannelGridAnimation().add_row(*panels).build()
    except ValueError:
        pass
    else:
        raise AssertionError("預期 panel 之間 n_frames 不一致要拋 ValueError")


def test_animation_mismatched_frame_labels_length_raises():
    try:
        ChannelGridAnimation().add_row(*_two_fake_panels()).build(frame_labels=["only-one"])
    except ValueError:
        pass
    else:
        raise AssertionError("預期 frame_labels 長度不對要拋 ValueError")
