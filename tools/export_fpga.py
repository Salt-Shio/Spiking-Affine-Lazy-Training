"""量化資料夾的 conv 層匯出成 SALT-FPGA RTL 用的 .mem 檔,每層三個,格式見 salt_core.fpga_export。

權重碼位元寬 b 取 model.npz 中繼資料的 spec.bits(example.quantize 寫的)。FC 層還沒有格式,略過。

用法(在 repo 根目錄):python -m tools.export_fpga <量化資料夾> --out-dir <輸出目錄>
輸出:<輸出目錄>/<層名>_weight.mem、<層名>_threshold.mem、<層名>_decay.mem
"""
import argparse
import sys
from pathlib import Path

from salt_core.fpga_export import write_conv_layer_mem
from salt_core.io import load_quantized
from salt_core.layers import ConvLayer

MODEL_FILENAME = "model.npz"


def export(quant_dir: Path, out_dir: Path) -> None:
    """quant_dir/model.npz 的每個 conv 層寫進 out_dir。中繼資料沒有 spec.bits 時 raise ValueError。"""
    model = load_quantized(quant_dir / MODEL_FILENAME)
    weight_width = model.metadata.get("spec", {}).get("bits")
    if weight_width is None:
        raise ValueError(f"{quant_dir / MODEL_FILENAME} 的中繼資料沒有 spec.bits")
    for layer, params in zip(model.network.layers, model.params):
        if not isinstance(layer, ConvLayer):
            print(f"{layer.name}:不是 conv 層,略過")
            continue
        paths = write_conv_layer_mem(out_dir, layer, params, weight_width)
        print(f"{layer.name}:" + "、".join(str(p) for p in paths.values()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("quant_dir", type=Path, help="量化資料夾(裡面有 model.npz)")
    parser.add_argument("--out-dir", type=Path, required=True, help="輸出目錄,不存在會建立")
    args = parser.parse_args()
    export(args.quant_dir, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
