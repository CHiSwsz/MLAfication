"""FP8 cache support for Transformers versions without a native FP8 backend."""

from transformers.cache_utils import Cache, QuantizedLayer


class QuantoFP8QuantizedLayer(QuantizedLayer):
    """Quantized cache layer backed by Quanto FP8 E4M3 tensors."""

    def __init__(
        self,
        axis_key: int = 0,
        axis_value: int = 0,
        residual_length: int = 128,
    ):
        # q_group_size is unused for 8-bit qtypes, but QuantizedLayer expects it.
        super().__init__(
            nbits=8,
            axis_key=axis_key,
            axis_value=axis_value,
            q_group_size=64,
            residual_length=residual_length,
        )

        try:
            from optimum.quanto import AbsmaxOptimizer, qfloat8_e4m3fn
        except ImportError as exc:
            raise ImportError(
                "FP8 cache requires optimum-quanto with qfloat8_e4m3fn support."
            ) from exc

        if self.axis_key not in (0, -1):
            raise ValueError(
                f"`axis_key` for the FP8 Quanto cache must be 0 or -1, got {self.axis_key}."
            )
        if self.axis_value not in (0, -1):
            raise ValueError(
                f"`axis_value` for the FP8 Quanto cache must be 0 or -1, got {self.axis_value}."
            )

        self.qtype = qfloat8_e4m3fn
        self.optimizer = AbsmaxOptimizer()

    def _quantize(self, tensor, axis):
        from optimum.quanto import quantize_weight

        # Quanto forbids both an affine shift and group_size for 8-bit qtypes.
        scale = self.optimizer(tensor, self.qtype, axis)
        return quantize_weight(
            tensor,
            self.qtype,
            axis,
            scale,
            shift=None,
            group_size=None,
        )

    def _dequantize(self, qtensor):
        return qtensor.dequantize()


class QuantoFP8Cache(Cache):
    """A residual cache whose compressed portion uses FP8 E4M3."""

    def __init__(
        self,
        config,
        axis_key: int = 0,
        axis_value: int = 0,
        residual_length: int = 128,
    ):
        text_config = config.get_text_config(decoder=True)
        layers = [
            QuantoFP8QuantizedLayer(
                axis_key=axis_key,
                axis_value=axis_value,
                residual_length=residual_length,
            )
            for _ in range(text_config.num_hidden_layers)
        ]
        super().__init__(layers=layers)
