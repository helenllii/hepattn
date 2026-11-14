import math
from functools import partial
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn.attention.flex_attention import create_block_mask, create_mask

from hepattn.flex import relative_position, relative_position_wrapped
from hepattn.flex.sliding_window import sliding_window_mask, sliding_window_mask_wrapped
from hepattn.models.attention import Attention, repad_from_flash_varlen, unpad_for_flash_varlen
from hepattn.models.dense import Dense
from hepattn.models.norm import NORM_TYPES, get_hybrid_norm_config

# attn_type: patch selects PatchEncoderLayer. Within a patch it runs this Attention backend: the patched
# constituents are masked per patch, which flash-varlen's unpadded layout cannot express.
PATCH_ATTN_TYPE = "patch"
PATCH_INNER_ATTN_TYPE = "torch"

create_block_mask = torch.compile(create_block_mask, dynamic=True)  # ty: ignore[invalid-assignment]

SCORE_MODS = {
    "relative_position": relative_position,
    "relative_position_wrapped": relative_position_wrapped,
}


def sort_index(value: Tensor, valid: Tensor | None) -> Tensor:
    """Argsort of value with the padding last, so the real tokens stay contiguous in sorted order."""
    if valid is not None:
        value = value.masked_fill(~valid, float("inf"))
    return torch.argsort(value, dim=-1)


@torch.no_grad()
def periodic_sort_value(value: Tensor, valid: Tensor | None, period: float) -> Tensor:
    """Map a periodic value (e.g. phi, period 2pi) to [0, period), measured from the end of the event's largest
    empty gap, so its linear order goes once round the circle and cuts it only where there are no tokens.
    Sorting the raw value instead cuts the circle at the wrap point (phi = +-pi), whatever lies on either side.

    Args:
        value: (B, S) periodic values; padded entries may hold anything, including NaN.
        valid: Optional (B, S) mask, True for real tokens.
        period: The period of value.

    Returns:
        (B, S) values in [0, period) for the real tokens, for sort_index; padded entries are left to its masking.
    """
    if valid is None:
        valid = torch.ones_like(value, dtype=torch.bool)
    num_valid = valid.sum(-1, keepdim=True)
    s = torch.sort(torch.remainder(value, period).masked_fill(~valid, float("inf")), dim=-1).values
    # Gap k ends at s[k]: k = 0 is the wrap-around gap from the last real token, k >= 1 the gap from s[k - 1]
    last = s.gather(-1, (num_valid - 1).clamp_min(0))
    gaps = torch.cat([s[..., :1] + period - last, s[..., 1:] - s[..., :-1]], dim=-1)
    gaps = gaps.masked_fill(torch.arange(s.shape[-1], device=s.device) >= num_valid, float("-inf"))
    start = s.gather(-1, gaps.argmax(-1, keepdim=True))
    # An event without real tokens has start = inf; any finite start will do
    start = torch.where(start.isfinite(), start, 0.0)
    return torch.remainder(value - start, period)


def gmp_neighbourhood(
    eta: Tensor, phi: Tensor, valid: Tensor, kernel_size: int, eta_cell: float, phi_cells: int, eta_max: float
) -> tuple[Tensor, Tensor]:
    """The cell offsets between the tokens of each event on a fixed (eta, phi) grid, for GeometricMessagePassing.

    Cells: u = floor((eta + eta_max) / eta_cell), not periodic; v = floor(phi / (2pi / phi_cells)) mod phi_cells,
    periodic, so tokens either side of phi = +-pi are neighbours. Only the real tokens with finite coordinates and
    |eta| <= eta_max are on the grid: the rest neither send nor receive messages (they are not clamped onto the edge).

    Args:
        eta: (B, N) pseudorapidity; padded entries may hold anything, including NaN.
        phi: (B, N) azimuth in radians, any branch.
        valid: (B, N), True for real tokens.
        kernel_size: Odd side k of the square kernel, in cells.
        eta_cell: Cell size in eta.
        phi_cells: Number of cells round the circle in phi.
        eta_max: Grid range in eta, [-eta_max, eta_max].

    Returns:
        neighbours: (B, k * k * N, N) bool; row o * N + i, column j is True when j is i's neighbour at kernel offset
            o = (du + k // 2) * k + (dv + k // 2), du = u_j - u_i, dv = v_j - v_i (wrapped). A token is its own du = dv = 0
            neighbour, and tokens sharing a cell are each other's.
        in_grid: (B, N) bool, the tokens on the grid.
    """
    r = kernel_size // 2
    in_grid = valid & eta.isfinite() & phi.isfinite() & (eta.abs() <= eta_max)
    eta = torch.where(in_grid, eta, 0.0)
    phi = torch.where(in_grid, phi, 0.0)
    u = torch.floor((eta + eta_max) / eta_cell).long()
    v = torch.remainder(torch.floor(phi * (phi_cells / (2 * math.pi))), phi_cells).long()
    du = u.unsqueeze(-2) - u.unsqueeze(-1)  # (B, N_i, N_j)
    dv = torch.remainder(v.unsqueeze(-2) - v.unsqueeze(-1) + r, phi_cells) - r
    pairs = in_grid.unsqueeze(-1) & in_grid.unsqueeze(-2)
    offsets = [pairs & (du == a) & (dv == b) for a in range(-r, r + 1) for b in range(-r, r + 1)]
    return torch.cat(offsets, dim=-2), in_grid


def gmp_neighbourhood_3d(
    eta: Tensor,
    phi: Tensor,
    depth: Tensor,
    valid: Tensor,
    kernel_size: int,
    eta_cell: float,
    phi_cells: int,
    eta_max: float,
    depth_cell: float,
    depth_kernel_size: int | None = None,
    depth_wildcard: bool = False,
) -> tuple[Tensor, Tensor]:
    """gmp_neighbourhood with a third, non-periodic depth axis: the cell offsets on an (eta, phi, depth) grid.

    Cells: u and v as in gmp_neighbourhood; w = floor(max(depth, 0) / depth_cell), unbounded above (the pairwise form
    needs only cell differences, so there is no depth range and no overflow bin; a small negative depth, e.g. a cluster
    barycentre a few mm in front of the calorimeter face, joins the first bin). Off the grid, as in 2D: padding, and
    real tokens with a non-finite coordinate or |eta| > eta_max.

    Args:
        eta: (B, N) pseudorapidity; padded entries may hold anything, including NaN.
        phi: (B, N) azimuth in radians, any branch.
        depth: (B, N) depth coordinate (e.g. mm into the calorimeter); padded entries may hold anything, including NaN.
        valid: (B, N), True for real tokens.
        kernel_size: Odd side k of the kernel in eta and phi, in cells.
        eta_cell: Cell size in eta.
        phi_cells: Number of cells round the circle in phi.
        eta_max: Grid range in eta, [-eta_max, eta_max].
        depth_cell: Cell size in depth.
        depth_kernel_size: Odd kernel extent in depth, in cells; defaults to kernel_size.
        depth_wildcard: If True, a token with a negative depth is at ANY depth: in a pair where either token has
            depth < 0, the depth difference is ignored (dw = 0, the kernel's centre depth slice), in both directions,
            so the neighbourhood stays symmetric. Used to let tracks (depth -1, CLICDataset gmp_track_depth "any")
            interact with clusters at every depth. False (the default): a negative depth joins the first bin.

    Returns:
        neighbours: (B, K * N, N) bool, K = kernel_size**2 * depth_kernel_size; row o * N + i, column j is True when j is
            i's neighbour at offset o = ((du + r) * k + (dv + r)) * kd + (dw + rd), du = u_j - u_i, dv = v_j - v_i
            (wrapped), dw = w_j - w_i: (du, dv, dw) in row-major order, the order of a depthwise
            Conv3d(kernel (k, k, kd)) weight flattened to (D, K). Offsets o and K - 1 - o are opposite, so
            neighbours[o, i, j] == neighbours[K - 1 - o, j, i], as in 2D.
        in_grid: (B, N) bool, the tokens on the grid.
    """
    kd = kernel_size if depth_kernel_size is None else depth_kernel_size
    r, rd = kernel_size // 2, kd // 2
    in_grid = valid & eta.isfinite() & phi.isfinite() & depth.isfinite() & (eta.abs() <= eta_max)
    eta = torch.where(in_grid, eta, 0.0)
    phi = torch.where(in_grid, phi, 0.0)
    depth = torch.where(in_grid, depth, 0.0)
    u = torch.floor((eta + eta_max) / eta_cell).long()
    v = torch.remainder(torch.floor(phi * (phi_cells / (2 * math.pi))), phi_cells).long()
    w = torch.floor(depth.clamp_min(0.0) / depth_cell).long()
    du = u.unsqueeze(-2) - u.unsqueeze(-1)  # (B, N_i, N_j)
    dv = torch.remainder(v.unsqueeze(-2) - v.unsqueeze(-1) + r, phi_cells) - r
    dw = w.unsqueeze(-2) - w.unsqueeze(-1)
    if depth_wildcard:
        wild = in_grid & (depth < 0)
        dw = torch.where(wild.unsqueeze(-2) | wild.unsqueeze(-1), 0, dw)
    pairs = in_grid.unsqueeze(-1) & in_grid.unsqueeze(-2)
    offsets = [pairs & (du == a) & (dv == b) & (dw == c) for a in range(-r, r + 1) for b in range(-r, r + 1) for c in range(-rd, rd + 1)]
    return torch.cat(offsets, dim=-2), in_grid


class GeometricMessagePassing(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        eta_cell: float = 0.2,
        phi_cells: int = 32,
        eta_max: float = 5.2,
        depth_cell: float | None = None,
        depth_kernel_size: int | None = None,
        depth_wildcard: bool = False,
    ) -> None:
        """Geometric message passing (GMP) after PHAT-JeT [2605.21789]: sum the tokens in each (eta, phi) cell, a depthwise
        k x k convolution over the grid, read back at each token's cell, then x + LayerNorm(Linear(.)), as the reference
        GeometricMessagePassing. Adapted to a full-azimuth detector: a fixed grid with phi periodic, padding masked.

        No grid is built: the convolution is evaluated only at the occupied cells, as a sum over the token pairs at each
        kernel offset (gmp_neighbourhood), which is exactly what the reference gathers back to the tokens. The shapes are
        static and there are no atomic scatters, so it is deterministic.

        Args:
            dim: Dimension of the embeddings.
            kernel_size: Odd side k of the square kernel, in cells: symmetric on the periodic axis.
            eta_cell: Cell size in eta.
            phi_cells: Number of cells round the circle in phi (cell size 2pi / phi_cells), at least kernel_size.
            eta_max: Grid range in eta, [-eta_max, eta_max]; tokens outside it skip GMP unchanged.
            depth_cell: If set, 3D GMP: a third, non-periodic depth axis with this cell size (gmp_neighbourhood_3d), and
                a depthwise Conv3d (k, k, depth_kernel_size) in place of the Conv2d. None (the default) is the 2D GMP,
                unchanged: same modules, parameters, initialization and arithmetic.
            depth_kernel_size: 3D only: odd kernel extent in depth, in cells; defaults to kernel_size.
            depth_wildcard: 3D only: tokens with a negative depth (tracks under gmp_track_depth "any") ignore the depth
                difference to every other token (gmp_neighbourhood_3d). Geometry only: no parameter changes.
        """
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(f"GMP kernel_size must be odd and positive, got {kernel_size}")
        if phi_cells < kernel_size:
            raise ValueError(f"GMP phi_cells ({phi_cells}) must be at least kernel_size ({kernel_size}), or the kernel would wrap onto itself")
        if eta_cell <= 0 or eta_max <= 0:
            raise ValueError(f"GMP eta_cell and eta_max must be positive, got {eta_cell}, {eta_max}")
        if depth_cell is None and depth_kernel_size is not None:
            raise ValueError("GMP depth_kernel_size is a 3D setting: it needs depth_cell")
        if depth_cell is None and depth_wildcard:
            raise ValueError("GMP depth_wildcard is a 3D setting: it needs depth_cell")
        if depth_cell is not None:
            if depth_cell <= 0:
                raise ValueError(f"GMP depth_cell must be positive, got {depth_cell}")
            depth_kernel_size = kernel_size if depth_kernel_size is None else depth_kernel_size
            if depth_kernel_size < 1 or depth_kernel_size % 2 == 0:
                raise ValueError(f"GMP depth_kernel_size must be odd and positive, got {depth_kernel_size}")
        self.kernel_size = kernel_size
        self.eta_cell = eta_cell
        self.phi_cells = phi_cells
        self.eta_max = eta_max
        self.depth_cell = depth_cell
        self.depth_kernel_size = depth_kernel_size
        self.depth_wildcard = bool(depth_wildcard)
        # Number of kernel offsets K: the rows of the neighbours tensor are K blocks of N
        self.num_offsets = kernel_size**2 * (depth_kernel_size or 1)
        # Only the parameters of this depthwise conv are used (its weight[c, 0, du, dv] or [c, 0, du, dv, dw], flattened
        # to (D, K) in the neighbourhood's offset order), and nn.Conv2d's / nn.Conv3d's initialization
        if depth_cell is None:
            self.dw = nn.Conv2d(dim, dim, kernel_size, groups=dim, bias=True)
        else:
            self.dw = nn.Conv3d(dim, dim, (kernel_size, kernel_size, depth_kernel_size), groups=dim, bias=True)
        self.pointwise = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    @property
    def geometry(self) -> dict:
        geometry = {"kernel_size": self.kernel_size, "eta_cell": self.eta_cell, "phi_cells": self.phi_cells, "eta_max": self.eta_max}
        if self.depth_cell is not None:
            geometry |= {"depth_cell": self.depth_cell, "depth_kernel_size": self.depth_kernel_size}
            if self.depth_wildcard:
                geometry["depth_wildcard"] = True
        return geometry

    def forward(self, x: Tensor, neighbours: Tensor, in_grid: Tensor) -> Tensor:
        """Update the tokens on the grid.

        Args:
            x: (B, N, D) tokens.
            neighbours: (B, K * N, N) from gmp_neighbourhood (K = k * k) or gmp_neighbourhood_3d, as a float tensor.
            in_grid: (B, N) from gmp_neighbourhood.

        Returns:
            (B, N, D): x + LayerNorm(Linear(conv)) for the tokens on the grid, x unchanged for the rest.
        """
        batch_size, num_tokens, dim = x.shape
        num_offsets = self.num_offsets
        # Zero the tokens off the grid, so a padded inf/NaN cannot reach the others through 0 * inf
        z = torch.bmm(neighbours, torch.where(in_grid.unsqueeze(-1), x, 0.0)).view(batch_size, num_offsets, num_tokens, dim)
        z = torch.einsum("bond,do->bnd", z, self.dw.weight.view(dim, num_offsets)) + self.dw.bias
        return torch.where(in_grid.unsqueeze(-1), x + self.norm(self.pointwise(z)), x)


class DropPath(nn.Module):
    """Randomly drop layers: https://arxiv.org/abs/1603.09382."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep_prob + torch.rand(mask_shape, dtype=x.dtype, device=x.device)
        return x * mask.floor().div(keep_prob)


class LayerScale(nn.Module):
    """Learn residual strength: https://arxiv.org/abs/2103.17239."""

    def __init__(self, dim: int, init_value: float = 1e-5) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim) * init_value)

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


class Residual(nn.Module):
    def __init__(
        self,
        fn: nn.Module,
        dim: int,
        norm: str | None,
        post_norm: bool = False,
        layer_scale: float | None = None,
        drop_path: float = 0.0,
    ) -> None:
        """Neatly wrap x = x + drop(scale * fn(norm(x))).

        Args:
            dim: Dimension of the input and output.
            fn: The module to wrap. Must be non-resizing.
            norm: The normalization layer.
            post_norm: Whether to apply hybrid norm [2503.04598] style post norm.
            layer_scale: Initial value for the layer_scale. If None, no layer_scale is applied.
            drop_path: Drop path rate.

        Raises:
            ValueError: If the input arguments are invalid.
        """
        super().__init__()

        if post_norm and not norm:
            raise ValueError("post_norm is True but no norm is provided.")
        if norm is not None and not isinstance(norm, str):
            raise ValueError("norm must be a string or None.")
        if norm is not None and norm not in NORM_TYPES:
            raise ValueError(f"Unsupported norm: {norm}. Must be one of {list(NORM_TYPES.keys())}")

        self.fn = fn
        self.ls = LayerScale(dim, layer_scale) if layer_scale is not None else nn.Identity()
        self.dp = DropPath(drop_path) if drop_path else nn.Identity()
        self.post_norm = post_norm

        self.norm = NORM_TYPES[norm](dim) if norm else nn.Identity()

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        if self.post_norm:
            x = self.norm(x)
            return x + self.dp(self.ls(self.fn(x, **kwargs)))
        return x + self.dp(self.ls(self.fn(self.norm(x), **kwargs)))


class EncoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int = 0,
        norm: str = "LayerNorm",
        layer_scale: float | None = None,
        drop_path: float = 0.0,
        value_residual: bool = False,
        qkv_norm: bool = False,
        hybrid_norm: bool = False,
        dense_norm_placement: Literal["hybridnorm", "legacy"] = "hybridnorm",
        dense_kwargs: dict | None = None,
        attn_kwargs: dict | None = None,
    ) -> None:
        """Encoder layer: self-attention followed by feed-forward.

        Args:
            dim: Dimension of the embeddings.
            depth: The depth of the layer.
            norm: The normalization layer.
            drop_path: Drop path rate.
            layer_scale: Initial layer_scale value.
            value_residual: Whether to apply a residual connection from initial values.
            qkv_norm: Whether to use qkv norm in the Attention layer
            hybrid_norm: Whether to use HybridNorm from 2503.04598.
            dense_norm_placement: Where the dense layer's norm goes, see get_hybrid_norm_config.
            dense_kwargs: Keyword arguments for dense layer.
            attn_kwargs: Keyword arguments for self-attention layer.
        """
        super().__init__()

        attn_kwargs = attn_kwargs or {}
        dense_kwargs = dense_kwargs or {}

        attn_norm, dense_post_norm, qkv_norm = get_hybrid_norm_config(norm, depth, hybrid_norm, qkv_norm, dense_norm_placement)

        # handle value residual
        attn_kwargs["value_residual"] = value_residual
        attn_kwargs["is_first_layer"] = depth == 0

        self.dim = dim
        residual = partial(Residual, dim=dim, layer_scale=layer_scale, drop_path=drop_path)
        self.attn = residual(Attention(self.dim, qkv_norm=qkv_norm, norm=norm, **attn_kwargs), norm=attn_norm)
        self.dense = residual(Dense(self.dim, **dense_kwargs), norm=norm, post_norm=dense_post_norm)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return self.dense(self.attn(x, **kwargs))


class PatchEncoderLayer(EncoderLayer):
    def __init__(self, dim: int, depth: int = 0, patch_global: bool = True, gmp: dict | None = None, **kwargs) -> None:
        """Patch attention layer after PHAT-JeT [2605.21789]: optional geometric message passing, then exact
        self-attention within each patch, then a global stage where mean-pooled patch tokens and the register
        tokens attend to each other and every patch token is broadcast back to its constituents, then the
        feed-forward.

        Args:
            dim: Dimension of the embeddings.
            depth: The depth of the layer.
            patch_global: Whether to run the global patch-token stage. If False the layer is local only.
            gmp: Keyword arguments for GeometricMessagePassing, run first on the residual stream as in the reference
                PHATBlock. If None (the default) there is no GMP and the layer is unchanged.
            **kwargs: Keyword arguments for EncoderLayer, which also configure the global stage attention.
        """
        super().__init__(dim, depth=depth, **kwargs)
        self.gmp = GeometricMessagePassing(dim, **gmp) if gmp is not None else None
        self.patch_global = patch_global
        if not patch_global:
            return

        attn_kwargs = dict(kwargs.get("attn_kwargs") or {})
        norm = kwargs.get("norm", "LayerNorm")
        attn_norm, _, qkv_norm = get_hybrid_norm_config(
            norm, depth, kwargs.get("hybrid_norm", False), kwargs.get("qkv_norm", False), kwargs.get("dense_norm_placement", "hybridnorm")
        )
        attn_kwargs["value_residual"] = kwargs.get("value_residual", False)
        attn_kwargs["is_first_layer"] = depth == 0
        self.patch_norm = NORM_TYPES[attn_norm](dim) if attn_norm else nn.Identity()
        self.patch_attn = Attention(dim, qkv_norm=qkv_norm, norm=norm, **attn_kwargs)
        self.patch_message = nn.Linear(dim, dim)

    def forward(
        self,
        x: Tensor,
        valid: Tensor,
        registers: Tensor | None = None,
        initial_values: dict | None = None,
        patch_initial_values: dict | None = None,
        gmp_geometry: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Update patched constituents and the register tokens.

        Args:
            x: Constituents in patches, (B, N_P, P, D). Padded slots must be finite.
            valid: (B, N_P, P), True for real constituents.
            registers: Optional register tokens (B, R, D), which take part only in the global stage.
            initial_values: Value residual state of the local attention.
            patch_initial_values: Value residual state of the global attention.
            gmp_geometry: With GMP, the (neighbours, in_grid) of gmp_neighbourhood over the B x (N_P * P) slots of x.

        Returns:
            The updated constituents (padded slots zeroed) and registers.
        """
        batch_size, num_patches, patch_size, dim = x.shape
        patch_valid = valid.any(-1)

        if self.gmp is not None:
            assert gmp_geometry is not None, "a layer with GMP needs gmp_geometry"
            x = self.gmp(x.reshape(batch_size, num_patches * patch_size, dim), *gmp_geometry).view(batch_size, num_patches, patch_size, dim)

        # A patch with no real constituent would give an all-masked softmax (NaN): let its padded
        # slots attend to each other instead. They are zeroed below and never reach a real constituent.
        local_mask = valid | ~patch_valid.unsqueeze(-1)
        x = self.attn(
            x.reshape(batch_size * num_patches, patch_size, dim),
            kv_mask=local_mask.reshape(batch_size * num_patches, patch_size),
            initial_values=initial_values,
        ).reshape(batch_size, num_patches, patch_size, dim)

        if self.patch_global:
            # Mean over the real constituents of each patch, not over P, so padding does not dilute it
            counts = valid.sum(-1, keepdim=True).clamp_min(1).to(x.dtype)
            patch_tokens = torch.where(valid.unsqueeze(-1), x, 0.0).sum(-2) / counts
            tokens, tokens_valid = patch_tokens, patch_valid
            if registers is not None:
                tokens = torch.cat([registers, patch_tokens], dim=1)
                tokens_valid = torch.cat([patch_valid.new_ones(registers.shape[:2]), patch_valid], dim=1)
            # Without registers an event with no real constituent has no valid patch token at all
            tokens_valid = tokens_valid | ~tokens_valid.any(-1, keepdim=True)
            updated = self.patch_attn(self.patch_norm(tokens), kv_mask=tokens_valid, initial_values=patch_initial_values)
            if registers is not None:
                registers = registers + updated[:, : registers.shape[1]]
                updated = updated[:, registers.shape[1] :]
            x = x + self.patch_message(updated).unsqueeze(-2)

        x = self.dense(x)
        if registers is not None:
            registers = self.dense(registers)

        # Keep padded slots at zero, so they stay finite whatever the input held there
        return torch.where(valid.unsqueeze(-1), x, 0.0), registers


class Encoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        dim: int,
        attn_type: str = "torch",
        window_size: int | None = None,
        window_wrap: bool = False,
        score_mod: str | None = None,
        value_residual: bool = False,
        num_register_tokens: int | None = None,
        patch_size: int | None = None,
        patch_global: bool = True,
        gmp: dict | None = None,
        **layer_kwargs,
    ) -> None:
        """Transformer encoder.

        Args:
            num_layers: Number of layers.
            dim: Dimension of the embeddings at each layer.
            attn_type: Type of attention to use: an Attention backend, or "patch" for patch attention (PatchEncoderLayer).
            window_size: Window size for the sliding window.
            window_wrap: Whether to wrap the window by wrapping the input sequence or the mask, depending on the attn_type.
            score_mod: Score modification function.
            value_residual: Add a residual connection from the initial layer values.
            num_register_tokens: Number of register tokens to add at the beginning of the sequence. If None, no register tokens are added.
                Register tokens are removed from the output by default. With patch attention they take part only in the
                global patch-token stage.
            patch_size: attn_type "patch" only, and required there: the real constituents, in sorted order with the
                padding last, are cut into consecutive patches of this many slots and attend exactly within each patch.
            patch_global: attn_type "patch" only: whether to run the global patch-token stage.
            gmp: attn_type "patch" only: if set, every layer starts with geometric message passing on an (eta, phi) grid,
                configured by these GeometricMessagePassing keyword arguments (kernel_size, eta_cell, phi_cells, eta_max),
                and forward needs x_coords. With depth_cell (and optionally depth_kernel_size) set, the grid is
                (eta, phi, depth) and x_coords carries a third, depth column. None (the default) leaves the patch
                encoder without GMP.
            **layer_kwargs: Keyword arguments for EncoderLayer.
        """
        super().__init__()

        assert not window_wrap or window_size, "Window size must be set if window wrap is True."
        assert attn_type != "flex" or score_mod is None, "Score mod is only supported with flex attention."
        assert not (num_register_tokens is not None and window_size is not None), "Register tokens are not compatible with window attention."
        if attn_type == PATCH_ATTN_TYPE:
            if patch_size is None or patch_size < 1:
                raise ValueError(f"attn_type 'patch' needs a positive patch_size, got {patch_size}")
            if window_size is not None or score_mod is not None:
                raise ValueError("Patch attention is not compatible with window attention or score mods.")
            if not patch_global and num_register_tokens is not None:
                # They would be parameters that never reach a constituent
                raise ValueError("Register tokens take part only in the global patch stage: unset num_register_tokens if patch_global is False")
        elif patch_size is not None:
            raise ValueError(f"patch_size is only used with attn_type 'patch', got attn_type {attn_type!r}")
        if gmp is not None and attn_type != PATCH_ATTN_TYPE:
            raise ValueError(f"gmp is only used with attn_type 'patch', got attn_type {attn_type!r}")
        layer_kwargs = layer_kwargs or {}

        self.num_layers = num_layers
        self.dim = dim
        self.attn_type = attn_type
        self.window_size = window_size
        self.window_wrap = window_wrap
        self.score_mod = SCORE_MODS[score_mod] if score_mod else None
        self.value_residual = value_residual
        self.num_register_tokens = num_register_tokens
        self.patch_size = patch_size if attn_type == PATCH_ATTN_TYPE else None
        self.patch_global = patch_global
        self.gmp = gmp
        # 3D GMP when depth_cell is set; the neighbourhood is built from the non-None geometry keys only, so a config
        # that spells out depth_cell: null is the 2D GMP and calls gmp_neighbourhood exactly as before
        self.gmp_coord_dims = 3 if gmp is not None and gmp.get("depth_cell") is not None else 2
        # None-valued keys are dropped; so is depth_wildcard: false, so a 3D config without the wildcard calls
        # gmp_neighbourhood_3d exactly as before
        self._gmp_kwargs = {k: v for k, v in gmp.items() if v is not None and not (k == "depth_wildcard" and not v)} if gmp is not None else None

        # Initialize register tokens if specified
        if self.num_register_tokens is not None:
            self.register_tokens = nn.Parameter(torch.randn(1, self.num_register_tokens, dim))
        else:
            self.register_tokens = None

        # handle masking
        self.mask_mod = None
        self.seq_len = None

        # handle attention
        attn_kwargs = layer_kwargs.get("attn_kwargs", None) or {}
        attn_kwargs["attn_type"] = PATCH_INNER_ATTN_TYPE if attn_type == PATCH_ATTN_TYPE else attn_type
        layer_kwargs["value_residual"] = self.value_residual
        attn_kwargs["window_size"] = window_size
        layer_kwargs["attn_kwargs"] = attn_kwargs

        # Keep the EncoderLayer call literal: jsonargparse resolves **layer_kwargs from it
        if attn_type != PATCH_ATTN_TYPE:
            self.layers = torch.nn.ModuleList([EncoderLayer(dim=dim, depth=i, **layer_kwargs) for i in range(num_layers)])
        else:
            self.layers = torch.nn.ModuleList([
                PatchEncoderLayer(dim=dim, depth=i, patch_global=patch_global, gmp=gmp, **layer_kwargs) for i in range(num_layers)
            ])

    def set_backend(self, attn_type: str):
        if self.patch_size is not None:
            # A patch encoder stays a patch encoder; its patches can only use PATCH_INNER_ATTN_TYPE
            if attn_type not in {PATCH_ATTN_TYPE, PATCH_INNER_ATTN_TYPE}:
                raise ValueError(f"A patch encoder supports the attention backends {[PATCH_ATTN_TYPE, PATCH_INNER_ATTN_TYPE]}, got {attn_type}")
            return
        self.attn_type = attn_type
        for layer in self.layers:
            self.attn_type = layer.attn.fn.set_backend(self.attn_type)

    def forward(
        self, x: Tensor, x_sort_value: Tensor | None = None, kv_mask: Tensor | None = None, x_coords: Tensor | None = None, **kwargs
    ) -> Tensor:
        """Encode (B, S, D) tokens, returned in the input order.

        Args:
            x: (B, S, D) tokens.
            x_sort_value: Optional (B, S) value to sort the tokens by before the layers.
            kv_mask: Optional (B, S) mask, True for real tokens.
            x_coords: (B, S, 2) (eta, phi), or (B, S, 3) (eta, phi, depth) for 3D GMP, of the tokens; required exactly
                when gmp is set.
            **kwargs: Passed to the layers (not with attn_type "patch").
        """
        batch_size = x.shape[0]
        seq_len = x.shape[-2]
        if (x_coords is not None) != (self.gmp is not None):
            raise ValueError("x_coords (eta, phi) are needed by, and only used by, an encoder with gmp")
        if x_coords is not None and x_coords.shape[-1] != self.gmp_coord_dims:
            raise ValueError(f"this GMP needs {self.gmp_coord_dims} coordinates per token, got x_coords {tuple(x_coords.shape)}")

        # If value to sort on is provided, use it to sort the tokens
        # We don't need to use the stable sort assuming that the sort values are unique
        x_sort_idx = None
        if x_sort_value is not None:
            x_sort_idx = sort_index(x_sort_value, kv_mask)
            x = torch.gather(x, dim=-2, index=x_sort_idx.unsqueeze(-1).expand_as(x))

            # Also permute the kv mask if we have one
            if kv_mask is not None:
                kv_mask = torch.gather(kv_mask, dim=-1, index=x_sort_idx)
            # and the GMP coordinates, which must stay with their tokens
            if x_coords is not None:
                x_coords = torch.gather(x_coords, dim=-2, index=x_sort_idx.unsqueeze(-1).expand_as(x_coords))

        if self.patch_size is not None:
            x = self._patch_forward(x, kv_mask, x_coords)
            if x_sort_idx is not None:
                x = torch.gather(x, -2, torch.argsort(x_sort_idx, dim=-1).unsqueeze(-1).expand_as(x))
            return x

        # Add register tokens at the beginning of the sequence
        if self.register_tokens is not None:
            register_tokens = self.register_tokens.expand(batch_size, -1, -1)
            x = torch.cat([register_tokens, x], dim=1)

            # Allow registers to participate in attention if a mask is provided
            if kv_mask is not None:
                register_mask = torch.full((1, self.num_register_tokens), True, device=kv_mask.device, dtype=kv_mask.dtype).expand(batch_size, -1)
                kv_mask = torch.cat([register_mask, kv_mask], dim=1)

        # Handle flash-varlen attention unpadding at encoder level
        varlen_kwargs = None
        if self.attn_type == "flash-varlen" and kv_mask is not None:
            x, indices, varlen_kwargs = unpad_for_flash_varlen(x, kv_mask)
            kwargs["varlen_kwargs"] = varlen_kwargs
        elif self.attn_type == "flash-varlen":
            raise ValueError("kv_mask must be provided for flash-varlen attention.")

        # Initialise sliding window mask
        if self.mask_mod is None and self.attn_type != "flash" and self.window_size:
            self.seq_len = torch.tensor([1], device=x.device)
            self.mask_mod = (
                sliding_window_mask(self.window_size) if not self.window_wrap else sliding_window_mask_wrapped(self.window_size, self.seq_len)
            )

        # Handle masking
        attn_mask = None
        if self.attn_type == "torch" and self.mask_mod:
            attn_mask = create_mask(self.mask_mod, 1, 1, seq_len, seq_len, device=str(x.device))
        elif self.attn_type == "flex" and self.mask_mod:
            self.seq_len[0] = seq_len
            attn_mask = create_block_mask(self.mask_mod, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=str(x.device))

        # Add wrapping for flash attention with sliding window
        if self.attn_type == "flash" and self.window_wrap:
            x = torch.cat([x[:, -self.window_size // 2 :], x, x[:, : self.window_size // 2]], dim=1)

        # Apply layers
        initial_values = {} if self.value_residual else None
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask, score_mod=self.score_mod, initial_values=initial_values, kv_mask=kv_mask, **kwargs)

        # Remove wrapping for flash attention with sliding window
        if self.attn_type == "flash" and self.window_wrap:
            x = x[:, self.window_size // 2 : -self.window_size // 2]

        # Repad sequence if flash-varlen attention is used
        if varlen_kwargs is not None:
            seq_len = seq_len if not self.num_register_tokens else seq_len + self.num_register_tokens
            x = repad_from_flash_varlen(x, batch_size, seq_len, indices)

        # Remove register tokens
        if self.register_tokens is not None:
            x = x[:, self.num_register_tokens :]
            if kv_mask is not None:
                kv_mask = kv_mask[:, self.num_register_tokens :]

        # If we sorted the tokens, undo the sorting
        if x_sort_value is not None and x_sort_idx is not None:
            x_unsort_idx = torch.argsort(x_sort_idx, dim=-1)
            x = torch.gather(x, -2, x_unsort_idx.unsqueeze(-1).expand_as(x))

        return x

    def _patch_forward(self, x: Tensor, kv_mask: Tensor | None, x_coords: Tensor | None = None) -> Tensor:
        """Patch attention over (B, S, D) constituents, with their (B, S, 2) (eta, phi) or (B, S, 3) (eta, phi, depth)
        coordinates for GMP, returning (B, S, D) in the same slot order.
        """
        assert self.patch_size is not None
        batch_size, seq_len, dim = x.shape
        valid = kv_mask if kv_mask is not None else torch.ones(batch_size, seq_len, dtype=torch.bool, device=x.device)

        # Compact the real constituents to the front, keeping their order, so a patch holds consecutive
        # real constituents and only the last partly filled patch mixes them with padding. After a sort
        # with padding last this is the identity.
        order = torch.argsort((~valid).to(torch.int8), dim=-1, stable=True)
        x = torch.gather(x, -2, order.unsqueeze(-1).expand_as(x))
        valid = torch.gather(valid, -1, order)
        x = torch.where(valid.unsqueeze(-1), x, 0.0)

        # Pad the sequence up to a whole number of patches
        num_patches = -(-seq_len // self.patch_size)
        pad = num_patches * self.patch_size - seq_len
        if pad:
            x = torch.cat([x, x.new_zeros(batch_size, pad, dim)], dim=1)
            valid = torch.cat([valid, valid.new_zeros(batch_size, pad)], dim=1)

        # The GMP neighbourhoods, in the same compacted, padded slot order as x, shared by all the layers
        gmp_geometry = None
        if self.gmp is not None:
            assert x_coords is not None
            x_coords = torch.gather(x_coords, -2, order.unsqueeze(-1).expand_as(x_coords))
            if pad:
                x_coords = torch.cat([x_coords, x_coords.new_zeros(batch_size, pad, x_coords.shape[-1])], dim=1)
            if self.gmp_coord_dims == 3:
                neighbours, in_grid = gmp_neighbourhood_3d(x_coords[..., 0], x_coords[..., 1], x_coords[..., 2], valid, **self._gmp_kwargs)
            else:
                # Only the 2D geometry keys: a 2D config may still spell out e.g. depth_wildcard: false
                kw2d = {k: self._gmp_kwargs[k] for k in ("kernel_size", "eta_cell", "phi_cells", "eta_max") if k in self._gmp_kwargs}
                neighbours, in_grid = gmp_neighbourhood(x_coords[..., 0], x_coords[..., 1], valid, **kw2d)
            gmp_geometry = (neighbours.to(x.dtype), in_grid)

        x = x.view(batch_size, num_patches, self.patch_size, dim)
        valid = valid.view(batch_size, num_patches, self.patch_size)

        registers = self.register_tokens.expand(batch_size, -1, -1) if self.register_tokens is not None else None
        initial_values = {} if self.value_residual else None
        patch_initial_values = {} if self.value_residual else None
        for layer in self.layers:
            x, registers = layer(
                x, valid, registers=registers, initial_values=initial_values, patch_initial_values=patch_initial_values, gmp_geometry=gmp_geometry
            )

        # Back to the input slot order, dropping the patch padding
        x = x.reshape(batch_size, -1, dim)[:, :seq_len]
        return torch.gather(x, -2, torch.argsort(order, dim=-1).unsqueeze(-1).expand_as(x))


def change_attn_backends(module: nn.Module, backend: str) -> None:
    """Recursively change the attention backend of a module and all its children.

    Args:
        module: The module to update.
        backend: The attention backend to set.
    """
    if isinstance(module, Encoder):
        module.set_backend(backend)
        return
    if isinstance(module, Attention):
        module.set_backend(backend)
        return
    for child in module.children():
        change_attn_backends(child, backend)
