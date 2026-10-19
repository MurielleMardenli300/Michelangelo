# temporal_predictors.py
import importlib
import inspect
import sys

import torch
import torch.nn as nn

sys.path.insert(0, "/home/fellahr/4D_MoPred")
from utils.io import custom_load
from models.temporal.mamba.lung_edge_true_mamba import build_lung_edge_mamba_forecaster
from models.temporal.condiNet_Tr_prior_multi import CondiNet_Tr_priormulti
from models.temporal.convlstm import ConvLSTMTemporalPredictor
from models.temporal.lstm import LSTMTemporalPredictor
from models.temporal.gru import GRUTemporalPredictor


# ---------------------------------------------------------------------------
# Surface temporal model location.
#
# EDIT THESE TWO LINES to point at the surface model you trained, or override
# them per run through temporal_predictor_kwargs:
#     {"surface_module": "...", "surface_class": "..."}
#
# The import is done lazily inside the builder, so a wrong path here only
# breaks "condinet_surface" and not the other predictors.
# ---------------------------------------------------------------------------
SURFACE_MODEL_MODULE = "models.temporal.condiNet_surface"
SURFACE_MODEL_CLASS = "CondiNet_Surface"


# ---------------------------------------------------------------------------
# Output normalisation helpers
# ---------------------------------------------------------------------------

def _ensure_tokens(x):
    """
    Normalize predictor output to [B, horizon, D].
    """
    if isinstance(x, list):
        return torch.stack(x, dim=1)

    if torch.is_tensor(x):
        if x.ndim == 2:
            # [B, D] -> [B, 1, D]
            x = x.unsqueeze(1)

        if x.ndim != 3:
            raise ValueError(
                f"Expected temporal tokens [B,H,D], got {tuple(x.shape)}"
            )

        return x

    raise TypeError(
        f"Unsupported temporal predictor output type: {type(x)}"
    )


def _unpack_output(out):
    """
    Accept either
        tokens
        (tokens, kl_loss)
        (tokens, kl_loss, ...extra)
    and return (tokens, kl_loss_or_None).
    """
    if isinstance(out, tuple):
        if len(out) == 0:
            raise ValueError("Temporal predictor returned an empty tuple.")
        feats = out[0]
        kl = out[1] if len(out) > 1 else None
        if kl is not None and not torch.is_tensor(kl):
            kl = None
        return feats, kl
    return out, None


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------

class TemporalPredictorAdapter(nn.Module):
    """
    Common contract seen by the diffusion model:

        tokens, aux = adapter(past, future=None)
        tokens: [B, horizon, D]
        aux:    dict, may contain "kl_loss"

    Class attributes tell the diffusion model WHAT to feed the predictor:

        input_modality: "slices"  -> the 2-D navigator sequence Iseq
                        "surface" -> the abdominal surface sequence Sseq
        time_dim:       axis of that sequence that indexes time
                        (Iseq is [B,2,T,H,W] -> 2)
        requires_future_for_training:
                        True if the net has a posterior that needs the
                        future frames while it is being trained.
    """

    requires_future_for_training: bool = False
    input_modality: str = "slices"
    time_dim: int = 2

    def __init__(self, net, frozen=False):
        super().__init__()
        self.net = net
        self.frozen = frozen

    def forward(self, past, future=None):
        raise NotImplementedError

    def train(self, mode=True):
        """
        Important:
        if the temporal predictor is frozen, keep its BN/dropout in eval mode
        even when the parent model calls model.train().
        """
        super().train(mode)

        if self.frozen:
            self.net.eval()

        return self


class MambaAdapter(TemporalPredictorAdapter):

    requires_future_for_training = False

    def forward(self, past, future=None):

        out = self.net(past)

        # In case your Mamba implementation ever returns (tokens, something)
        if isinstance(out, tuple):
            out = out[0]

        tokens = _ensure_tokens(out)

        return tokens, {}


class CondiNetTRAdapter(TemporalPredictorAdapter):

    requires_future_for_training = True

    def forward(self, past, future=None):

        # DO NOT use:
        #
        # if self.training:
        #     require future
        #
        # because .training is controlled by the parent model and does not
        # necessarily mean that you want the posterior.
        #
        # CondiNet itself already knows:
        # future != None -> posterior
        # future == None -> prior

        out_feats, kl_loss = self.net(past, future)

        tokens = _ensure_tokens(out_feats)

        aux = {}

        if kl_loss is not None:
            aux["kl_loss"] = kl_loss

        return tokens, aux


class CondiNetSurfaceAdapter(TemporalPredictorAdapter):
    """
    Abdominal-surface temporal model.

    Assumed contract (same as CondiNet_Tr_priormulti, but on surfaces):
        out = net(surface_past, surface_future_or_None)
        out = tokens            or
        out = (tokens, kl_loss)
    where tokens is list[horizon] of [B,D] or a [B,horizon,D] tensor.

    future != None -> posterior path (training), future == None -> prior.
    If your surface model is deterministic, pass uses_posterior=False in
    temporal_predictor_kwargs and the diffusion model will never send future.
    """

    requires_future_for_training = True
    input_modality = "surface"
    time_dim = 2  # override with input_time_dim in temporal_predictor_kwargs

    def forward(self, past, future=None):

        out = self.net(past, future)
        out_feats, kl_loss = _unpack_output(out)

        tokens = _ensure_tokens(out_feats)

        aux = {}

        if kl_loss is not None:
            aux["kl_loss"] = kl_loss

        return tokens, aux


class ConvLSTMAdapter(TemporalPredictorAdapter):

    requires_future_for_training = False

    def forward(self, past, future=None):

        # ConvLSTM contract:
        # out_feats = list of horizon tensors [B,D]
        out_feats, _ = self.net(past, future)

        tokens = _ensure_tokens(out_feats)

        return tokens, {}


class LSTMAdapter(TemporalPredictorAdapter):

    requires_future_for_training = False

    def forward(self, past, future=None):

        out_feats, _ = self.net(past, future)

        tokens = _ensure_tokens(out_feats)

        return tokens, {}


class GRUAdapter(TemporalPredictorAdapter):

    requires_future_for_training = False

    def forward(self, past, future=None):

        out_feats, _ = self.net(past, future)

        tokens = _ensure_tokens(out_feats)

        return tokens, {}


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _build_mamba_net(
    *,
    num_frames,
    horizon,
    pre_latent_dim,
    **kw,
):
    return build_lung_edge_mamba_forecaster(
        horizon=horizon,
        latent_dim=pre_latent_dim,
        **kw,
    )


def read_checkpoint_config(path):
    """Return the "config" dict saved in a training checkpoint, or {}."""
    if not path:
        return {}
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # older torch without weights_only
        payload = torch.load(path, map_location="cpu")
    config = payload.get("config") if isinstance(payload, dict) else None
    return dict(config) if isinstance(config, dict) else {}


def _kwargs_from_checkpoint_config(net_cls, checkpoint, reserved):
    """Constructor kwargs recovered from the checkpoint's saved config.

    Only keys whose NAME matches a constructor parameter are used; anything
    else (e.g. argparse names that differ from the constructor's) is reported
    so it can be passed explicitly.
    """
    config = read_checkpoint_config(checkpoint)
    if not config:
        print(f"[Surface] No 'config' found in {checkpoint}; using constructor defaults "
              "for anything not passed explicitly.")
        return {}

    params = inspect.signature(net_cls.__init__).parameters
    accepted = {
        name for name, p in params.items()
        if name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    }
    picked = {k: v for k, v in config.items() if k in accepted and k not in reserved}
    unset = sorted(accepted - set(picked) - set(reserved))
    print(f"[Surface] Constructor kwargs from checkpoint config: {picked}")
    if unset:
        print(f"[Surface] Constructor params NOT in config (left at default "
              f"unless passed explicitly): {unset}")
    return picked


def _build_surface_condinet(
    *,
    num_frames,
    horizon,
    pre_latent_dim,
    surface_module=None,
    surface_class=None,
    config_checkpoint=None,
    **kw,
):
    module_name = surface_module or SURFACE_MODEL_MODULE
    class_name = surface_class or SURFACE_MODEL_CLASS

    try:
        module = importlib.import_module(module_name)
        net_cls = getattr(module, class_name)
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            f"Could not load surface temporal model '{class_name}' from "
            f"'{module_name}'. Edit SURFACE_MODEL_MODULE / SURFACE_MODEL_CLASS "
            f"in temporal_predictors.py or pass surface_module / surface_class "
            f"in temporal_predictor_kwargs."
        ) from exc

    # Constructor args saved with the checkpoint, overridden by explicit kw.
    reserved = {"num_inputs", "horizon", "output_dim"}
    if config_checkpoint:
        kw = {**_kwargs_from_checkpoint_config(net_cls, config_checkpoint, reserved), **kw}

    # Same naming convention as CondiNet_Tr_priormulti. Any surface-specific
    # constructor arguments (n_points, surface channels, ...) go through **kw.
    return net_cls(
        num_inputs=num_frames,
        horizon=horizon,
        output_dim=pre_latent_dim,
        **kw,
    )


def _build_condinet_tr_net(
    *,
    num_frames,
    horizon,
    pre_latent_dim,
    **kw,
):
    return CondiNet_Tr_priormulti(
        num_inputs=num_frames,
        horizon=horizon,
        in_channels=2,
        output_dim=pre_latent_dim,
        **kw,
    )


def _build_convlstm_net(
    *,
    num_frames,
    horizon,
    pre_latent_dim,
    in_channels=2,
    out_channels=(32, 64, 128, 256),
    hidden_ch=128,
    kernel_size=3,
    condi_type="2",
    **kw,
):
    return ConvLSTMTemporalPredictor(
        num_inputs=num_frames,
        horizon=horizon,
        in_channels=in_channels,
        out_channels=list(out_channels),
        output_dim=pre_latent_dim,
        condi_type=condi_type,
        hidden_ch=hidden_ch,
        kernel_size=kernel_size,
        **kw,
    )


def _build_lstm_net(
    *,
    num_frames,
    horizon,
    pre_latent_dim,
    in_channels=2,
    out_channels=(32, 64, 128, 256),
    hidden_dim=128,
    num_layers=1,
    dropout=0.0,
    condi_type="2",
    **kw,
):
    return LSTMTemporalPredictor(
        num_inputs=num_frames,
        horizon=horizon,
        in_channels=in_channels,
        out_channels=list(out_channels),
        output_dim=pre_latent_dim,
        condi_type=condi_type,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        **kw,
    )


def _build_gru_net(
    *,
    num_frames,
    horizon,
    pre_latent_dim,
    in_channels=2,
    out_channels=(32, 64, 128, 256),
    hidden_dim=128,
    num_layers=1,
    dropout=0.0,
    condi_type="2",
    **kw,
):
    return GRUTemporalPredictor(
        num_inputs=num_frames,
        horizon=horizon,
        in_channels=in_channels,
        out_channels=list(out_channels),
        output_dim=pre_latent_dim,
        condi_type=condi_type,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        **kw,
    )


TEMPORAL_PREDICTOR_REGISTRY = {
    "mamba": (
        _build_mamba_net,
        MambaAdapter,
    ),

    "condinet_tr": (
        _build_condinet_tr_net,
        CondiNetTRAdapter,
    ),

    "convlstm": (
        _build_convlstm_net,
        ConvLSTMAdapter,
    ),

    "lstm": (
        _build_lstm_net,
        LSTMAdapter,
    ),

    "gru": (
        _build_gru_net,
        GRUAdapter,
    ),

    "condinet_surface": (
        _build_surface_condinet,
        CondiNetSurfaceAdapter,
    ),
}


def build_temporal_predictor(
    name,
    *,
    num_frames,
    horizon,
    pre_latent_dim,
    checkpoint=None,
    freeze=False,
    **kwargs,
):
    """
    Extra adapter-level kwargs (consumed here, NOT passed to the raw net):

        input_time_dim : int  - time axis of the input sequence
                                (default: adapter.time_dim, 2 for [B,C,T,...])
        uses_posterior : bool - override requires_future_for_training
    """

    try:
        net_builder, adapter_cls = TEMPORAL_PREDICTOR_REGISTRY[name]

    except KeyError:
        raise ValueError(
            f"Unknown temporal_predictor='{name}'. "
            f"Available: {sorted(TEMPORAL_PREDICTOR_REGISTRY)}"
        )

    kwargs = dict(kwargs)
    input_time_dim = kwargs.pop("input_time_dim", None)

    # The surface model reads its own constructor args from its checkpoint.
    if name == "condinet_surface" and checkpoint:
        kwargs.setdefault("config_checkpoint", checkpoint)
    uses_posterior = kwargs.pop("uses_posterior", None)

    # ---------------------------------------------------------
    # Build RAW predictor
    # ---------------------------------------------------------

    net = net_builder(
        num_frames=num_frames,
        horizon=horizon,
        pre_latent_dim=pre_latent_dim,
        **kwargs,
    )

    # ---------------------------------------------------------
    # Load checkpoint into RAW predictor
    #
    # Important because your existing checkpoints don't have
    # the "net." prefix introduced by the adapter.
    # ---------------------------------------------------------

    if checkpoint:

        custom_load(
            net,
            checkpoint,
            device="cpu",
        )

    elif freeze:

        raise ValueError(
            f"freeze=True requires a checkpoint "
            f"for temporal_predictor='{name}'"
        )

    # ---------------------------------------------------------
    # Freeze RAW predictor
    # ---------------------------------------------------------

    if freeze:

        for p in net.parameters():
            p.requires_grad_(False)

        net.eval()

    # ---------------------------------------------------------
    # Wrap
    # ---------------------------------------------------------

    predictor = adapter_cls(
        net,
        frozen=freeze,
    )

    if input_time_dim is not None:
        predictor.time_dim = int(input_time_dim)

    if uses_posterior is not None:
        predictor.requires_future_for_training = bool(uses_posterior)

    # A frozen CondiNet should NOT force the training loop
    # to provide future frames.
    predictor.requires_future_for_training = (
        predictor.requires_future_for_training
        and not freeze
    )

    if freeze:
        predictor.eval()

    print(
        f"[Temporal predictor] name={name} | "
        f"input={predictor.input_modality} (time_dim={predictor.time_dim}) | "
        f"frozen={freeze} | "
        f"needs_future_in_training={predictor.requires_future_for_training}"
    )

    return predictor