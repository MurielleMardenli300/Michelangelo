# temporal_predictors.py
import torch
import torch.nn as nn
import sys
sys.path.insert(0, "/home/fellahr/4D_MoPred") 
from utils.io import custom_load
from models.temporal.mamba.lung_edge_true_mamba import build_lung_edge_mamba_forecaster
from models.temporal.condiNet_Tr_prior_multi import CondiNet_Tr_priormulti
from models.temporal.convlstm import ConvLSTMTemporalPredictor
from models.temporal.lstm import LSTMTemporalPredictor
from models.temporal.gru import GRUTemporalPredictor


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


class TemporalPredictorAdapter(nn.Module):

    requires_future_for_training: bool = False

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
    
def _build_surface_condinet(
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
    )
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

    try:
        net_builder, adapter_cls = TEMPORAL_PREDICTOR_REGISTRY[name]

    except KeyError:
        raise ValueError(
            f"Unknown temporal_predictor='{name}'. "
            f"Available: {sorted(TEMPORAL_PREDICTOR_REGISTRY)}"
        )

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

    # A frozen CondiNet should NOT force the training loop
    # to provide future frames.
    predictor.requires_future_for_training = (
        predictor.requires_future_for_training
        and not freeze
    )

    if freeze:
        predictor.eval()

    return predictor