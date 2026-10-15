import torch
from torch import nn
import torch.nn.functional as F
from .hedging import MarketState
from .strategies import clamp_into_band

class MLP(nn.Module):
    def __init__(self, in_features, out_features, n_layers=4, n_units=32):
        super().__init__()
        layers, width= [], in_features
        for _ in range(n_layers):
            layers += [nn.Linear(width, n_units), nn.ReLU()]
            width = n_units
        layers.append(nn.Linear(n_units, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
def market_features(state: MarketState, T: float) -> torch.Tensor:
    tau = torch.full_like(state.spot, state.time_to_maturity / T)
    return torch.stack([state.log_moneyness, tau], dim=1)

class FeedForwardHedger(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T
        self.mlp = MLP(in_features=3, out_features=1)

    def forward(self, state: MarketState) -> torch.Tensor:
        x = torch.cat([market_features(state, self.T), state.prev_hedge.unsqueeze(-1)], dim=1)
        dtype = next(self.parameters()).dtype
        correction = torch.tanh(self.mlp(x.to(dtype))).squeeze(-1)
        return state.bs_delta + correction.to(state.spot.dtype)
    
class NoTransactionBandNet(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T
        self.mlp = MLP(in_features=2, out_features=2)

    def band(self, state: MarketState):
        x = market_features(state, self.T)
        dtype = next(self.parameters()).dtype
        widths = self.mlp(x.to(dtype)).to(state.spot.dtype)
        lower = state.bs_delta - F.leaky_relu(widths[:, 0])
        upper = state.bs_delta + F.leaky_relu(widths[:, 1])
        return lower, upper

    def forward(self, state: MarketState) -> torch.Tensor:
        lower, upper = self.band(state)
        return clamp_into_band(state.prev_hedge, lower, upper)