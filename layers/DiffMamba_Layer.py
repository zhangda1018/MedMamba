
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
except ImportError:
    raise ImportError(
        "mamba_ssm is required for MedMamba. "
        "Install with: pip install mamba-ssm"
    )


class BidirectionalMamba(nn.Module):
    """
    Bidirectional Mamba wrapper.
    Since Mamba is causal by default, we run forward and backward passes
    and combine them for non-causal (classification) tasks.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.mamba_fwd = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.mamba_bwd = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        # Fusion projection
        self.proj = nn.Linear(d_model * 2, d_model)
    
    def forward(self, x):
        """
        Args:
            x: (B, L, D) input tensor
        Returns:
            out: (B, L, D) bidirectional output
        """
        # Forward pass
        out_fwd = self.mamba_fwd(x)
        
        # Backward pass: flip, process, flip back
        x_flip = torch.flip(x, dims=[1])
        out_bwd = self.mamba_bwd(x_flip)
        out_bwd = torch.flip(out_bwd, dims=[1])
        
        # Concatenate and project
        out = torch.cat([out_fwd, out_bwd], dim=-1)
        out = self.proj(out)
        return out


class FrequencyBranch(nn.Module):
    """
    Frequency domain processing branch.
    
    CRITICAL for medical signals:
    - EEG: Alpha (8-13Hz), Beta (14-30Hz), Theta (4-7Hz) waves
    - ECG: Heart rate variability, rhythm analysis
    
    Uses FFT -> Learnable Filter -> IFFT
    """
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Learnable frequency filter (complex-valued)
        # Note: Using real weights and handling complex separately for stability
        self.freq_weight_real = nn.Parameter(torch.ones(d_model))
        self.freq_weight_imag = nn.Parameter(torch.zeros(d_model))
        
        # Scale factor for stability
        self.scale = nn.Parameter(torch.ones(1))
        
        # Post-processing
        self.post_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (B, L, D) input tensor (real-valued)
        Returns:
            out: (B, L, D) frequency-filtered output (real-valued)
        """
        B, L, D = x.shape
        
        # FFT along time dimension
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')  # (B, L//2+1, D)
        
        # Apply learnable frequency filter
        # Construct complex weight
        freq_filter = torch.complex(self.freq_weight_real, self.freq_weight_imag)
        freq_filter = freq_filter.unsqueeze(0).unsqueeze(0)  # (1, 1, D)
        
        # Element-wise multiply in frequency domain
        x_filtered = x_fft * freq_filter * self.scale
        
        # IFFT back to time domain
        x_out = torch.fft.irfft(x_filtered, n=L, dim=1, norm='ortho')  # (B, L, D)
        
        # Post-processing
        x_out = self.post_proj(self.dropout(x_out))
        
        return x_out


class DiffSSMBlock(nn.Module):
    """
    Tri-Branch Differential State Space Model Block.
    
    Three branches:
    1. Raw Branch: Bidirectional Mamba on raw input (captures trends)
    2. Diff Branch: Bidirectional Mamba on temporal differences (captures changes)
    3. Freq Branch: FFT-based filtering (captures rhythmic patterns)
    
    Learnable gating fuses all three branches adaptively.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # === Branch 1: Raw Signal ===
        self.mamba_raw = BidirectionalMamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        
        # === Branch 2: Differential Signal ===
        self.mamba_diff = BidirectionalMamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        
        # === Branch 3: Frequency Domain ===
        self.freq_branch = FrequencyBranch(d_model, dropout)
        
        # === Gating Mechanism ===
        # Separate gates for each branch
        self.gate_raw = nn.Linear(d_model, d_model)
        self.gate_diff = nn.Linear(d_model, d_model)
        self.gate_freq = nn.Linear(d_model, d_model)
        
        # Global gate to weight branch importance
        self.global_gate = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 3),
            nn.Softmax(dim=-1)
        )
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def _compute_diff(self, x):
        """
        Compute temporal differences with first-step padding.
        Δx[t] = x[t] - x[t-1], with Δx[0] = 0
        """
        x_pad = F.pad(x, (0, 0, 1, 0))  # (B, L+1, D)
        x_diff = x - x_pad[:, :-1, :]  # (B, L, D)
        return x_diff
    
    def forward(self, x):
        """
        Args:
            x: (B, L, D) input tensor
        Returns:
            out: (B, L, D) output tensor
        """
        B, L, D = x.shape
        residual = x
        x = self.norm1(x)
        
        # === Branch 1: Raw signal processing ===
        raw_out = self.mamba_raw(x)  # (B, L, D)
        
        # === Branch 2: Differential signal processing ===
        x_diff = self._compute_diff(x)  # (B, L, D)
        diff_out = self.mamba_diff(x_diff)  # (B, L, D)
        
        # === Branch 3: Frequency domain processing ===
        freq_out = self.freq_branch(x)  # (B, L, D)
        
        # === Gating and Fusion ===
        # Per-branch gating (element-wise importance)
        g_raw = torch.sigmoid(self.gate_raw(raw_out))
        g_diff = torch.sigmoid(self.gate_diff(diff_out))
        g_freq = torch.sigmoid(self.gate_freq(freq_out))
        
        raw_gated = g_raw * raw_out
        diff_gated = g_diff * diff_out
        freq_gated = g_freq * freq_out
        
        # Global gating (branch-level weighting)
        concat_mean = torch.cat([
            raw_gated.mean(dim=1),
            diff_gated.mean(dim=1),
            freq_gated.mean(dim=1)
        ], dim=-1)  # (B, 3*D)
        
        global_weights = self.global_gate(concat_mean)  # (B, 3)
        global_weights = global_weights.unsqueeze(1).unsqueeze(-1)  # (B, 1, 3, 1)
        
        # Stack branches and apply global weights
        branches = torch.stack([raw_gated, diff_gated, freq_gated], dim=2)  # (B, L, 3, D)
        fused = (branches * global_weights).sum(dim=2)  # (B, L, D)
        
        # Output projection with residual
        out = self.out_proj(self.dropout(fused))
        out = self.norm2(out + residual)
        
        return out


class FeedForward(nn.Module):
    """Standard FFN block with GELU activation."""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        return self.norm(x + self.net(x))
