import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.DiffMamba_Layer import DiffSSMBlock, FeedForward
from layers.SpatialMamba_Layer import SpatialGraphMambaBlock


class MultiScaleEmbedding(nn.Module):
    """
    Multi-scale temporal embedding using Conv1d with different kernel sizes.
    Captures short, medium, and long-range temporal patterns.
    """
    def __init__(self, enc_in, d_model, kernel_sizes=[3, 5, 7], dropout=0.1):
        super().__init__()
        self.enc_in = enc_in
        self.d_model = d_model
        self.n_scales = len(kernel_sizes)
        
        # Multi-scale conv projections
        self.convs = nn.ModuleList([
            nn.Conv1d(
                enc_in, 
                d_model // self.n_scales,
                kernel_size=k,
                padding=k // 2,
                padding_mode='replicate'
            )
            for k in kernel_sizes
        ])
        
        # Combine scales
        self.proj = nn.Linear(d_model // self.n_scales * self.n_scales, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (B, L, C) input, C = enc_in (channels/electrodes)
        Returns:
            out: (B, L, D) embedded output, D = d_model
        """
        # x: (B, L, C) -> (B, C, L) for Conv1d
        x = x.transpose(1, 2)
        
        # Multi-scale convolution
        scales = [conv(x) for conv in self.convs]  # Each: (B, D//n_scales, L)
        
        # Concatenate scales
        x = torch.cat(scales, dim=1)  # (B, D, L)
        
        # Back to (B, L, D)
        x = x.transpose(1, 2)
        
        # Project and normalize
        x = self.proj(x)
        x = self.norm(x)
        x = self.dropout(x)
        
        return x


class MedMambaEncoderBlock(nn.Module):
    """
    Single encoder block: DiffSSM -> SpatialGraphMamba -> FFN
    
    Now returns adjacency matrix for DAG loss computation.
    """
    def __init__(self, d_model, d_ff, enc_in, d_state=16, d_conv=4, expand=2,
                 node_dim=16, dropout=0.1):
        super().__init__()
        
        # Temporal dynamics (tri-branch differential SSM)
        self.diff_ssm = DiffSSMBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout
        )
        
        # Spatial dynamics (graph-aware mamba)
        self.spatial_mamba = SpatialGraphMambaBlock(
            enc_in=enc_in,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            node_dim=node_dim,
            dropout=dropout
        )
        
        # Feed-forward network
        self.ffn = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, L, D)
        Returns:
            out: (B, L, D)
            adj: (enc_in, enc_in) adjacency matrix
        """
        # Temporal processing
        x = self.diff_ssm(x)
        
        # Spatial processing (returns adjacency matrix)
        x, adj = self.spatial_mamba(x)
        
        # FFN
        x = self.ffn(x)
        
        return x, adj


class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.enc_in = configs.enc_in          # Number of input channels (electrodes)
        self.d_model = configs.d_model        # Hidden dimension
        self.d_ff = configs.d_ff              # FFN dimension
        self.e_layers = configs.e_layers      # Number of encoder layers
        self.num_class = configs.num_class    # Number of output classes
        self.dropout = configs.dropout
        
        # SSM-specific parameters
        self.d_state = getattr(configs, 'd_state', 16)   # SSM state dimension
        self.d_conv = getattr(configs, 'd_conv', 4)      # SSM local conv width
        self.expand = getattr(configs, 'expand', 2)      # SSM expansion factor
        
        # Graph-specific parameters
        self.node_dim = getattr(configs, 'nodedim', 16)  # Node embedding dimension
        
        # Multi-scale embedding
        self.embedding = MultiScaleEmbedding(
            enc_in=self.enc_in,
            d_model=self.d_model,
            kernel_sizes=[3, 5, 7],
            dropout=self.dropout
        )
        
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList([
            MedMambaEncoderBlock(
                d_model=self.d_model,
                d_ff=self.d_ff,
                enc_in=self.enc_in,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                node_dim=self.node_dim,
                dropout=self.dropout
            )
            for _ in range(self.e_layers)
        ])
        
        # Final normalization
        self.norm = nn.LayerNorm(self.d_model)
        
        # Classification head
        if self.task_name == "classification":
            self.act = nn.GELU()
            self.dropout_layer = nn.Dropout(self.dropout)
            self.classifier = nn.Linear(self.d_model, self.num_class)
    
    def classification(self, x_enc, x_mark_enc):
        """
        Classification forward pass.
        
        Args:
            x_enc: (B, L, C) input time series
            x_mark_enc: (B, L, D) time features (not used in this model)
        Returns:
            output: (B, num_class) class logits
            adj_list: List of adjacency matrices from each encoder layer
        """
        # Embedding
        x = self.embedding(x_enc)  # (B, L, d_model)
        
        # Collect adjacency matrices from each layer
        adj_list = []
        
        # Encoder blocks
        for block in self.encoder_blocks:
            x, adj = block(x)  # (B, L, d_model), (enc_in, enc_in)
            adj_list.append(adj)
        
        # Final normalization
        x = self.norm(x)
        
        # Mean pooling over time
        x = x.mean(dim=1)  # (B, d_model)
        
        # Classification
        x = self.act(x)
        x = self.dropout_layer(x)
        output = self.classifier(x)  # (B, num_class)
        
        # Return adjacency list for DAG loss computation
        return output, adj_list
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Main forward pass, dispatches to task-specific method.
        """
        if self.task_name == "classification":
            return self.classification(x_enc, x_mark_enc)
        else:
            raise NotImplementedError(f"Task {self.task_name} not implemented for MedMamba")
