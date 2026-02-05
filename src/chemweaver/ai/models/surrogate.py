"""
AISUAM - AI Docking Surrogate Model
===================================

Multi-modal neural network architecture for predicting docking scores
from ligand representations and optional structural context.

Author: AISUAM Development Team
Version: 1.0.0
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID, uuid4


@dataclass
class ModelConfig:
    """Configuration for Docking Surrogate Model."""
    
    # Model identification
    model_id: UUID = field(default_factory=uuid4)
    model_name: str = "aisuam_surrogate"
    model_version: str = "1.0.0"
    
    # Input dimensions
    node_feature_dim: int = 74  # Atom features
    edge_feature_dim: int = 10  # Bond features
    fingerprint_dim: int = 2048  # Morgan fingerprint
    descriptor_dim: int = 200   # Physicochemical descriptors
    
    # Structural context (optional)
    pocket_embedding_dim: int = 128
    interaction_fingerprint_dim: int = 1024
    use_structural_context: bool = False
    
    # Architecture
    gnn_hidden_dim: int = 128
    gnn_num_layers: int = 5
    gnn_dropout: float = 0.2
    
    mlp_hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    mlp_dropout: float = 0.3
    
    # Multi-task
    num_tasks: int = 2  # docking_score, activity_probability
    task_weights: List[float] = field(default_factory=lambda: [1.0, 1.0])
    
    # Uncertainty
    predict_uncertainty: bool = True
    uncertainty_method: str = "evidential"  # evidential, ensemble_head
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": str(self.model_id),
            "model_name": self.model_name,
            "model_version": self.model_version,
            "input_dims": {
                "node_features": self.node_feature_dim,
                "edge_features": self.edge_feature_dim,
                "fingerprint": self.fingerprint_dim,
                "descriptor": self.descriptor_dim
            },
            "architecture": {
                "gnn_hidden": self.gnn_hidden_dim,
                "gnn_layers": self.gnn_num_layers,
                "mlp_dims": self.mlp_hidden_dims
            },
            "num_tasks": self.num_tasks,
            "uncertainty": self.predict_uncertainty
        }


class GraphConvLayer(nn.Module):
    """
    Graph Convolution Layer with edge features.
    
    Implements message passing with attention mechanism.
    """
    
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        
        # Message network
        self.message_net = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Update network
        self.update_net = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )
        
        # Attention
        self.attention = nn.Linear(hidden_dim, 1)
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of graph convolution.
        
        Args:
            node_features: [num_nodes, node_dim]
            edge_index: [2, num_edges]
            edge_features: [num_edges, edge_dim]
            batch: [num_nodes] batch assignment
            
        Returns:
            Updated node features [num_nodes, node_dim]
        """
        row, col = edge_index
        
        # Create messages
        if edge_features is not None:
            messages = torch.cat([
                node_features[row],
                node_features[col],
                edge_features
            ], dim=-1)
        else:
            messages = torch.cat([
                node_features[row],
                node_features[col],
                torch.zeros(row.size(0), self.edge_dim, device=node_features.device)
            ], dim=-1)
        
        # Compute messages
        messages = self.message_net(messages)
        
        # Attention weights
        attn_weights = F.softmax(self.attention(messages), dim=0)
        messages = messages * attn_weights
        
        # Aggregate messages
        aggregated = torch.zeros_like(node_features)
        aggregated.index_add_(0, col, messages)
        
        # Update
        updated = self.update_net(torch.cat([node_features, aggregated], dim=-1))
        
        return updated + node_features  # Residual connection


class MultiModalFusion(nn.Module):
    """
    Fusion module combining multiple molecular representations.
    
    Combines:
    - GNN output (graph-level)
    - Molecular fingerprint
    - Physicochemical descriptors
    - Optional structural context
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Projection layers for each modality
        self.graph_proj = nn.Linear(config.gnn_hidden_dim, 128)
        self.fp_proj = nn.Linear(config.fingerprint_dim, 128)
        self.desc_proj = nn.Linear(config.descriptor_dim, 128)
        
        # Structural context projection (optional)
        if config.use_structural_context:
            self.pocket_proj = nn.Linear(config.pocket_embedding_dim, 64)
            self.interaction_proj = nn.Linear(config.interaction_fingerprint_dim, 64)
            fusion_input_dim = 128 * 3 + 64 * 2
        else:
            fusion_input_dim = 128 * 3
        
        # Fusion network with attention
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=4,
            batch_first=True
        )
        
        # Final fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(config.mlp_dropout),
            nn.Linear(256, 128),
            nn.ReLU()
        )
    
    def forward(
        self,
        graph_embedding: torch.Tensor,
        fingerprint: torch.Tensor,
        descriptor: torch.Tensor,
        pocket_embedding: Optional[torch.Tensor] = None,
        interaction_fp: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Fuse multiple representations.
        
        Args:
            graph_embedding: [batch, gnn_hidden_dim]
            fingerprint: [batch, fingerprint_dim]
            descriptor: [batch, descriptor_dim]
            pocket_embedding: [batch, pocket_dim] (optional)
            interaction_fp: [batch, interaction_dim] (optional)
            
        Returns:
            Fused representation [batch, 128]
        """
        # Project each modality
        graph_repr = self.graph_proj(graph_embedding)  # [batch, 128]
        fp_repr = self.fp_proj(fingerprint)           # [batch, 128]
        desc_repr = self.desc_proj(descriptor)        # [batch, 128]
        
        # Stack for attention
        modalities = torch.stack([graph_repr, fp_repr, desc_repr], dim=1)  # [batch, 3, 128]
        
        # Self-attention across modalities
        attn_output, _ = self.fusion_attention(modalities, modalities, modalities)
        attn_output = attn_output.mean(dim=1)  # [batch, 128]
        
        # Concatenate with original representations
        fused = [graph_repr, fp_repr, desc_repr, attn_output]
        
        # Add structural context if available
        if self.config.use_structural_context and pocket_embedding is not None:
            pocket_repr = self.pocket_proj(pocket_embedding)
            fused.append(pocket_repr)
            
            if interaction_fp is not None:
                interaction_repr = self.interaction_proj(interaction_fp)
                fused.append(interaction_repr)
        
        fused = torch.cat(fused, dim=-1)
        
        # Final fusion
        output = self.fusion_mlp(fused)
        
        return output


class DockingSurrogateModel(nn.Module):
    """
    Multi-modal Docking Surrogate Model.
    
    Predicts docking scores and activity probabilities from:
    - Molecular graph (GNN)
    - Molecular fingerprint
    - Physicochemical descriptors
    - Optional structural context
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Node and edge embedding
        self.node_embedding = nn.Linear(config.node_feature_dim, config.gnn_hidden_dim)
        self.edge_embedding = nn.Linear(config.edge_feature_dim, config.gnn_hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            GraphConvLayer(
                config.gnn_hidden_dim,
                config.gnn_hidden_dim,
                config.gnn_hidden_dim
            )
            for _ in range(config.gnn_num_layers)
        ])
        
        self.gnn_dropout = nn.Dropout(config.gnn_dropout)
        
        # Graph readout (mean + max pooling)
        self.graph_readout = nn.Sequential(
            nn.Linear(config.gnn_hidden_dim * 2, config.gnn_hidden_dim),
            nn.ReLU()
        )
        
        # Multi-modal fusion
        self.fusion = MultiModalFusion(config)
        
        # Task-specific heads
        self.task_heads = nn.ModuleList()
        for i in range(config.num_tasks):
            head = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(config.mlp_dropout),
                nn.Linear(64, 1)
            )
            self.task_heads.append(head)
        
        # Uncertainty head (if using evidential learning)
        if config.predict_uncertainty and config.uncertainty_method == "evidential":
            self.uncertainty_head = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 4)  # gamma, nu, alpha, beta (NIG parameters)
            )
    
    def forward_gnn(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through GNN.
        
        Args:
            node_features: [num_nodes, node_feature_dim]
            edge_index: [2, num_edges]
            edge_features: [num_edges, edge_feature_dim]
            batch: [num_nodes] batch assignment for graph-level output
            
        Returns:
            Graph-level embedding [batch_size, gnn_hidden_dim]
        """
        # Initial embedding
        x = self.node_embedding(node_features)
        
        if edge_features is not None:
            edge_attr = self.edge_embedding(edge_features)
        else:
            edge_attr = None
        
        # GNN layers
        for layer in self.gnn_layers:
            x = layer(x, edge_index, edge_attr, batch)
            x = self.gnn_dropout(F.relu(x))
        
        # Graph-level readout
        if batch is None:
            # Single graph
            x_mean = x.mean(dim=0, keepdim=True)
            x_max = x.max(dim=0, keepdim=True)[0]
        else:
            # Batch of graphs
            batch_size = batch.max().item() + 1
            x_mean = torch.zeros(batch_size, x.size(1), device=x.device)
            x_max = torch.zeros(batch_size, x.size(1), device=x.device)
            
            for i in range(batch_size):
                mask = batch == i
                x_mean[i] = x[mask].mean(dim=0)
                x_max[i] = x[mask].max(dim=0)[0]
        
        graph_repr = torch.cat([x_mean, x_max], dim=-1)
        graph_repr = self.graph_readout(graph_repr)
        
        return graph_repr
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        fingerprint: torch.Tensor,
        descriptor: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
        pocket_embedding: Optional[torch.Tensor] = None,
        interaction_fp: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        return_uncertainty: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Full forward pass.
        
        Args:
            node_features: [num_nodes, node_feature_dim] or [batch, num_nodes, node_dim]
            edge_index: [2, num_edges]
            fingerprint: [batch, fingerprint_dim]
            descriptor: [batch, descriptor_dim]
            edge_features: [num_edges, edge_feature_dim]
            pocket_embedding: [batch, pocket_dim] (optional)
            interaction_fp: [batch, interaction_dim] (optional)
            batch: [num_nodes] batch assignment
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            predictions: [batch, num_tasks]
            uncertainty: [batch, num_tasks] (if return_uncertainty=True)
        """
        # GNN encoding
        graph_embedding = self.forward_gnn(
            node_features, edge_index, edge_features, batch
        )
        
        # Multi-modal fusion
        fused = self.fusion(
            graph_embedding,
            fingerprint,
            descriptor,
            pocket_embedding,
            interaction_fp
        )
        
        # Task predictions
        predictions = []
        for head in self.task_heads:
            pred = head(fused)
            predictions.append(pred)
        
        predictions = torch.cat(predictions, dim=-1)  # [batch, num_tasks]
        
        # Uncertainty prediction
        if return_uncertainty and self.config.predict_uncertainty:
            if self.config.uncertainty_method == "evidential":
                # Evidential learning output
                nig_params = self.uncertainty_head(fused)
                return predictions, nig_params
            else:
                # Standard uncertainty
                uncertainty = torch.ones_like(predictions) * 0.5
                return predictions, uncertainty
        
        return predictions
    
    def predict_docking_score(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        fingerprint: torch.Tensor,
        descriptor: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Convenience method for docking score prediction.
        
        Returns dict with score, uncertainty, and confidence interval.
        """
        output = self.forward(
            node_features, edge_index, fingerprint, descriptor,
            return_uncertainty=True, **kwargs
        )
        
        if isinstance(output, tuple):
            predictions, uncertainty = output
        else:
            predictions = output
            uncertainty = torch.ones_like(predictions) * 0.5
        
        # Assuming task 0 is docking score
        score = predictions[:, 0]
        
        return {
            "score": score,
            "uncertainty": uncertainty[:, 0] if uncertainty.dim() > 1 else uncertainty,
            "confidence_interval": (
                score - 1.96 * uncertainty[:, 0],
                score + 1.96 * uncertainty[:, 0]
            ) if uncertainty.dim() > 1 else (score - 1.96 * uncertainty, score + 1.96 * uncertainty)
        }


# Export
__all__ = [
    'ModelConfig',
    'GraphConvLayer',
    'MultiModalFusion',
    'DockingSurrogateModel'
]
