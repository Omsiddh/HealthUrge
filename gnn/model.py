try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GATConv, global_mean_pool
    
    class ResourceGNN(torch.nn.Module):
        def __init__(self, num_node_features, hidden_channels=32, num_classes=1, dropout=0.3):
            """
            Improved Graph Attention Network for hospital stress prediction.
            
            Args:
                num_node_features: Number of input features per node
                hidden_channels: Size of hidden layers
                num_classes: Number of output classes (1 for stress score)
                dropout: Dropout rate for regularization
            """
            super(ResourceGNN, self).__init__()
            
            # Deeper network with 3 GAT layers for better feature propagation
            self.conv1 = GATConv(num_node_features, hidden_channels, heads=4, concat=True, edge_dim=3)
            self.bn1 = torch.nn.BatchNorm1d(hidden_channels * 4)
            
            self.conv2 = GATConv(hidden_channels * 4, hidden_channels, heads=4, concat=True, edge_dim=3)
            self.bn2 = torch.nn.BatchNorm1d(hidden_channels * 4)
            
            self.conv3 = GATConv(hidden_channels * 4, hidden_channels, heads=2, concat=True, edge_dim=3)
            self.bn3 = torch.nn.BatchNorm1d(hidden_channels * 2)
            
            # Additional layers for node-level prediction
            self.fc1 = torch.nn.Linear(hidden_channels * 2, hidden_channels)
            self.fc2 = torch.nn.Linear(hidden_channels, num_classes)
            
            self.dropout = dropout

        def forward(self, x, edge_index, edge_attr=None, batch=None):
            """
            Forward pass through the network.
            
            Args:
                x: Node feature matrix [num_nodes, num_features]
                edge_index: Graph connectivity [2, num_edges]
                edge_attr: Edge features [num_edges, edge_features]
                batch: Batch vector (for batched graphs)
            
            Returns:
                Node-level predictions [num_nodes, num_classes]
            """
            # First GAT layer
            x = self.conv1(x, edge_index, edge_attr=edge_attr)
            x = self.bn1(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Second GAT layer
            x = self.conv2(x, edge_index, edge_attr=edge_attr)
            x = self.bn2(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Third GAT layer
            x = self.conv3(x, edge_index, edge_attr=edge_attr)
            x = self.bn3(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Final prediction layers
            x = self.fc1(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            x = self.fc2(x)
            
            return x  # Return logits (sigmoid applied during inference/loss)

    
    class HospitalStressMetrics:
        """Utility class for calculating evaluation metrics"""
        
        @staticmethod
        def calculate_metrics(predictions, targets, threshold=0.5):
            """
            Calculate classification metrics for stress prediction.
            
            Args:
                predictions: Model predictions [0, 1]
                targets: Ground truth labels [0, 1]
                threshold: Classification threshold
            
            Returns:
                Dictionary of metrics
            """
            pred_binary = (predictions >= threshold).float()
            target_binary = (targets >= threshold).float()
            
            # True/False Positives/Negatives
            tp = ((pred_binary == 1) & (target_binary == 1)).sum().item()
            tn = ((pred_binary == 0) & (target_binary == 0)).sum().item()
            fp = ((pred_binary == 1) & (target_binary == 0)).sum().item()
            fn = ((pred_binary == 0) & (target_binary == 1)).sum().item()
            
            # Metrics
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Regression metrics
            mse = F.mse_loss(predictions, targets).item()
            mae = F.l1_loss(predictions, targets).item()
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'mse': mse,
                'mae': mae,
                'tp': tp,
                'tn': tn,
                'fp': fp,
                'fn': fn
            }

except ImportError:
    print("Warning: Torch or Torch Geometric not found. GNN features will be disabled.")
    ResourceGNN = None
    HospitalStressMetrics = None