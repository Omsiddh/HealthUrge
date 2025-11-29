try:
    import torch
    from torch.optim.lr_scheduler import ReduceLROnPlateau
except ImportError:
    torch = None

import os
import json
import pickle
import numpy as np
from datetime import datetime

# FIXED: Changed from relative to absolute imports
from model import ResourceGNN, HospitalStressMetrics
from graph_utils import build_hospital_graph, create_stress_labels

class GNNService:
    def __init__(self, model_dir="models"):
        self.model = None
        self.scaler = None
        self.training_history = []
        
        if torch is None or ResourceGNN is None:
            print("GNNService disabled due to missing dependencies.")
            return
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, "gnn_resource_model.pth")
        self.scaler_path = os.path.join(model_dir, "feature_scaler.pkl")
        self.history_path = os.path.join(model_dir, "training_history.json")
        
        os.makedirs(model_dir, exist_ok=True)
        self._initialize_model()

    def _initialize_model(self):
        """Initialize or load the GNN model"""
        # 9 input features: capacity, load, ratio, available, lat, lon, hour, day, weekend
        self.model = ResourceGNN(num_node_features=9, hidden_channels=32, num_classes=1)
        self.model.to(self.device)
        
        # Load existing model if available
        if os.path.exists(self.model_path):
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✓ Loaded existing GNN model from {self.model_path}")
                
                # Load scaler
                if os.path.exists(self.scaler_path):
                    with open(self.scaler_path, 'rb') as f:
                        self.scaler = pickle.load(f)
                    print(f"✓ Loaded feature scaler")
                
                # Load training history
                if os.path.exists(self.history_path):
                    with open(self.history_path, 'r') as f:
                        self.training_history = json.load(f)
                    print(f"✓ Loaded training history ({len(self.training_history)} epochs)")
                    
            except Exception as e:
                print(f"⚠ Failed to load GNN model: {e}")
                print("Initializing new model...")
        else:
            print("✓ Initialized new GNN model")

    def analyze_resources(self, hospitals):
        """
        Run GNN inference on the current hospital state.
        
        Args:
            hospitals: List of hospital dictionaries
            
        Returns:
            Dictionary mapping hospital_id to stress_score with confidence
        """
        if not self.model or not hospitals:
            return {}
        
        try:
            self.model.eval()
            
            # Build graph (use existing scaler, don't fit)
            data, _ = build_hospital_graph(
                hospitals, 
                k_neighbors=5, 
                max_distance_km=50.0,
                scaler=self.scaler,
                fit_scaler=False
            )
            
            if data is None:
                print("⚠ Failed to build hospital graph")
                return {}
            
            data = data.to(self.device)
            
            with torch.no_grad():
                logits = self.model(data.x, data.edge_index, data.edge_attr)
                scores = torch.sigmoid(logits).cpu().numpy()
            
            # Build result with metadata
            result = {}
            for i, h in enumerate(hospitals):
                stress_score = float(scores[i][0])
                
                # Determine stress level
                if stress_score < 0.3:
                    level = "low"
                elif stress_score < 0.6:
                    level = "moderate"
                elif stress_score < 0.8:
                    level = "high"
                else:
                    level = "critical"
                
                result[h['id']] = {
                    'stress_score': stress_score,
                    'stress_level': level,
                    'current_load': h.get('predicted_load', 0),
                    'capacity': h.get('beds_capacity', 0),
                    'load_ratio': h.get('predicted_load', 0) / h.get('beds_capacity', 1)
                }
            
            return result
            
        except Exception as e:
            print(f"⚠ Error during inference: {e}")
            return {}

    def train(self, hospital_history, epochs=100, batch_size=32, learning_rate=0.001, 
              validation_split=0.2, early_stopping_patience=10):
        """
        Train the GNN model on historical hospital data.
        
        Args:
            hospital_history: List of hospital state snapshots (each is a list of hospital dicts)
            epochs: Number of training epochs
            batch_size: Batch size (not fully implemented for graph batching)
            learning_rate: Learning rate for optimizer
            validation_split: Fraction of data for validation
            early_stopping_patience: Stop if no improvement for N epochs
            
        Returns:
            Dictionary with training metrics
        """
        if not self.model or not hospital_history:
            print("⚠ Cannot train: model not initialized or no training data")
            return None
        
        print(f"\n{'='*60}")
        print(f"Starting GNN Training")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(hospital_history)}")
        print(f"Epochs: {epochs}")
        print(f"Learning rate: {learning_rate}")
        print(f"{'='*60}\n")
        
        # Split data into train and validation
        split_idx = int(len(hospital_history) * (1 - validation_split))
        train_data = hospital_history[:split_idx]
        val_data = hospital_history[split_idx:]
        
        print(f"Train set: {len(train_data)} samples")
        print(f"Validation set: {len(val_data)} samples\n")
        
        # Prepare optimizer and scheduler
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)

        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)        
        criterion = torch.nn.BCEWithLogitsLoss()
   
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_losses = []
            
            for hospitals in train_data:
                # Build graph with scaler fitting on first epoch
                data, self.scaler = build_hospital_graph(
                    hospitals,
                    k_neighbors=5,
                    max_distance_km=50.0,
                    scaler=self.scaler,
                    fit_scaler=(epoch == 0 and self.scaler is None)
                )
                
                if data is None:
                    continue
                
                data = data.to(self.device)
                labels = create_stress_labels(hospitals).to(self.device)
                
                optimizer.zero_grad()
                out = self.model(data.x, data.edge_index, data.edge_attr)
                loss = criterion(out, labels)
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses) if train_losses else float('inf')
            
            # Validation phase
            self.model.eval()
            val_losses = []
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for hospitals in val_data:
                    data, _ = build_hospital_graph(
                        hospitals,
                        scaler=self.scaler,
                        fit_scaler=False
                    )
                    
                    if data is None:
                        continue
                    
                    data = data.to(self.device)
                    labels = create_stress_labels(hospitals).to(self.device)
                    
                    out = self.model(data.x, data.edge_index, data.edge_attr)
                    loss = criterion(out, labels)
                    val_losses.append(loss.item())
                    
                    all_preds.append(torch.sigmoid(out))
                    all_targets.append(labels)
            
            avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
            
            # Calculate metrics
            if all_preds and all_targets:
                all_preds = torch.cat(all_preds)
                all_targets = torch.cat(all_targets)
                metrics = HospitalStressMetrics.calculate_metrics(all_preds, all_targets)
            else:
                metrics = {}
            
            # Update learning rate
            scheduler.step(avg_val_loss)
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                self._save_checkpoint(epoch, optimizer)
            else:
                patience_counter += 1
            
            # Log progress
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Train Loss: {avg_train_loss:.4f}")
                print(f"  Val Loss: {avg_val_loss:.4f}")
                if metrics:
                    print(f"  Val Accuracy: {metrics.get('accuracy', 0):.4f}")
                    print(f"  Val F1: {metrics.get('f1_score', 0):.4f}")
                    print(f"  Val MAE: {metrics.get('mae', 0):.4f}")
                print()
            
            # Store history
            self.training_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            })
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"✓ Early stopping triggered at epoch {epoch+1}")
                break
        
        # Save final training history
        with open(self.history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best Validation Loss: {best_val_loss:.4f}")
        print(f"{'='*60}\n")
        
        return {
            'epochs_trained': len(self.training_history),
            'best_val_loss': best_val_loss,
            'final_metrics': metrics if metrics else None
        }

    def _save_checkpoint(self, epoch, optimizer):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint, self.model_path)
        
        # Save scaler
        if self.scaler:
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
        
        print(f"✓ Checkpoint saved")

    def get_training_summary(self):
        """Get summary of training history"""
        if not self.training_history:
            return "No training history available"
        
        recent = self.training_history[-1]
        return {
            'total_epochs': len(self.training_history),
            'last_train_loss': recent.get('train_loss'),
            'last_val_loss': recent.get('val_loss'),
            'last_metrics': recent.get('metrics'),
            'last_trained': recent.get('timestamp')
        }