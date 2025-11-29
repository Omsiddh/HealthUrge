"""
Complete training script for the Hospital Resource GNN
Usage: python train_gnn.py
"""

import json
import sys
from service import GNNService
from data_generator import HospitalDataGenerator

def main():
    print("\n" + "="*70)
    print("Hospital Resource Management - GNN Training Pipeline")
    print("="*70 + "\n")
    
    # Step 1: Generate training data
    print("STEP 1: Generating Training Data")
    print("-" * 70)
    
    generator = HospitalDataGenerator('hospitals.csv')
    
    # Generate 30 days of hourly snapshots (720 samples)
    history = generator.generate_historical_data(
        days=30, 
        snapshots_per_day=24
    )
    
    # Save for future use
    generator.save_training_data(history, 'training_data.json')
    
    print(f"\n✓ Generated {len(history)} training samples\n")
    
    # Step 2: Initialize GNN Service
    print("STEP 2: Initializing GNN Service")
    print("-" * 70)
    
    gnn_service = GNNService(model_dir="models")
    print()
    
    # Step 3: Train the model
    print("STEP 3: Training GNN Model")
    print("-" * 70)
    
    training_results = gnn_service.train(
        hospital_history=history,
        epochs=100,
        learning_rate=0.001,
        validation_split=0.2,
        early_stopping_patience=15
    )
    
    if training_results:
        print("\nTraining Results:")
        print(f"  Epochs Trained: {training_results['epochs_trained']}")
        print(f"  Best Val Loss: {training_results['best_val_loss']:.4f}")
        
        if training_results['final_metrics']:
            metrics = training_results['final_metrics']
            print(f"\nFinal Metrics:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1 Score: {metrics['f1_score']:.4f}")
            print(f"  MAE: {metrics['mae']:.4f}")
    
    # Step 4: Test inference
    print("\n" + "="*70)
    print("STEP 4: Testing Inference")
    print("-" * 70)
    
    # Generate current snapshot
    current_snapshot = generator.generate_current_snapshot()
    
    # Run inference
    predictions = gnn_service.analyze_resources(current_snapshot)
    
    print("\nStress Predictions for Current Snapshot:")
    print("-" * 70)
    
    for hospital_id, pred in sorted(predictions.items(), 
                                     key=lambda x: x[1]['stress_score'], 
                                     reverse=True):
        stress = pred['stress_score']
        level = pred['stress_level']
        load = pred['current_load']
        capacity = pred['capacity']
        ratio = pred['load_ratio']
        
        # Color coding for terminal
        if level == 'critical':
            marker = '🔴'
        elif level == 'high':
            marker = '🟠'
        elif level == 'moderate':
            marker = '🟡'
        else:
            marker = '🟢'
        
        print(f"{marker} {hospital_id}: Stress={stress:.3f} ({level:8s}) | "
              f"Load={load:4d}/{capacity:4d} ({ratio:.1%})")
    
    # Step 5: Save summary
    print("\n" + "="*70)
    print("STEP 5: Saving Training Summary")
    print("-" * 70)
    
    summary = gnn_service.get_training_summary()
    
    with open('models/training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("✓ Training summary saved to models/training_summary.json")
    
    print("\n" + "="*70)
    print("✓ Training Pipeline Complete!")
    print("="*70)
    print("\nNext Steps:")
    print("  1. Review training metrics in models/training_history.json")
    print("  2. Test the API server: python api_server.py")
    print("  3. Monitor predictions and retrain as needed")
    print("\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
