import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc

def main_combined():
    """
    Main demonstration comparing black box and transparent models
    on realistic synthetic satellite imagery for defense/disaster response.
    """
    
    # Generate realistic synthetic satellite data
    print("=" * 60)
    print("GENERATING REALISTIC SATELLITE IMAGERY")
    print("=" * 60)
    print("\nSimulating 64x64 grid with:")
    print("  - 12 spectral/spatial features")
    print("  - 5 anomaly types (construction, camouflage, fires, floods, vehicles)")
    print("  - Spatially correlated terrain")
    print("  - Realistic spectral signatures")
    
    df, X, y, feature_names = generate_realistic_satellite_data(grid_size=64)
    
    print(f"\nGenerated {len(df)} pixels")
    print(f"Total anomalies: {y.sum()} ({100*y.sum()/len(y):.1f}%)")
    print(f"\nAnomaly breakdown:")
    anomaly_types = ['Construction', 'Camouflage', 'Fire', 'Flood', 'Vehicles']
    for i in range(1, 6):
        count = (df['anomaly_type'] == i).sum()
        print(f"  {anomaly_types[i-1]}: {count} pixels")
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
        X, y, np.arange(len(X)), test_size=0.3, random_state=42, stratify=y
    )
    
    # Create test dataframe for visualization
    df_test = df.iloc[indices_test].copy().reset_index(drop=True)
    
    # For autoencoder, train only on normal data
    X_train_normal = X_train[y_train == 0]
    
    print(f"\nTraining set: {len(X_train)} pixels ({(y_train==0).sum()} normal, {(y_train==1).sum()} anomalies)")
    print(f"Test set: {len(X_test)} pixels ({(y_test==0).sum()} normal, {(y_test==1).sum()} anomalies)")
    
    # ========================================================================
    # TRAIN BLACK BOX MODEL (Autoencoder)
    # ========================================================================
    print("\n" + "=" * 60)
    print("TRAINING BLACK BOX MODEL (Autoencoder)")
    print("=" * 60)
    
    autoencoder, encoder, ae_pred, mse, threshold, ae_accuracy, ae_scaler, history = build_black_box(
        X_train_normal, X_test, y_test
    )
    
    print(f"Black Box Accuracy: {ae_accuracy:.4f}")
    print(f"Threshold: {threshold:.4f}")
    
    # ========================================================================
    # TRAIN TRANSPARENT MODEL 1 (Decision Tree)
    # ========================================================================
    print("\n" + "=" * 60)
    print("TRAINING TRANSPARENT MODEL 1 (Decision Tree)")
    print("=" * 60)
    
    dt_model, dt_pred, dt_accuracy = train_transparent_model(
        X_train, y_train, X_test, y_test, feature_names
    )
    
    print(f"Decision Tree Accuracy: {dt_accuracy:.4f}")
    
    # ========================================================================
    # TRAIN TRANSPARENT MODEL 2 (Reservoir Computing)
    # ========================================================================
    print("\n" + "=" * 60)
    print("TRAINING TRANSPARENT MODEL 2 (Reservoir Computing)")
    print("=" * 60)
    
    rc_model, rc_pred, rc_accuracy, rc_scaler, feature_importance = train_reservoir_model(
        X_train, y_train, X_test, y_test, feature_names
    )
    
    print(f"Reservoir Computing Accuracy: {rc_accuracy:.4f}")
    
    # ========================================================================
    # CALCULATE ROC CURVES
    # ========================================================================
    print("\n" + "=" * 60)
    print("CALCULATING PERFORMANCE METRICS")
    print("=" * 60)
    
    # Autoencoder
    fpr_ae, tpr_ae, _ = roc_curve(y_test, mse)
    roc_auc_ae = auc(fpr_ae, tpr_ae)
    
    # Decision tree
    dt_proba = dt_model.predict_proba(X_test)[:, 1]
    fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_proba)
    roc_auc_dt = auc(fpr_dt, tpr_dt)
    
    # Reservoir computing
    rc_proba = rc_model.predict_proba(rc_scaler.transform(X_test))[:, 1]
    fpr_rc, tpr_rc, _ = roc_curve(y_test, rc_proba)
    roc_auc_rc = auc(fpr_rc, tpr_rc)
    
    print(f"AUC-ROC Scores:")
    print(f"  Black Box (Autoencoder): {roc_auc_ae:.4f}")
    print(f"  Decision Tree:           {roc_auc_dt:.4f}")
    print(f"  Reservoir Computing:     {roc_auc_rc:.4f}")
    
    # ========================================================================
    # VISUALIZATION 1: Spectral Band Overview
    # ========================================================================
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)
    
    print("\n1. Spectral bands overview...")
    fig_bands = plot_spectral_bands(df_test)
    plt.savefig("spectral_bands.png", dpi=300, bbox_inches='tight')
    print("   Saved: spectral_bands.png")
    plt.close()
    
    # ========================================================================
    # VISUALIZATION 2: RGB and False Color Composites
    # ========================================================================
    print("2. RGB and false-color composites...")
    fig_rgb = plot_rgb_composite(df_test)
    plt.savefig("rgb_composites.png", dpi=300, bbox_inches='tight')
    print("   Saved: rgb_composites.png")
    plt.close()
    
    # ========================================================================
    # VISUALIZATION 3: Model Comparison Grid
    # ========================================================================
    print("3. Model predictions comparison...")
    fig = plt.figure(figsize=(24, 6))
    
    ax1 = plt.subplot(1, 4, 1)
    plot_realistic_satellite_data(df_test, "Ground Truth\n(by Anomaly Type)", 
                                 ax=ax1, show_type=True)
    
    ax2 = plt.subplot(1, 4, 2)
    plot_realistic_satellite_data(df_test, f"Autoencoder\nAcc: {ae_accuracy:.3f} | AUC: {roc_auc_ae:.3f}",
                                 predictions=ae_pred, ax=ax2)
    
    ax3 = plt.subplot(1, 4, 3)
    plot_realistic_satellite_data(df_test, f"Decision Tree\nAcc: {dt_accuracy:.3f} | AUC: {roc_auc_dt:.3f}",
                                 predictions=dt_pred, ax=ax3)
    
    ax4 = plt.subplot(1, 4, 4)
    plot_realistic_satellite_data(df_test, f"Reservoir Computing\nAcc: {rc_accuracy:.3f} | AUC: {roc_auc_rc:.3f}",
                                 predictions=rc_pred, ax=ax4)
    
    plt.tight_layout()
    plt.savefig("model_predictions_comparison.png", dpi=300, bbox_inches='tight')
    print("   Saved: model_predictions_comparison.png")
    plt.close()
    
    # ========================================================================
    # VISUALIZATION 4: Detailed Analysis for Each Model
    # ========================================================================
    print("4. Detailed analysis plots...")
    
    # Autoencoder details
    fig_ae = plot_anomaly_detection_details(df_test, ae_pred, y_test)
    fig_ae.suptitle("Black Box Model (Autoencoder) - Detailed Analysis", fontsize=16, y=0.995)
    plt.savefig("autoencoder_details.png", dpi=300, bbox_inches='tight')
    print("   Saved: autoencoder_details.png")
    plt.close()
    
    # Decision Tree details
    fig_dt = plot_anomaly_detection_details(df_test, dt_pred, y_test)
    fig_dt.suptitle("Transparent Model 1 (Decision Tree) - Detailed Analysis", fontsize=16, y=0.995)
    plt.savefig("decision_tree_details.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Reservoir Computing details
    fig_rc = plot_anomaly_detection_details(df_test, rc_pred, y_test)
    fig_rc.suptitle("Transparent Model 2 (Reservoir Computing) - Detailed Analysis", fontsize=16, y=0.995)
    plt.savefig("reservoir_details.png", dpi=300, bbox_inches='tight')
    print("   Saved: decision_tree_details.png")
    print("   Saved: reservoir_details.png")
    
    # ========================================================================
    # VISUALIZATION 5: ROC Curves and Feature Importance
    # ========================================================================
    print("5. Performance metrics and interpretability...")
    fig = plt.figure(figsize=(20, 10))
    
    # ROC curves
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(fpr_ae, tpr_ae, label=f'Autoencoder (AUC={roc_auc_ae:.3f})', linewidth=2)
    ax1.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC={roc_auc_dt:.3f})', linewidth=2)
    ax1.plot(fpr_rc, tpr_rc, label=f'Reservoir (AUC={roc_auc_rc:.3f})', linewidth=2)
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax1.set_xlabel('False Positive Rate', fontsize=11)
    ax1.set_ylabel('True Positive Rate', fontsize=11)
    ax1.set_title('ROC Curves - Model Comparison', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Autoencoder learning curve
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Mean Squared Error', fontsize=11)
    ax2.set_title('Autoencoder Training Progress', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Reconstruction error distribution
    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(mse[y_test == 0], bins=30, alpha=0.6, label='Normal', color='blue', edgecolor='black')
    ax3.hist(mse[y_test == 1], bins=30, alpha=0.6, label='Anomaly', color='red', edgecolor='black')
    ax3.axvline(threshold, color='k', linestyle='--', linewidth=2, label=f'Threshold={threshold:.3f}')
    ax3.set_xlabel('Reconstruction Error', fontsize=11)
    ax3.set_ylabel('Count', fontsize=11)
    ax3.set_title('Autoencoder: Reconstruction Error Distribution', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Feature importance comparison
    ax4 = plt.subplot(2, 3, 4)
    importances_dt = dt_model.feature_importances_
    importances_rc = np.array([feature_importance[name] for name in feature_names])
    
    x = np.arange(len(feature_names))
    width = 0.35
    
    ax4.bar(x - width/2, importances_dt, width, label='Decision Tree', color='steelblue', edgecolor='black')
    ax4.bar(x + width/2, importances_rc, width, label='Reservoir Computing', color='coral', edgecolor='black')
    
    ax4.set_ylabel('Importance', fontsize=11)
    ax4.set_title('Feature Importance Comparison', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=9)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Accuracy comparison bar chart
    ax5 = plt.subplot(2, 3, 5)
    models = ['Autoencoder', 'Decision\nTree', 'Reservoir\nComputing']
    accuracies = [ae_accuracy, dt_accuracy, rc_accuracy]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    bars = ax5.bar(models, accuracies, color=colors, edgecolor='black', linewidth=1.5)
    ax5.set_ylabel('Accuracy', fontsize=11)
    ax5.set_title('Model Accuracy Comparison', fontsize=12, fontweight='bold')
    ax5.set_ylim(0, 1.0)
    ax5.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Decision tree visualization (simplified)
    ax6 = plt.subplot(2, 3, 6)
    from sklearn.tree import plot_tree
    plot_tree(dt_model, filled=True, feature_names=feature_names,
             class_names=["Normal", "Anomaly"], rounded=True, ax=ax6,
             fontsize=7, max_depth=3)
    ax6.set_title("Decision Tree (depth=3)", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("performance_metrics.png", dpi=300, bbox_inches='tight')
    print("   Saved: performance_metrics.png")
    plt.close()
    
    # ========================================================================
    # EXPLAINABILITY DEMONSTRATION
    # ========================================================================
    print("\n" + "=" * 60)
    print("EXPLAINABILITY COMPARISON")
    print("=" * 60)
    
    # Find a good anomaly sample to explain
    anomaly_indices = np.where(y_test == 1)[0]
    if len(anomaly_indices) > 0:
        # Pick one of each type if possible
        sample_idx = anomaly_indices[0]
        sample_type = df_test.iloc[sample_idx]['anomaly_type']
    else:
        sample_idx = 0
        sample_type = 0
    
    sample_features = X_test[sample_idx]
    sample_label = y_test[sample_idx]
    
    print(f"\nExplaining Sample #{sample_idx}")
    print(f"True Label: {'ANOMALY' if sample_label == 1 else 'NORMAL'}")
    if sample_type > 0:
        print(f"Anomaly Type: {anomaly_types[int(sample_type)-1]}")
    
    print(f"\nFeature Values:")
    for i, name in enumerate(feature_names):
        print(f"  {name:18s}: {sample_features[i]:.3f}")
    
    # Black Box Explanation
    print("\n" + "-" * 60)
    print("1. BLACK BOX MODEL (Autoencoder)")
    print("-" * 60)
    sample_scaled = ae_scaler.transform(sample_features.reshape(1, -1))
    sample_recon = autoencoder.predict(sample_scaled, verbose=0)
    sample_err = np.mean(np.square(sample_scaled - sample_recon))
    
    print(f"Reconstruction Error: {sample_err:.4f}")
    print(f"Threshold:            {threshold:.4f}")
    print(f"Decision:             {'ANOMALY' if sample_err > threshold else 'NORMAL'}")
    print("\n⚠️  NO EXPLANATION: The model cannot explain WHY this classification")
    print("    was made. It's a black box - we only see the error value.")
    
    # Decision Tree Explanation
    print("\n" + "-" * 60)
    print("2. TRANSPARENT MODEL 1 (Decision Tree)")
    print("-" * 60)
    decision_path = explain_transparent_prediction(dt_model, sample_features, feature_names)
    print("Decision Path:")
    for step in decision_path:
        print(f"  {step}")
    print("\n✓ CLEAR EXPLANATION: We can see exactly which rules were applied")
    print("  and why the decision was made.")
    
    # Reservoir Computing Explanation
    print("\n" + "-" * 60)
    print("3. TRANSPARENT MODEL 2 (Reservoir Computing)")
    print("-" * 60)
    rc_sample_scaled = rc_scaler.transform(sample_features.reshape(1, -1))[0]
    explanation = rc_model.explain_prediction(rc_sample_scaled, feature_names)
    
    print(f"Prediction:  {'ANOMALY' if explanation['binary_prediction'] == 1 else 'NORMAL'}")
    print(f"Confidence:  {explanation['prediction']:.4f}")
    print("\nFeature Contributions (sorted by importance):")
    
    sorted_contributions = sorted(explanation['feature_contributions'].items(),
                                 key=lambda x: abs(x[1]), reverse=True)
    for feature, contribution in sorted_contributions:
        bar_length = int(abs(contribution) * 40)
        bar = '█' * bar_length
        print(f"  {feature:18s}: {contribution:+.4f} {bar}")
    
    print("\n✓ FEATURE-LEVEL EXPLANATION: We can see which features contributed")
    print("  most to the decision and by how much.")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY: BLACK BOX vs TRANSPARENT MODELS")
    print("=" * 60)
    
    print("\n📊 PERFORMANCE COMPARISON:")
    print(f"  Autoencoder (Black Box):  Acc={ae_accuracy:.3f}, AUC={roc_auc_ae:.3f}")
    print(f"  Decision Tree:            Acc={dt_accuracy:.3f}, AUC={roc_auc_dt:.3f}")
    print(f"  Reservoir Computing:      Acc={rc_accuracy:.3f}, AUC={roc_auc_rc:.3f}")
    
    print("\n🔍 EXPLAINABILITY COMPARISON:")
    print("  ❌ Black Box: No interpretable explanation")
    print("  ✓ Decision Tree: Clear rule-based logic")
    print("  ✓ Reservoir Computing: Feature contribution analysis")
    
    print("\n💡 KEY INSIGHTS FOR DEFENSE/DISASTER RESPONSE:")
    print("  1. Transparent models provide comparable performance to black boxes")
    print("  2. Explainability is critical for:")
    print("     - Building trust with operators")
    print("     - Validating detections before action")
    print("     - Understanding false positives/negatives")
    print("     - Meeting accountability requirements")
    print("  3. Different transparent approaches offer different tradeoffs:")
    print("     - Decision Trees: Simple, interpretable rules")
    print("     - Reservoir Computing: Richer feature interactions")
    
    print("\n📁 Generated Files:")
    print("  - spectral_bands.png")
    print("  - rgb_composites.png")
    print("  - model_predictions_comparison.png")
    print("  - autoencoder_details.png")
    print("  - decision_tree_details.png")
    print("  - reservoir_details.png")
    print("  - performance_metrics.png")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main_combined()
