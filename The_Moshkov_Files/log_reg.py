import pandas as pd
import numpy as np
from sklearn.metrics import (precision_score, recall_score, roc_auc_score,
                           balanced_accuracy_score, confusion_matrix, average_precision_score, 
                           precision_recall_curve, roc_curve, f1_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


def log_regression_mkov_splits(df_assay, 
                                endpoint_name, 
                                cp_feats, 
                                base_model,
                                res_dir=None,
                                y_scramble=False,
                                ):    
    # Store results across folds
    fold_results = []
    fold_imps = []
            
    # Define the pipeline:
    pipeline_steps = [('scaler', StandardScaler())]
    pipeline_steps.append(('classifier', base_model))
    pipeline = Pipeline(pipeline_steps)
    
    for i in range(5):
        print(f"\nProcessing fold {i+1}/5")

        # Load and preprocess data:
        train_data, test_data = load_and_preprocess_data(i, df_assay)
        
        # Split into training and test data
        X_train, y_train = train_data[cp_feats], train_data[endpoint_name]
        X_test, y_test = test_data[cp_feats], test_data[endpoint_name]

        # End the current loop if there are no positive or only positive labels in the train/test set:
        if (y_train.sum() == 0) or (y_train.sum() == len(y_train)) or (y_test.sum() == 0) or (y_test.sum() == len(y_test)):
            continue

        # Scramble the labels in y_train:
        if y_scramble:
            y_train = np.random.permutation(y_train)
        
        model = pipeline.fit(X_train, y_train)
        
        # Get predictions using best model
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate area under the precision-recall gain:
        area_under_prg = prg_auc(y_test, y_pred_proba)
        plot_prg_curve(y_test, y_pred_proba)
        
        # Calculate metrics
        metrics = {
            'fold': i,
            'f1_score': f1_score(y_test, y_pred, average="binary", zero_division=0),
            'precision': precision_score(y_test, y_pred, average="binary", zero_division=0),
            'recall': recall_score(y_test, y_pred, average="binary", zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'balanced_acc': balanced_accuracy_score(y_test, y_pred),
            'auprc': average_precision_score(y_test, y_pred_proba),
            'auprg': area_under_prg,
        }
        fold_results.append(metrics)
        
        # Calculate feature importance (for l2 penalty)
        if model.named_steps['classifier'].penalty == 'l2':
            feature_importance = pd.DataFrame({
                'feature': cp_feats,
                'coefficient': np.abs(model.named_steps['classifier'].coef_[0])
            })
            
            feature_importance = feature_importance.sort_values('coefficient', ascending=False)
            
            # Plot feature importance
            plt.figure(figsize=(10, 6), dpi=100)
            plt.barh(feature_importance['feature'][:10], feature_importance['coefficient'][:10])
            plt.title(f'Top 10 Feature Importance (Fold {i+1})')
            plt.xlabel('Absolute Coefficient Value')
            plt.tight_layout()
            if res_dir:
                plt.savefig(f"{res_dir}/feature_importance_fold_{i+1}.png")
            plt.show()
            fold_imps.append(feature_importance)
        
        # Plotting metrics
        plot_metrics(y_test, y_pred, y_pred_proba, i, res_dir)
        
    # Store results across folds
    results_df = pd.DataFrame(fold_results)
    if res_dir:
        results_df.to_csv(f"{res_dir}/results.csv", index=False)

    return results_df, fold_imps


def load_and_preprocess_data(fold, cp_data):
    train_path = f"data/predictions/chemical_cv{fold}/assay_matrix_discrete_train_scaff.csv"
    test_path = f"data/predictions/chemical_cv{fold}/assay_matrix_discrete_test_scaff.csv"

    train_data = pd.read_csv(train_path)[["smiles"]]
    train_data = pd.merge(train_data, cp_data, on="smiles")
    
    test_data = pd.read_csv(test_path)[["smiles"]]
    test_data = pd.merge(test_data, cp_data, on="smiles")
    
    return train_data, test_data


def precision_recall_gain_curve(y_true, y_scores):
    # Sort predictions and true labels by descending score
    sorted_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[sorted_indices]
    
    # Calculate baseline precision and recall
    baseline_precision = np.mean(y_true)
    baseline_recall = np.sum(y_true) / len(y_true)
    
    precisions = []
    recalls = []
    precision_gains = []
    recall_gains = []
    
    for threshold in range(1, len(y_true) + 1):
        # Predictions above threshold considered positive
        predicted_positive = y_true_sorted[:threshold]
        
        # Calculate precision and recall
        true_positives = np.sum(predicted_positive)
        precision = true_positives / threshold
        recall = true_positives / np.sum(y_true)
        
        # Calculate precision gain and recall gain
        precision_gain = precision - baseline_precision
        recall_gain = recall - baseline_recall
        
        precisions.append(precision)
        recalls.append(recall)
        precision_gains.append(precision_gain)
        recall_gains.append(recall_gain)
    
    return precisions, recalls, precision_gains, recall_gains


def prg_auc(y_true, y_scores):
    # Calculate PRG curve
    _, _, precision_gains, recall_gains = precision_recall_gain_curve(y_true, y_scores)
    
    # Calculate area under PRG curve using trapezoidal rule
    prg_auc = np.trapz(precision_gains, x=recall_gains)
    
    return prg_auc


def plot_metrics(y_test, y_pred, y_pred_proba, fold, res_dir=None):
    """Helper function for plotting metrics"""
    plt.figure(figsize=(15, 10), dpi=100)
    
    # 1. Confusion Matrix
    plt.subplot(221)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix (Fold {fold+1})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # 2. ROC Curve
    plt.subplot(222)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    
    # 3. Precision-Recall Curve
    plt.subplot(223)
    precisions, recalls, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.plot(recalls, precisions, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    
    # 4. Metrics Summary
    plt.subplot(224)
    metrics = {
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'ROC AUC': roc_auc,
        'Balanced Accuracy': balanced_accuracy_score(y_test, y_pred)
    }
    plt.bar(metrics.keys(), metrics.values())
    plt.ylim([0, 1])
    plt.title('Metrics Summary')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    if res_dir:
        plt.savefig(f"{res_dir}/metrics_plots_fold_{fold+1}.png")
    plt.show()

def print_threshold_analysis(y_test, y_pred_proba):
    """Helper function for threshold analysis"""
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    print("\nMetrics at different thresholds:")
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        print(f"\nThreshold: {threshold}")
        print(f"Precision: {precision_score(y_test, y_pred):.3f}")
        print(f"Recall: {recall_score(y_test, y_pred):.3f}")
        print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.3f}")


def plot_prg_curve(y_true, y_scores):
    # Calculate PRG curve
    _, _, precision_gains, recall_gains = precision_recall_gain_curve(y_true, y_scores)
    
    # Calculate AUC
    area_under_prg = prg_auc(y_true, y_scores)
    
    # Create the plot
    plt.figure(figsize=(8, 4))
    
    # Plot PRG curve
    plt.plot(recall_gains, precision_gains, color='blue', label='PRG Curve')
    
    # Plot baseline (zero line)
    plt.axhline(y=0, color='red', linestyle='--', label='Baseline')
    plt.axvline(x=0, color='red', linestyle='--')
    
    # Formatting
    plt.title(f'Precision-Recall-Gain (PRG) Curve\nArea Under Curve = {area_under_prg:.4f}')
    plt.xlabel('Recall Gain')
    plt.ylabel('Precision Gain')
    
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()