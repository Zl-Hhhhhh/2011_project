# evaluation_utils.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)
from sklearn.model_selection import learning_curve

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name="Model"):
    """
    对训练好的模型做出完整评估：
    - 在训练集、测试集、全量数据上计算指标
    - 输出分类报告
    - 绘制混淆矩阵
    - 绘制多分类ROC曲线（OvR）并计算平均AUC
    """
    # 预测
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # 合并全量集
    X_all = np.vstack([X_train, X_test])
    y_all = np.hstack([y_train, y_test])
    y_all_pred = model.predict(X_all)
    
    metrics = {}
    for name, y_true, y_pred in [("Train", y_train, y_train_pred),
                                  ("Test", y_test, y_test_pred),
                                  ("All", y_all, y_all_pred)]:
        metrics[name] = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision (macro)": precision_score(y_true, y_pred, average='macro'),
            "Recall (macro)": recall_score(y_true, y_pred, average='macro'),
            "F1 (macro)": f1_score(y_true, y_pred, average='macro'),
        }
    
    metrics_df = pd.DataFrame(metrics).round(4)
    print(f"\n{model_name} Evaluation:\n", metrics_df)
    
    # 分类报告（以测试集为例）
    print("\nClassification Report (Test):")
    print(classification_report(y_test, y_test_pred))
    
    # 混淆矩阵 - 测试集
    plot_confusion_matrix(y_test, y_test_pred, labels=np.unique(y_all),
                          title=f"{model_name} - Confusion Matrix (Test)")
    
    # ROC曲线 - 测试集（需预测概率）
    if hasattr(model, "predict_proba"):
        plot_roc_curves(model, X_test, y_test, model_name)
    
    return metrics_df

def plot_confusion_matrix(y_true, y_pred, labels, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_roc_curves(model, X_test, y_test, model_name):
    """多分类ROC曲线（One-vs-Rest）"""
    from sklearn.preprocessing import label_binarize
    classes = model.classes_
    y_test_bin = label_binarize(y_test, classes=classes)
    y_score = model.predict_proba(X_test)
    
    plt.figure(figsize=(8,6))
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{cls} (AUC = {roc_auc:.2f})')
    
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} - ROC Curves (One-vs-Rest)')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_learning_curve(estimator, X, y, cv=5, scoring='accuracy'):
    """绘制学习曲线，用于检测过拟合/欠拟合"""
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring,
        train_sizes=np.linspace(0.1, 1.0, 10), random_state=42)
    
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    
    plt.figure(figsize=(8,5))
    plt.plot(train_sizes, train_mean, 'o-', label='Training score')
    plt.plot(train_sizes, test_mean, 'o-', label='Cross-validation score')
    plt.xlabel('Training examples')
    plt.ylabel(scoring)
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.show()
