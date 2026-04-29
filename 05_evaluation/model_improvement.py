# model_improvement.py
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline

def hyperparameter_tuning(model, param_grid, X_train, y_train, cv=5, scoring='f1_macro'):
    """
    网格搜索 + 交叉验证，返回最佳模型和最优参数。
    """
    grid = GridSearchCV(model, param_grid, cv=cv, scoring=scoring,
                        n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    print(f"Best params: {grid.best_params_}")
    print(f"Best CV {scoring}: {grid.best_score_:.4f}")
    return grid.best_estimator_, grid.best_params_

def feature_selection_pipeline(estimator, X_train, y_train, threshold='median'):
    """
    基于模型特征重要性进行特征选择（适用于树模型等）。
    """
    selector = SelectFromModel(estimator, threshold=threshold)
    selector.fit(X_train, y_train)
    selected_features = selector.get_support()
    print(f"Selected {selected_features.sum()} features out of {X_train.shape[1]}")
    return selector
