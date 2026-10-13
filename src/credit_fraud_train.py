import os
from pathlib import Path
import warnings

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, GridSearchCV
from imblearn.pipeline import Pipeline
from credit_fraud_utils_data import load_data, get_sampler
from credit_fraud_utils_eval import evaluate_model
RANDOM_STATE = 42

def train_logistic_regression(X_train_scaled, y_train,technique, sampling_strategy):

    cv = StratifiedKFold(
        n_splits= 5,
        shuffle= True,
        random_state= RANDOM_STATE
    )

    pipeline = Pipeline([
            ('sampler', get_sampler(technique, sampling_strategy, RANDOM_STATE)),
            ('lr', LogisticRegression(random_state= RANDOM_STATE))
        ])

    # hyperparameter seach space

    # Grid size = 3 * 1 * 4 * 1 * 3 = 36 combination 
    param_grid = {
        'lr__C': [0.1 ,1.0 ,10.0],  # Keep this range
        'lr__penalty': ['l2'],
        # Includes None to see if SMOTE alone performs better than SMOTE + Class Weight
        'lr__class_weight': [None, {0: 1, 1: 2}, {0: 1, 1: 5}, {0: 1, 1: 10}, 'balanced'],
        'lr__solver': [ 'lbfgs'],
        'lr__max_iter': [1000,2000,3000],
    }


    # Search for best parameters
    search = GridSearchCV(
        estimator = pipeline,
        param_grid= param_grid,
        scoring = 'average_precision',
        n_jobs = -1,
        refit = True,
        cv= cv,
        verbose= 0
    )
    
    search.fit(X_train_scaled, y_train)
    print(f'Best RF Hyperparameters: {search.best_params_}')
    
    return search.best_estimator_



def train_random_forest(X_train, y_train, technique, sampling_strategy):
    
    cv = StratifiedKFold(n_splits= 5, shuffle= True,random_state= RANDOM_STATE)

    pipeline = Pipeline([
        ('sampler', get_sampler(technique, sampling_strategy, RANDOM_STATE)),
        ('clf', RandomForestClassifier(random_state= RANDOM_STATE))
    ])
    # size = 2 * 4 * 3 = 24 
    param_grid = {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [ 5, 10, 15],
        "clf__class_weight": [None, "balanced", "balanced_subsample"],
    }


    search = RandomizedSearchCV(
        estimator= pipeline,
        param_distributions= param_grid,
        scoring='average_precision',
        n_iter= 10,  # lower than combinations
        n_jobs= 4,
        refit= True,
        cv= cv,
        verbose= 0
    )

    search.fit(X_train, y_train)
    print(f'Best RF Hyperparameters: {search.best_params_}')

    return search.best_estimator_


def train_voting_classifier():
    ...

if __name__ == '__main__':
    data_dir = Path(__file__).resolve().parent.parent / 'data'
    train_path = data_dir / 'train.csv'
    val_path = data_dir / 'val.csv'

    X_train, y_train, X_val, y_val = load_data(train_path, val_path)

    techniques = ['smote', 'random_over', 'random_under']
    sampling_strategies = [0.01, 0.1, 0.2, 0.5]

    experiments = [(t, s) for t in techniques for s in sampling_strategies]
    for technique, strategy in experiments:
        print(f'\n--- Experiment: {technique}, strategy = {strategy} ---')

        

        
        # 2. Train Logistic Regression (tuning class_weight simultaneously)

        lr_model = train_logistic_regression(X_train, y_train, technique= technique, sampling_strategy= strategy)

        # Predict using the already refitted best estimator
        y_pred_lr = lr_model.predict(X_val)
        y_scores_lr = lr_model.predict_proba(X_val)[:, 1]


        evaluate_model(
            lr_model,
            X_val,
            y_val, 
            y_pred_lr,
            y_scores_lr,
            experiment_name=f'LR with smote ration  {strategy}'
        )


        # # 3. Train Random Forest (tuning class_weight simultaneously)
        # print("Training Random Forest...")
        # rf_model = train_random_forest(X_train, y_train, technique, strategy)
        
        # y_pred_rf = rf_model.predict(X_val)
        # y_scores_rf = rf_model.predict_proba(X_val)[:, 1]

        # evaluate_model(rf_model, X_val, y_val, y_pred_rf, y_scores_rf,
        #            experiment_name=f'RF {technique} ratio {strategy}')



# if __name__ == '__main__':
#     data_dir = Path(__file__).resolve().parent.parent / 'data'
#     train_path = data_dir / 'train.csv'
#     val_path = data_dir / 'val.csv'

#     X_train, y_train, X_val, y_val = load_data(train_path, val_path)

#     techniques = ['random_over']
#     sampling_strategies = [0.01, 0.1, 0.2, 0.5]

#     experiments = [(t, s) for t in techniques for s in sampling_strategies]
#     for technique, strategy in experiments:
#         print(f'\n--- Experiment: {technique}, strategy = {strategy} ---')

#         print("Training Random Forest...")
#         rf_model = train_random_forest(X_train, y_train, technique, strategy)

#         y_pred_rf = rf_model.predict(X_val)
#         y_scores_rf = rf_model.predict_proba(X_val)[:, 1]

#         evaluate_model(rf_model, X_val, y_val, y_pred_rf, y_scores_rf,
#                        experiment_name=f'RF {technique} ratio {strategy}')



# 0.01 sampling_strategy is the best with optimized randomforest 