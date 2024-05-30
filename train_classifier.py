import numpy as np
import pandas as pd
import os
import sklearn.metrics
from pair_features import UIPairFeatures
from train_valid_test_loader import load_train_valid_test_datasets
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

if __name__ == "__main__":
    train_tuple, valid_tuple, test_tuple, n_users, n_items = load_train_valid_test_datasets()
    
    pairs = UIPairFeatures()
    x_tr, ytrue_tr = pairs.train_pair_features(train_tuple)
    x_va, ytrue_va = pairs.pair_features_with_ratings(valid_tuple)
    x_te, ytrue_te = pairs.pair_features_with_ratings(test_tuple)

    print("test set shape", x_te.shape)
    print("train set shape", x_tr.shape)
    print("valid set shape", x_va.shape)

    print("x train: \n", x_tr[1])
    print("x valid: \n", x_va[1])
    print("x test: \n", x_te[1])

    masked_df = pd.read_csv(os.path.join("../data_movie_lens_100k/", "ratings_masked_leaderboard_set.csv"))
    masked_data_tuple = (
        masked_df['user_id'].values,
        masked_df['item_id'].values
    )

    ### random forest
    randforest = RandomForestClassifier(class_weight="balanced", random_state=42)
    parameters = {"n_estimators": [300, 800, 1000, 1500], "max_depth": [10, 30, 70, 100, 200, None], "min_samples_split": [2, 4, 10], "min_samples_leaf": [1, 5, 10, 15, 20]}
    
    final_model = 0
    final_accuracy_score = 0
    max_auc_score = 0

    for n_estimators in parameters["n_estimators"]:
        for max_depth in parameters["max_depth"]:
            for min_samples_split in parameters["min_samples_split"]:
                for min_samples_leaf in parameters["min_samples_leaf"]:
                    curr_model = RandomForestClassifier(class_weight='balanced', 
                                                    min_samples_leaf=min_samples_leaf, 
                                                    n_estimators=n_estimators, 
                                                    random_state=42, 
                                                    max_depth=max_depth, 
                                                    min_samples_split=min_samples_split)
                    print("running for:")
                    print(curr_model)
                    curr_model.fit(x_tr, ytrue_tr)

                    yproba_tr = curr_model.predict_proba(x_tr)[:, 1]
                    auc_tr= sklearn.metrics.roc_auc_score(ytrue_tr, yproba_tr)
                    print("auc train: ", auc_tr)

                    yproba_va = curr_model.predict_proba(x_va)[:, 1]
                    auc_va = sklearn.metrics.roc_auc_score(ytrue_va, yproba_va)
                    print("auc valid: ", auc_va)

                    accuracy_tr = sklearn.metrics.balanced_accuracy_score(ytrue_tr, curr_model.predict(x_tr))
                    print("balanced accuracy train: ", accuracy_tr)
                    accuracy_va = sklearn.metrics.balanced_accuracy_score(ytrue_va, curr_model.predict(x_va))
                    print("balanced accuracy valid: ", accuracy_va)

                    if max_auc_score < auc_va:
                        max_auc_score = auc_va
                        final_accuracy_score = accuracy_va
                        final_model = curr_model

    print("final model: \n", final_model)
    yproba_te = final_model.predict_proba(x_te)[:, 1]
    final_model_auc = sklearn.metrics.roc_auc_score(ytrue_te, yproba_te)
    print("auc test: ", final_model_auc)
    
    accuracy_te = sklearn.metrics.balanced_accuracy_score(ytrue_te, final_model.predict(x_te))
    print("balanced accuracy test: ", accuracy_te)

    print("masked dataset length: ", len(masked_data_tuple[0]))
    x_masked = pairs.pair_features_no_ratings(masked_data_tuple)
    yproba_masked = final_model.predict_proba(x_masked)[:, 1]
    print("masked predictions length:", yproba_masked.shape)
    np.savetxt("masked.txt", yproba_masked, fmt='%.10f')