import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn import preprocessing
from sklearn.metrics import (
    roc_curve, 
    roc_auc_score,
    confusion_matrix, 
)

def make_mi_scores(X, y):
    X = X.copy()
    for colname in X.select_dtypes(["object", "category"]):
        # factorize() convert str to int, but will not increase new columns as get_dummies()
        X[colname], _ = X[colname].factorize()
    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_classif(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    
    return mi_scores


def train_test_standardscaler(X, y, test_size, random_state):
    """Method for preparing train and test dataset, applying the mean
       and standard deviation of training dataset to testing dataset
    
    Args:
        data: the pandas dataframe
        target: the target (dependent) variable
        test_size: 0-1, the ratio of test dataset size
        random_state: a number used for reproducing the same train and test 
        every time
    Returns:
        train and test dataset, the mean and standard deviation of the train dataset
    """

    # split train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    # get mean and standard deviation
    X_train_mean = X_train.mean()
    X_train_std = X_train.std(ddof=0)

    # standardize train dataset
    X_train_ss = (X_train - X_train_mean)/X_train_std
    
    # standardize test dataset
    X_test_ss = (X_test - X_train_mean)/X_train_std

    return (X_train, X_test, 
            y_train, y_test, 
            X_train_ss, X_test_ss)

def get_predict_proba(model, X):
    # predict probabilities
    model_predicted_probs = model.predict_proba(X)
    # keep probabilities for class 1 only
    model_probs = model_predicted_probs[:, 1]
    
    return model_probs
        
def plot_auc(y, y_predict_probs, display_no_skill=True, model_name='Model', title=''):
    """Visualization for ROC curve
    
    Args:
        y: the dependent variable
        y_predict_probs: predicted probabilities to be 1
        display_no_skill: whether to display the no skill roc curve
        model_name: the model to dislay in the graph
        title: the graph title
        
        model_name: the model name that will be displayed in the chart
    Returns:
        The AUC value and the line chart with ROC Curve
    """
    if display_no_skill:
        # generate a no skill prediction which predicts all 0
        ns_probs = [0 for _ in range(len(y))]
        ns_auc = round(roc_auc_score(y, ns_probs), 3)
        ns_fpr, ns_tpr, ns_thresholds = roc_curve(y, ns_probs)
        # summarize scores
        print(f"No Skill: ROC AUC={ns_auc}")
        # display the roc curve for no skill
        plt.plot(ns_fpr, ns_tpr, linestyle='dashed', label='No Skill')
        
    # compute area under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
    model_auc = round(roc_auc_score(y, y_predict_probs) ,3)
    print(f"{model_name}: ROC AUC={model_auc}")

    # calculate roc curves
    model_fpr, model_tpr, model_thresholds = roc_curve(y, y_predict_probs)

    # plot the roc curve for the model
    plt.plot(model_fpr, model_tpr, marker='.', label=model_name)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.title(title)
    
    return model_auc, plt


def plot_cf_matrix(cf_matrix, model_name):
    """Visualization for confusion matrix
    
    Args:
        cf_matrix: the confusion matrix
        model_name: the model name that will be displayed in the chart
    Returns:
        A heatmap for confusion matrix with percentage of
        'True Negative', 'False Positive','False Negative','True Positive'
    """

    group_names = ['True Negative', 'False Positive','False Negative','True Positive']
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
              zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    # heatmap of the matrix
    ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(f"{model_name}: Confusion Matrix")
    
    return ax

    
def binary_classification_scores(cf_matrix, auc='', model_name=''):
    accuracy = np.trace(cf_matrix) / float(np.sum(cf_matrix))
    precision = cf_matrix[1,1] / sum(cf_matrix[:,1])
    recall    = cf_matrix[1,1] / sum(cf_matrix[1,:])
    f1_score  = 2*precision*recall / (precision + recall)

    # save to a dataframe
    df_performance = round(
        pd.DataFrame(
            {
                "Model": [model_name],
                "Accuracy": [accuracy],
                "Precision": [precision],
                "Recall": [recall],
                "F1 Score": [f1_score],
                "AUC": [auc],
            }
        ),
        3,
    )
    df_performance = df_performance.set_index('Model')
    
    return df_performance


