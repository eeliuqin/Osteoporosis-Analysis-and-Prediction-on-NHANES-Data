import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn import preprocessing
from sklearn.metrics import (
    roc_curve, 
    roc_auc_score, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    classification_report, 
    confusion_matrix,
)
import tensorflow as tf

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
        random_state: a number used for reproducing the same train and test every time
        
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


def get_nn_models(input_shape: tuple,
               num_layers: int,
               min_nodes_per_layer: int,
               max_nodes_per_layer: int,
               node_step_size: int = 1,
               hidden_layer_activation: str = 'relu',
               num_nodes_at_output: int = 1,
               output_layer_activation: str = 'linear') -> list:
    """Get list of neural networks with various architecture
    
    Args:
        input_shape: the shape of the input layer
        num_layers: the number of hidden layers
        min_nodes_per_layer: the minimal number of nodes of each hidden layer
        max_nodes_per_layer: the maximal number of nodes of each hidden layer
        node_step_size: how many nodes should be incremented each loop
        hidden_layer_activation: activation for hidden layers
        num_nodes_at_output: # of nodes for the output layer (1 for binary classification)
        output_layer_activation: activation for the output layer, set default to 'linear'
        and apply sigmoid to output later to get more numerically accurate results 
        
    Returns:
        A list of neural networks models with different number of nodes
    """
    
    node_options = list(range(min_nodes_per_layer, max_nodes_per_layer + 1, node_step_size))
    layer_possibilities = [node_options] * num_layers
    layer_node_permutations = list(itertools.product(*layer_possibilities))
    
    models = []
    for permutation in layer_node_permutations:
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
        model_name = ''

        for nodes_at_layer in permutation:
            model.add(tf.keras.layers.Dense(nodes_at_layer, activation=hidden_layer_activation))
            model_name += f'dense{nodes_at_layer}_'

        model.add(tf.keras.layers.Dense(num_nodes_at_output, activation=output_layer_activation))
        model._name = model_name[:-1]
        models.append(model)
        
    return models

def evaludate_nn_models(models: list,
             X_train: np.array,
             y_train: np.array,
             X_test: np.array,
             y_test: np.array,
             epochs: int = 100,
             learning_rate: float = 0.0001,
             need_sigmoid_output: bool = True,
             verbose: int = 0) -> pd.DataFrame:
    """Evaluation for Neural Network models
    
    Args:
        models: the list of neural networks models
        X_train: the training data of independent variables
        y_train: corresponding target variable for X_train
        X_test: the test data of independent variables
        y_test: corresponding target variable for X_test
        epochs: the number times that the learning algorithm will work through the entire training dataset
        learning_rate: how quickly the neural network updates the concepts it has learned
        need_sigmoid_output: it should be True if the models' output layers have no activation, otherwise No
        verbose: the choice that how you want to see the output of your Nural Network while it's training,
        it will show nothing when verbose=0
        
    Returns:
        A dataframe contains accuracy, precision, recall, f1, auc of the input models
    """
    
    results = []
    prediction_probs = []
    
    def train(model: tf.keras.Sequential) -> dict:
        # Change this however you want
        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(learning_rate),
        )
        
        # Train the model
        model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=20,
            validation_split=0.2,
            verbose=verbose
        )
        
        # Make predictions on the test set
        preds = model.predict(X_test)
        # Apply sigmoid to get the final probabilities when required
        if need_sigmoid_output == True:
            preds = tf.nn.sigmoid(preds)
        # The predicted class is 1 when the probability >= 0.5 else 0
        prediction_classes = tf.round(preds)
         
        # Return prediction probabilites and evaluation metrics on the test set
        return preds, {
            'model_name': model.name,
            'accuracy': accuracy_score(y_test, prediction_classes),
            'precision': precision_score(y_test, prediction_classes),
            'recall': recall_score(y_test, prediction_classes),
            'f1': f1_score(y_test, prediction_classes),
            'auc': roc_auc_score(y_test, preds)
        }
    
    # Train every model and save results
    for model in models:
        try:
            print(model.name, end=' ... ')
            preds, res = train(model=model)
            results.append(res)
            prediction_probs.append(preds)
        except Exception as e:
            print(f'{model.name} --> {str(e)}')
    
    metrics = pd.DataFrame(results).round(3)
    
    return prediction_probs, metrics