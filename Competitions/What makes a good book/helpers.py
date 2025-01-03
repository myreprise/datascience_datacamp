import matplotlib.pyplot as plt
import seaborn as sns

def sort_dict_by_keys(d):
    """
    Sort a dictionary by its keys.

    Parameters:
    d (dict): The dictionary to sort.

    Returns:
    dict: A new dictionary sorted by keys.
    """
    return dict(sorted(d.items()))


def plot_hist_boxplot(df, col, target):
    plt.figure(figsize = (16, 6))
    
    plt.subplot(1, 2, 1)
    plt.title(f"Histplot of {col}")
    sns.histplot(data = df, x = col, hue = target, kde=True)
     
    plt.subplot(1, 2, 2)
    plt.title(f"Boxplot of {col}")
    sns.boxplot(data = df, x = col, hue = target)
    plt.tight_layout()


def get_feature_names_from_column_transformer(column_transformer):
    feature_names = []
    for name, transformer, columns in column_transformer.transformers_:
        if hasattr(transformer, 'get_feature_names_out'):
            transformer_feature_names = transformer.get_feature_names_out()
            feature_names.extend([f"{name}__{feature}" for feature in transformer_feature_names])
        else:
            feature_names.extend([f"{name}__{col}" for col in columns])
    return feature_names