import re
from collections import Counter
from scipy import stats
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np


def get_best_model_features(models, best_model):
    for m in models:
        if m['name'] == best_model:
            return m


def remap_company_location(df):
    category_counts = df['company_location'].value_counts()
    location_threshold = category_counts.median()    
    common_categories = category_counts[category_counts >= location_threshold].index
    return df['company_location'].apply(lambda x: x if x in common_categories else "Other")
    

def ohe_features_and_concat(df, feature_list):
    df1 = df.copy()
    
    encoded_dfs = []
    for feature in feature_list:
        ohe = OneHotEncoder(sparse=False, drop = 'first')
        encoded_features = ohe.fit_transform(df1[[feature]])
        encoded_df = pd.DataFrame(encoded_features, columns = ohe.get_feature_names_out([feature]))
        encoded_dfs.append(encoded_df)
        df1.drop(feature, axis = 1, inplace = True)

    for frame in encoded_dfs:
        df1 = pd.concat([df1, frame], axis = 1)
        
    return df1


def visualize_feature_importances(X_train, result, top):
    threshold = 0.95
    
    feature_importances = result['model'].feature_importances_
    df_feat = pd.DataFrame(list(zip(X_train.columns, feature_importances)), columns = ['feature', 'importance']).sort_values(by="importance", ascending = False).set_index('feature')
    threshold = df_feat['importance'].quantile(threshold)
    
    # Visualization for Average Salary by Country
    plt.figure(figsize=(10, 6))
    df_feat[:top][::-1].plot(kind='barh', color='lightblue')
    plt.title(f'Feature Importances, Top {top}')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

    return df_feat


def train_and_evaluate_model(model_name, X_train, y_train, X_test, y_test):

    threshold = 0.95
    
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, y_train)
    
    # Evaluate the RandomForest model
    y_pred = rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse).round(2)

    feature_importances = rf.feature_importances_
    df_feat = pd.DataFrame(list(zip(X_train.columns, feature_importances)), columns = ['feature', 'importance']).sort_values(by="importance", ascending = False)
    threshold = df_feat['importance'].quantile(threshold)

    best_features = []
    for i, row in df_feat.iterrows():
        if row['importance'] > threshold:
            best_features.append(row['feature'])
    
    model_data = {
        'model': rf,
        'name': model_name,
        'n_features': X_train.shape[1],
        'rmse': rmse,
        'best_features': best_features
    }

    print(f"{model_name} RMSE score: {rmse}")
    print("\n")

    return model_data


def flag_outliers_zscore(df):
    # Flag outliers based on Z-score threshold
    threshold = 3
    
    # Compute Z-scores
    z_scores = np.abs(stats.zscore(df['salary_in_usd']))

    return np.where(z_scores > threshold, 1, 0)

def flag_outliers_via_salary(df, threshold):
    return np.where(df['salary_in_usd'] > threshold, 1, 0)
    
    

def get_unique_job_title_tokens(df):
    tokens = set()
    for title in df['job_title']:
        words = re.findall(r'\b\w+\b', title)
        tokens.update(words)
    return tokens


def get_job_title_token_freqs(df):
    data = []
    for title in df['job_title']:
        words = re.findall(r'\b\w+\b', title)
        data.extend(words)

    c = Counter(data)
    df = pd.DataFrame(c.items(), columns=['token', 'freq']).sort_values(by = 'freq', ascending = False)
    return df