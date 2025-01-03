from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import VarianceThreshold, SelectPercentile, mutual_info_classif
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_curve, roc_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer



def run_model(model_selector, target, df, text_features = []):

    models = {
        'xgb': XGBClassifier(),
        'rf': RandomForestClassifier()
    }

    X = df.drop(target, axis = 1)
    y = df[target]


    num_features = X.select_dtypes(exclude = ['object', 'category']).columns
    cat_features = X.select_dtypes(include = ['object', 'category']).columns
    cat_features = [c for c in cat_features if c not in text_features]

    print("Numerical features")
    for f in num_features:
        print(f"- {f}")

    print()
    print("Category features")
    for c in cat_features:
        print(f"- {c}")

    if len(text_features) != 0:
        print()
        print("Text features")
        for t in text_features:
            print(f"- {t}")

    num_preproc = make_pipeline(
        StandardScaler()
    )


    if len(text_features) == 0:
        preproc_transformer = make_column_transformer(
            (num_preproc, num_features),
            (OneHotEncoder(drop = 'first', handle_unknown='ignore'), cat_features),
            remainder = 'drop'
        )
    else:
        preproc_transformer = make_column_transformer(
            (num_preproc, num_features),
            (OneHotEncoder(drop = 'first', handle_unknown='ignore'), cat_features),
            (TfidfVectorizer(), text_features[0]),
            remainder = 'drop'
        )

    preproc = make_pipeline(
        preproc_transformer,
        models[model_selector['model_name']]
    )

    # Split into train and test set for model evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.2, random_state = 42)

    # Fit pipeline to training data
    model_preproc = preproc.fit(X_train, y_train)

    result = cross_validate(model_preproc, X_test, y_test, cv = 5, n_jobs=-1, scoring='accuracy')
    accuracy = round(result['test_score'].mean(), 3)

    # Make predictions
    y_pred = model_preproc.predict(X_test)
    #y_pred_prob = model_preproc.predict_proba(X_test)

    # Print classification report
    print()
    print("Classification Report")
    print(classification_report(y_test, y_pred))

    model_selector['accuracy'] = accuracy
    return model_selector
