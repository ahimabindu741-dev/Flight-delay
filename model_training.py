from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

def train_model(df):
    X = df.drop('Delayed', axis=1)
    y = df['Delayed']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    with open('models/model.pkl', 'wb') as f:
        pickle.dump(model, f)

    return model
