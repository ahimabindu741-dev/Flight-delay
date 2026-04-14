import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(path):
    df = pd.read_csv(path)
    df = df.dropna()

    df['Delayed'] = df['ArrDelay'].apply(lambda x: 1 if x > 15 else 0)
    df = df.drop(['ArrDelay'], axis=1)

    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    return df
