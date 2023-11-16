from sklearn.preprocessing import StandardScaler

def interpolation(df):
    df = df.resample('15s').ffill()
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    return df

def standardize(tses):
    scaler = StandardScaler()
    scaler.fit(tses)
    return scaler.transform(tses)
