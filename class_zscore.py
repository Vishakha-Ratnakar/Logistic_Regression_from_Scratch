

def z_score(df):
    # copy the dataframe
    # df_std = df.copy()
    #  df_std = df_std.drop('fire', axis=1)
    # apply the z-score method
    for column in df.columns:
        df[column] = (df[column] - df[column].mean()) / df[column].std()

    return df
