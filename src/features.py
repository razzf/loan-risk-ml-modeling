def encode_education_level(
    df, source_col="NAME_EDUCATION_TYPE", target_col="EDUCATION_LEVEL"
):
    """
    Adds an ordinal-encoded column for education level and drops the original
    categorical column to prevent multicollinearity.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the education column.
    - source_col (str): The name of the column with education categories.
    - target_col (str): The name of the new column to store ordinal values.

    Returns:
    - pd.DataFrame: The DataFrame with the new EDUCATION_LEVEL column and
                    the original NAME_EDUCATION_TYPE column removed.
    """
    education_order = {
        "Lower secondary": 0,
        "Secondary / secondary special": 1,
        "Incomplete higher": 2,
        "Higher education": 3,
        "Academic degree": 4,
    }

    df[target_col] = df[source_col].map(education_order)

    df = df.drop(columns=[source_col], errors="ignore")

    return df


def merge_academic_degree_into_higher(df):
    """
    Recode 'Academic degree' as 'Higher education' in the NAME_EDUCATION_TYPE column.

    Parameters:
        df (pd.DataFrame): The input DataFrame with a column 'NAME_EDUCATION_TYPE'.

    Returns:
        pd.DataFrame: The modified DataFrame with updated education type values.
    """
    df["NAME_EDUCATION_TYPE"] = df["NAME_EDUCATION_TYPE"].replace(
        {"Academic degree": "Higher education"}
    )
    return df
