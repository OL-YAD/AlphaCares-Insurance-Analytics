import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 



# calculate percentage of missing values
def calculate_missing_percentage(dataframe):
    # Determine the total number of elements in the DataFrame
    total_elements = np.prod(dataframe.shape)

    # Count the number of missing values in each column
    missing_values = dataframe.isna().sum()

    # Sum the total number of missing values
    total_missing = missing_values.sum()

    # Compute the percentage of missing values
    percentage_missing = (total_missing / total_elements) * 100

    # Print the result, rounded to two decimal places
    print(f"The dataset has {round(percentage_missing, 2)}% missing values.")


def check_missing_values(df):
    """Check for missing values in the dataset."""
    missing_values = df.isnull().sum()
    missing_percentages = 100 * df.isnull().sum() / len(df)
    column_data_types = df.dtypes
    missing_table = pd.concat([missing_values, missing_percentages, column_data_types], axis=1, keys=['Missing Values', '% of Total Values','Data type'])
    return missing_table.sort_values('% of Total Values', ascending=False).round(2)


def drop_high_missing_columns(df, threshold=50):
    """
    Drop columns with missing values above the specified threshold.
    
    :param df: pandas DataFrame
    :param threshold: percentage threshold for dropping columns (default 50%)
    :return: DataFrame with high-missing columns dropped
    """
    missing_series = df.isnull().sum() / len(df) * 100
    columns_to_drop = missing_series[missing_series > threshold].index
    df_cleaned = df.drop(columns=columns_to_drop)
    print(f"Dropped columns: {list(columns_to_drop)}")
    return df_cleaned

def impute_missing_values(df):
    """
    Impute missing values: mode for categorical, median for numerical columns.
    
    :param df: pandas DataFrame
    :return: DataFrame with imputed values
    """
    for column in df.columns:
        if df[column].dtype == 'object':
            # Categorical column: impute with mode
            mode_value = df[column].mode()[0]
            #df[column].fillna(mode_value)
            df[column] = df[column].fillna(mode_value)

        else:
            # Numerical column: impute with median
            median_value = df[column].median()
            #df[column].fillna(median_value)
            df[column] = df[column].fillna(median_value)

    return df


# Function to plot histogram for numerical columns
def plot_histogram(df, column):
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()


# Function to plot bar chart for categorical columns
def plot_bar_chart(df, column):
    plt.figure(figsize=(12, 6))
    df[column].value_counts().plot(kind='bar')
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

# function to perform correlation heatmap for key numerical columns
def plot_correlation_heatmap(df, columns):
    corr = df[columns].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()


def plot_scatter(df, x, y, hue=None):
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x=x, y=y, hue=hue)
    plt.title(f'{y} vs {x}')
    plt.show()


def plot_boxplot(df, x, y):
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x=x, y=y)
    plt.title(f'Boxplot of {y} by {x}')
    plt.xticks(rotation=45)
    plt.show()


def cap_outliers(df, columns=None):
    """
    Cap outliers in specified numeric columns using the IQR method.
    
    :param df: pandas DataFrame
    :param columns: list of column names to process (if None, all numeric columns will be processed)
    :return: DataFrame with capped outliers
    """
    # Create a copy of the DataFrame to avoid SettingWithCopyWarning
    df_capped = df.copy()
    
    if columns is None:
        columns = df_capped.select_dtypes(include=[np.number]).columns
    
    for column in columns:
        Q1 = df_capped[column].quantile(0.25)
        Q3 = df_capped[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df_capped.loc[df_capped[column] < lower_bound, column] = lower_bound
        df_capped.loc[df_capped[column] > upper_bound, column] = upper_bound
    
    return df_capped


# outlier plot for each individual numeric column
def outlier_box_plots(df):
    for column in df:
        plt.figure(figsize=(10, 5))
        sns.boxplot(x=df[column])
        plt.title(f'Box plot of {column}')
        plt.show()


def trend_over_geography(df, geography_column, value_column):
    grouped = df.groupby(geography_column)[value_column].mean().sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    grouped.plot(kind='bar')
    plt.title(f'Average {value_column} by {geography_column}')
    plt.xticks(rotation=45)
    plt.show()
