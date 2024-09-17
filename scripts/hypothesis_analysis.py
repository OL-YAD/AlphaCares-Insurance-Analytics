import numpy as np
import scipy.stats as stats
import pandas as pd

def calculate_risk(data, feature_column, target_column='TotalClaims'):
    """
    Calculate risk as the mean of the target column for each category in the feature column.
    """
    return data.groupby(feature_column)[target_column].mean()

def calculate_margin(data, feature_column):
    """
    Calculate margin (profit) as TotalPremium - TotalClaims for each category in the feature column.
    """
    return data.groupby(feature_column).apply(lambda x: x['TotalPremium'].sum() - x['TotalClaims'].sum())

def chi_square_test(data, feature_column, target_column):
    """
    Perform chi-square test for independence between two categorical variables.
    """
    contingency_table = pd.crosstab(data[feature_column], data[target_column])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    return chi2, p_value

def t_test(group1, group2):
    """
    Perform independent t-test between two groups.
    """
    t_stat, p_value = stats.ttest_ind(group1, group2)
    return t_stat, p_value

def perform_hypothesis_test(data, feature_column, target_column, test_type='chi_square'):
    """
    Perform the appropriate hypothesis test based on the data types.
    """
    if test_type == 'chi_square':
        return chi_square_test(data, feature_column, target_column)
    elif test_type == 't_test':
        group1 = data[data[feature_column] == data[feature_column].unique()[0]][target_column]
        group2 = data[data[feature_column] == data[feature_column].unique()[1]][target_column]
        return t_test(group1, group2)
    else:
        raise ValueError("Unsupported test type. Use 'chi_square' or 't_test'.")




def one_way_anova(data, group_column, metric_column):
    """
    Perform one-way ANOVA test for multiple groups.
    """
    groups = [group for _, group in data.groupby(group_column)[metric_column]]
    f_statistic, p_value = stats.f_oneway(*groups)
    return f_statistic, p_value

def perform_statistical_test(data, group_column, metric_column, test_type='anova'):
    """
    Perform appropriate statistical test based on the number of groups and test type.
    """
    groups = data[group_column].unique()
    
    if test_type == 'anova':
        statistic, p_value = one_way_anova(data, group_column, metric_column)
    elif test_type == 'chi_square':
        statistic, p_value = chi_square_test(data, group_column, metric_column)
    elif test_type == 't_test' and len(groups) == 2:
        group_a = data[data[group_column] == groups[0]][metric_column]
        group_b = data[data[group_column] == groups[1]][metric_column]
        statistic, p_value = t_test(group_a, group_b)
    else:
        return {
            "error": f"Error: Unsupported test type '{test_type}' or incorrect number of groups for the test",
            "test_type": test_type,
            "statistic": None,
            "p_value": None,
            "interpretation": None
        }
    
    interpretation = interpret_results(p_value)
    
    return {
        "test_type": test_type,
        "statistic": statistic,
        "p_value": p_value,
        "interpretation": interpretation
    }

def interpret_results(p_value, alpha=0.05):
    """
    Interpret the results of the hypothesis test.
    """
    if p_value < alpha:
        return f"Reject the null hypothesis (p-value: {p_value:.4f}). There is a significant difference."
    else:
        return f"Fail to reject the null hypothesis (p-value: {p_value:.4f}). There is no significant difference."