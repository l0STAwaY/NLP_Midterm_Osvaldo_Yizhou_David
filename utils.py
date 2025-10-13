import pandas as pd
import numpy as np


def top_entailment_per_target(df, entail = "ENTAILMENT"):
    """
    For each target, select the row with the highest 'entailment' score.
    
    Parameters:
        df (pd.DataFrame): DataFrame with columns including 'target' and 'entailment'.
        
    Returns:
        pd.DataFrame: DataFrame with one row per target, having the highest entailment score.
    """
    # For each target, find index of row with max entailment
    idx = df.groupby('target')[entail].idxmax(axis=0)
    return df.loc[idx].reset_index(drop=True)

def get_positive_example(df, percent=0.1, score_cols=['ENTAILMENT', 'NEUTRAL', 'CONTRADICTION']):
    """
    From the top entailment per target, select the top X% most confident examples 
    based on the delta between the top and second-highest MNLI scores.
    
    Parameters:
        df (pd.DataFrame): DataFrame with MNLI score columns.
        percent (float): Fraction of top examples to return (0 < percent <= 1)
        score_cols (list): Names of MNLI score columns to consider
        
    Returns:
        pd.DataFrame: DataFrame with top X% most confident examples.
    """
    # First, select top entailment per target
    top_df = top_entailment_per_target(df)
    
    # Extract MNLI score values
    scores = top_df[score_cols]
    
    #Maximum values per row:
    largeset_score = np.max(scores,axis = 1)
    
    # make it so that 
    # all value before -2 are less thanit and all value after it are greater, so we specify that there is only index -1 greater than it
    second_largest = pd.Series([np.partition(scores.iloc[i], -2)[-2] for i in range(len(scores))])
 
    # creates delta series
    
    delta =  largeset_score-second_largest
     
    # this gets us the value on the top
    
    top_df["delta"]= delta
    
    # the smaller index the greater
    df_sorted_delta = top_df.sort_values(by='delta', ascending=False)
    
    top_percent_df = df_sorted_delta[:int(len(df_sorted_delta)*percent)]

    return top_percent_df


    
def generate_contrast_random(positve,contra = "CONTRADICTION"):
    """
    For each entailment pair, generate a negative example by replacing the class
    in the hypothesis with a random different class, and assign the contradict label.
    
    Parameters:
        df (pd.DataFrame): DataFrame with at least 'premise', 'hypothesis', and 'class' columns.
        class_col (str): Column name that indicates the class (c) in the hypothesis.
        premise_col (str): Column name for the premise.
        hypothesis_col (str): Column name for the hypothesis.
        entailment_label (str): Label for positive examples.
        contradict_label (str): Label for negative examples.
        
    Returns:
        pd.DataFrame: DataFrame containing only negative examples with contradict label.
    """
