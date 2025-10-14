import pandas as pd
import numpy as np
import random

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



def get_positive_example(df, percent=0.1, score_cols=['entailment', 'neutral', 'contradiction']):
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

    
    # Extract MNLI score values
    top_df = top_entailment_per_target(df)
    scores = df.groupby("target")["ENTAILMENT"]
    # print(  scores)
    #Maximum values per row:
    largeset_score = scores.max()
    # make it so that 
    # all value before -2 are less thanit and all value after it are greater, so we specify that there is only index -1 greater than it
    second_largest = scores.apply(lambda x: np.partition(x, -2)[-2])
    
    # print("largest\n")
    # print(largeset_score)
    # print("second largest\n")
    # print( second_largest )
    
    delta =  largeset_score-second_largest 
    # print("delta\n")
    # print( delta )
    
    # this gets us the value on the top
    
    top_df["delta"]= delta
    
    # the smaller index the greater
    df_sorted_delta = top_df.sort_values(by='delta', ascending=False)
    
    top_percent_df = df_sorted_delta[:int(len(df_sorted_delta)*percent)]
    
    top_percent_df["target"] =  int(len(df_sorted_delta)*percent) * [score_cols[0]]

    return top_percent_df
    



    

def get_negative_random(data,percent=0.1,contra = "CONTRADICTION"):
    """
    For each entailment pair, generate a negative example by replacing the class
    in the hypothesis with a random different class, and assign the contradict label.
    
    Parameters:
    
       Positve df
       data is the eval output file
        
    Returns:
       Full finetuning dataset
    """

    pos = get_positive_example(data,percent=percent).dropna().reset_index(drop=True)
    sample_space = set(data["textid"].unique())
    for idx in range(len(pos)):
        pos.loc[idx, "target"] = contra 
        event1 =  {pos.loc[idx, "textid"]}
        pos.loc[idx, "textid"] = int(random.sample(list(sample_space - event1), 1)[0])
        mapping_mask =  {0:"World" ,1:"Sports",2:"Business",3:"Sci/Tech"}
        label_type = mapping_mask[int(pos.loc[idx, "textid"])]
        pos.loc[idx, "pair"] = f"This example is {label_type}"
    
    return pos
        






def create_fintune_data(data,originl_train_data,percent,num_label = 4):
    
    pos = get_positive_example(data,percent)
    neg = get_negative_random(data,percent)
    
    org_pos = pos.index 
    mnli_index_pos = org_pos * num_label
    mapping_mask =  {0:"World" ,1:"Sports",2:"Business",3:"Sci/Tech"}
    # Map textid -> label and format each as a string
    
    # apply a mask and create template
    pos_pairs = pos["textid"].map(mapping_mask).apply(lambda x: f"This example is {x}")

    # Combine with neg pairs
    # 
    pair_series = pd.concat([pos_pairs, neg["pair"]], ignore_index=True)


    output = pd.DataFrame({
        
        "textid": range(2 * len(pos)),
        "pair": pair_series,
        "text":  pd.concat([originl_train_data.loc[mnli_index_pos,"text"], originl_train_data.loc[mnli_index_pos,"text"],],ignore_index=True),
        
        "label": pd.concat([pd.Series(len(pos)*["ENTAILMENT"]), neg["target"]],ignore_index=True)
        
        
    })
    
    return output




    
    
