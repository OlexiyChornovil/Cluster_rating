import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import itertools
import matplotlib.style as style
import os
import operator
from fastdtw import fastdtw

def clustering_kmeans(df, n, count_offset):
    clustering = KMeans(n_clusters=n).fit(df)
    for i in range(0, len(clustering.labels_)):
        if clustering.labels_[i] > -1:
            clustering.labels_[i] += count_offset
    count_offset = np.max(clustering.labels_) + 1
    labels_df = pd.DataFrame({'cluster': clustering.labels_}, df.index)
    return df.join(labels_df) , count_offset

def df_separate_on_column(df, column_name):
    #returns list of dfs with same value in said column
    lst = []
    for item in list(df[column_name].unique()):
        lst.append(df[df[column_name] == item])
    return lst

def df_drop_col(df, col_names):
    if isinstance(col_names, list):
        for item in col_names:
            if item in list(df.columns):
                df = df.drop(item, axis=1)
        return df
    else:
        if col_names in list(df.columns):
            df = df.drop(col_names, axis=1)
        return df
    
def dfs_join(df1, df2, col_names_to_drop=None):
    df1 = df_drop_col(df1, col_names_to_drop)
    df2 = df_drop_col(df2, col_names_to_drop)
    return df1.append(df2)

def keep_columns(df, col_names):
    drop = list(set(list(df)) - set(col_names))
    return df.drop(drop, axis = 1)

def find_same_rows_by_value(df1, df2, col_names_to_keep):
    df1_n = keep_columns(df1, col_names_to_keep)
    df2_n = keep_columns(df2, col_names_to_keep)
    return df1_n.merge(df2_n)

def rate_by_amount(df_from_prev_gen, df, cols_intersected):
    scores = []
    for cluster in df_separate_on_column(df, "cluster"):
        score = len(find_same_rows_by_value(df_from_prev_gen, cluster, cols_intersected))/len(df_from_prev_gen)
        scores.append((score, df_from_prev_gen, cluster))
    return scores

def select_best_rating(ratings_list):
    if len(ratings_list) == 0:
        return None
    return max(ratings_list,key=operator.itemgetter(0))

def merge_clusters_completely(df_prev_gen, df_new, merge_labels):
    df = df_prev_gen.merge(df_new, on=merge_labels, how="outer")
    label = df["cluster_y"].unique()[0]
    df = df.drop("cluster_x", axis=1).drop("cluster_y", axis=1)
    df["cluster"] = label
    return df

def remove_df_from_list(lst, df):
    for i in range(len(lst)):
        if df.equals(lst[i]):
            del lst[i]
            return
    print("no same element found")

def remove_rows_contained_in_df(df, df_to_remove, column_to_ignore):
    return pd.concat([df, df_to_remove.drop(column_to_ignore, axis=1), 
               df_to_remove.drop(column_to_ignore, axis=1)]).drop_duplicates(keep=False)

def rebase_column_in_df_list(list_df, col_name):
    for i in range(len(list_df)):
        list_df[i][col_name] = i
    df = pd.concat(list_df)
    return df.reset_index(drop=True)


def dtw_rating(df_from_prev_gen, df, cols_intersected):
    # https://cs.stackexchange.com/questions/53250/normalized-measure-from-dynamic-time-warping
    # problem with negative results on dtw
    df1 = df_from_prev_gen[cols_intersected].to_numpy()
    df2 = df[cols_intersected].to_numpy()
    dist, path = fastdtw(df1, df2)
    
    m1 = len(df1)*np.amax(df1)
    m2 = len(df2)*np.amax(df2)
    m = max(m1, m2)
    
    rating = (m-dist)/m
    return rating

def rate_by_dtw(df_from_prev_gen, df, cols_intersected):
    scores = []
    for cluster in df_separate_on_column(df, "cluster"):
        score = dtw_rating(df_from_prev_gen, cluster, cols_intersected)
        #print(score)
        scores.append((score, df_from_prev_gen, cluster))
    return scores

def kmeans(data, n, score, clustering_method, rating_strat, merge_strat):
    to_do = pd.DataFrame()
    running = []
    run_in_next_iter = []
    finished = []
    first_time = True
    count_offset = 0
    persistent_labels = list(data.columns)
    for timestamp_data in df_separate_on_column(data, "time"):
        if first_time:
            first_time = False
            results, count_offset = clustering_method(timestamp_data, n, count_offset)
            running.append(df_separate_on_column(results, "cluster"))
            running = [item for sublist in running for item in sublist]
        
        else:
            still_running = True
            while still_running:
                ratings = []
                for current in running:
                    tmp = dfs_join(timestamp_data, current, ["cluster"])
                    results, count_offset = clustering_method(tmp, n, count_offset)
                    ratings.append(rating_strat(current, results, persistent_labels))
                ratings = [item for sublist in ratings for item in sublist] #flatten list
                best = select_best_rating(ratings)
                
                if (best is not None) and (best[0] >= score):
                        run_in_next_iter.append(merge_strat(best[1], best[2], persistent_labels))
                        remove_df_from_list(running, best[1])
                        timestamp_data = remove_rows_contained_in_df(timestamp_data, best[2], "cluster")
                else:
                    results, count_offset = clustering_method(timestamp_data, n, count_offset)
                    
                    finished.append(running)
                    running = run_in_next_iter
                    for item in df_separate_on_column(results, "cluster"):
                        running.append(item)
                    run_in_next_iter = []
                    still_running = False
    finished.append(running)
    final_result = [item for sublist in finished for item in sublist] #flatten list
    return rebase_column_in_df_list(final_result, "cluster")                

def return_clustering_1(data, n, score):
    return kmeans(data, n, score, clustering_kmeans, rate_by_amount, merge_clusters_completely)

def return_clustering_2(data, n, score):
    return kmeans(data, n, score, clustering_kmeans, rate_by_dtw, merge_clusters_completely)
