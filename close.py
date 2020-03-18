import pandas
import numpy as np
from rand_index import rand_score
from scipy.spatial.distance import euclidean


class CLOSE(object):


    '''
    Input:
        data - pandas dataframe with columns 'object_id', 'time', 'cluster_id' containing cluster belongings, features ..
               Note: outliers should have negative labels/cluster_ids, these should be different for different times
        measure (optional) - str or callable for used quality measure
        minPts (optional) - int: used minPts for density-based quality measure
    '''
    def __init__(self, data, measure='mse', minPts=None):
        self._data = data
        self._column_names = data.columns.values
        self._object_column_name = self._column_names[0]
        self._time_column_name = self._column_names[1]
        self._cluster_column_name = self._column_names[2]

        self._minPts = minPts
        self.pos_measures = {'sse': self.calc_sse,  # NOTE: sse is not between 0 and 1
                             'mse': self.calc_mse,
                             'mae': self.calc_mae,
                             'max': self.calc_max_dist,
                             'dbi': self.calc_min_pts}

        if measure in self.pos_measures:
            self.measure = self.pos_measures[measure]
        elif callable(measure):
            self.measure = measure
        else:
            self.measure = self.pos_measures['mse']

    '''     
    Input:
        start_time (optional) - int: time that should be considered as beginning
        end_time (optional) - int: time which should be rated up to 
        return_measures (optional) - boolean: whether additional information such as average stability 
                                              and quality should be returned 
    Output:
        CLOSE score - float with rating of clustering
                      dict with additional information if 'return_measures' is True
    '''
    def rate_clustering(self, start_time=None, end_time=None, return_measures=False):
        cluster_ratings = self.rate_clusters(start_time, end_time)
        gr_clusters = self._data.groupby(self._cluster_column_name)

        score = 0
        avg_quality = 0
        avg_stab = 0

        for cluster in cluster_ratings:
            cluster_objects = gr_clusters.get_group(cluster)[self._object_column_name].unique()
            cluster_time = gr_clusters.get_group(cluster)[self._time_column_name].iloc[0]
            feature_list = self.get_feature_list(cluster_objects, cluster_time)

            measure = self.measure(feature_list)
            avg_quality += measure
            avg_stab += cluster_ratings[cluster]
            score += (cluster_ratings[cluster] * (1 - measure))

        num_clusters = len(cluster_ratings)
        num_timestamps = self.get_num_timestamps(start_time, end_time)

        if num_clusters <= 0:
            print('Clustering has no Clusters!!')
            return 0

        avg_quality /= num_clusters
        print('Average Quality: ', str(avg_quality))
        avg_stab /= num_clusters
        print('Average Stability: ', str(avg_stab))

        factor = (1 / num_clusters) * (1 - (num_timestamps / num_clusters)**2)

        if not return_measures:
            return score * factor

        else:
            return {'close': score * factor,
                    'stability': avg_stab,
                    'quality': avg_quality,
                    'pre-factor': (1 - (num_timestamps / num_clusters) ** 2)}

    '''
    Input:
        start_time (optional) - time that should be considered as beginning
        end_time (optional) - time which should be rated up to 
        id (optional) - int, str, list or None representing the cluster_ids that should be rated. If id is None, 
                        all objects are rated
    
    Output:
        ratings - dict {<cluster_id>: <rating>} with ratings of clusters
    '''
    def rate_clusters(self, start_time=None, end_time=None, id=None):
        ids_to_rate = self.get_ids_to_rate(id, self._cluster_column_name, start_time, end_time)
        ids = ids_to_rate[:]

        # don't rate outliers
        for i in ids_to_rate:
            if int(i) < 0:
                ids.remove(i)

        ratings = self.calc_cluster_rating(ids, start_time)
        return ratings

    '''
    Input:
        ids_to_rate - list of clusters that should be rated
        start_time (optional) - int: time that should be considered as beginning

    Output:
        ratings - dict {<cluster_id>: <rating>} with ratings of clusters
    '''

    def calc_cluster_rating(self, ids_to_rate, start_time=None):
        if start_time is None:
            start_time = np.min(self._data[self._time_column_name].unique())

        ratings = {}
        cluster_compositions = self.obtain_cluster_compositions()
        gr_clusters = self._data.groupby(self._cluster_column_name)

        # iterate over all cluster ids
        for id in ids_to_rate:
            time = gr_clusters.get_group(id)[self._time_column_name].iloc[0]

            # rate the clusters of all timestamps except of the first one
            if time != start_time:
                num_merged_clusters = len(cluster_compositions[id])
                obj_list = gr_clusters.get_group(id)[self._object_column_name].unique().tolist()
                obj_ratings = self.calc_object_rating(cluster_compositions, obj_list, time)
                score = 0
                for obj in obj_ratings:
                    score += obj_ratings[obj]
                try:
                    score /= len(obj_ratings)
                except ZeroDivisionError:
                    print('Cluster ', str(id), ' has no non-outlier members.')

                clusters = list(cluster_compositions[id].keys())
                num_timestamps = len(self._data.loc[self._data[self._cluster_column_name].isin(clusters)]
                                     [self._time_column_name].unique())
                try:
                    div = num_merged_clusters / num_timestamps
                    score /= div
                except ZeroDivisionError:
                    print("<<ZeroDivisionError - Cluster Score>> Cluster ID: ", str(id), "  Merged Clusters: ", str(num_merged_clusters),
                          "  Num Timestamps: ", str(num_timestamps))
                ratings[id] = score

            # clusters of the first timestamp have a stability of 1.0
            else:
                ratings[id] = 1.0
        return ratings

    '''
    Input:
        id (optional) - int, str, list or None representing the data points that should be rated. If id is None, 
                        all objects are rated
        start_time (optional) - time that should be considered as beginning
        end_time (optional) - int representing the timestamp which should be rated up to
        
    Output:
        ratings - dict {<object_id>: <rating>} with ratings of objects
    '''
    def rate_object(self, id=None, start_time=None, end_time=None):
        ids_to_rate = self.get_ids_to_rate(id, self._object_column_name)
        if end_time is None:
            end_time = np.max(self._data[self._time_column_name].unique())
        cluster_compositions = self.obtain_cluster_compositions()
        ratings = self.calc_object_rating(cluster_compositions, ids_to_rate, end_time, start_time)
        return ratings

    '''
    Input:
        cluster_composition - dict {<cluster_id>: {<contained_cluster_id>: <proportion>}} containing the proportions of 
                              clusters (contained_cluster_id) that belong to cluster (cluster_id)
        ids_to_rate - list of data points that should be rated
        end_time - int representing the timestamp which should be rated up to
        start_time (optional) - time that should be considered as beginning
    
    Output:
        ratings - dict {<object_id>: <rating>} with ratings of objects
    '''
    def calc_object_rating(self, cluster_composition, ids_to_rate, end_time, start_time=None):
        ratings = {}
        gr_clusters = self._data.groupby(self._object_column_name)

        # iterate over object ids
        for id in ids_to_rate:
            cur_group = gr_clusters.get_group(id)
            cur_group = cur_group[cur_group[self._time_column_name] <= end_time]

            if start_time is not None:
                cur_group = cur_group[cur_group[self._time_column_name] >= start_time]

            try:
                # id of the cluster of the last considered timestamp
                last_cluster = cur_group[cur_group[self._time_column_name] == end_time][self._cluster_column_name].iloc[0]
            except IndexError:
                print(">>INDEXERROR - LAST CLUSTER<< ID: ", str(id), ", Start Time: ", str(start_time), ", End Time: ", str(end_time))
                continue

            # if object is an outlier for the considered timestamp, it is skipped
            if int(last_cluster) < 0:
                continue

            cluster_ids = cur_group[self._cluster_column_name].unique()

            object_rating = 0
            num_clusters = 0
            for cluster in cluster_ids:
                # Ignore Outliers
                if cluster == last_cluster or int(cluster) < 0:
                    continue
                # Add the proportion of clusters before last timestamp, that merged in last cluster
                else:
                    object_rating += cluster_composition[last_cluster][cluster]
                    num_clusters += 1
            try:
                object_rating /= num_clusters
            except ZeroDivisionError:
                # print(str(id) + " is not assigned to any cluster before t=" + str(end_time))
                x = 1
            ratings[id] = round(object_rating, 3)
        return ratings


    ######## HELPER FUNCTIONS ########

    '''
    Input:
        objects - list of objects_ids that belong to considered cluster
        time - time of cluster that is considered

    Output:
        feature_list - list of lists containing the features of objects in the considered cluster
    '''
    def get_feature_list(self, objects, time):
        feature_list = []
        for obj in objects:
            features = self._data[
                (self._data[self._object_column_name] == obj) & (self._data[self._time_column_name] == time)]
            try:
                features = \
                    features.drop([self._object_column_name, self._cluster_column_name, self._time_column_name],
                                  axis=1).iloc[0].tolist()
            except IndexError:
                print(">>INDEXERROR - FEATURE LIST<< ID: ", str(obj), ", Time: ", str(time))
                continue

            if len(features) <= 0:
                print("No features found for object ", str(obj))
                continue
            feature_list.append(features)
        return np.array(feature_list)

    '''
        Input:
            start_time - int: first timestamp to be considered
            end_time - int: last timestamp to be considered

        Output:
            num_timestamps - int: number of timestamps between start_time and end_time
    '''
    def get_num_timestamps(self, start_time, end_time):
        timestamp_list = self._data[self._time_column_name].unique()
        if start_time is not None:
            timestamp_list = [i for i in timestamp_list if i >= start_time]
        if end_time is not None:
            timestamp_list = [i for i in timestamp_list if i <= end_time]
        num_timestamps = len(timestamp_list)
        return num_timestamps

    '''
    Input:
        id - int, str, list or None representing the data points that should be rated. If id is None, all objects are rated
        id_name - either self._cluster_column_name or self._object_column_name, which ids to extract
    
    Output:
        ids_to_rate - list of ids that should be rated
    '''
    def get_ids_to_rate(self, id, id_name, start_time=None, end_time=None):
        if id is None:
            data = self._data.copy()
            if start_time is not None:
                data = data[data[self._time_column_name] >= start_time]
            if end_time is not None:
                data = data[data[self._time_column_name] <= end_time]
            ids_to_rate = data[id_name].unique().tolist()
        elif isinstance(id, int) or isinstance(id, str):
            ids_to_rate = [id]
        elif isinstance(id, list):
            ids_to_rate = id[:]
        else:
            raise Exception('id has to be int, str, list or None')
        return ids_to_rate

    '''
    Output:
        cluster_compositions - dict of dicts {<cluster_id>: {<cluster_id>: <proportion>}} with cluster compositions
        
        Example:
            {5: {1: 1.0, 2: 0.1, 4: 0.5}} describes that 
                    100% of cluster 1, 10% of cluster 2 and 50% of cluster 4 belong to cluster 5
    '''
    def obtain_cluster_compositions(self):
        cluster_compositions = {}
        cluster_members = self._data.groupby(self._cluster_column_name).count()
        g_clusters = self._data.groupby([self._time_column_name, self._cluster_column_name])

        # iterate over all clusters - 'group' contains the time and cluster_id
        # and 'objects' is the corresponding dataframe
        for group, objects in g_clusters:
            # Ignore outliers
            if int(group[1]) < 0:
                continue

            objects = objects[self._object_column_name].values.tolist()

            # temporal intersection
            # select considered clusters with later timestamps than the current one to check which clusters the
            # current one merged into and count, how many objects of the current cluster are in the considered clusters
            # example of a series from the dataframe: [cluster_id, count] with [2, 10]
            # meaning: 10 objects of the current cluster merged into the cluster with the id 2
            temp_intersection = (self._data.loc[(self._data[self._object_column_name].isin(objects)) &
                            (self._data[self._time_column_name] > group[0])]).groupby(self._cluster_column_name).count()

            # iterate over all clusters which the current cluster has merged into
            # 'cluster' contains the cluster_id
            # and 'con_objects' is the corresponding number of objects of the temporal intersection
            for cluster, num_objects in temp_intersection.iterrows():
                # Ignore outliers
                if int(cluster) < 0:
                    continue

                # for all considered clusters save the proportion of the current cluster that merged into the considered
                # one
                # example: {3: {2: 0.3}, 4: {2: 0.1}}
                # meaning: 30% of (current) cluster 2 merged into (considered) cluster 3 and 10% into (considered) cluster 4
                if cluster not in cluster_compositions:
                    cluster_compositions[cluster] = {}
                cluster_compositions[cluster][group[1]] = round(float(num_objects.values[1]) /
                                                                float(cluster_members.loc[group[1]].values[1]), 3)
            if group[1] not in cluster_compositions:
                cluster_compositions[group[1]] = {}
        return cluster_compositions


    ######## QUALITY MEASURES ########

    '''
        Input:
            feature_list - list of lists containing the features of objects in the considered cluster

        Output:
            sse - float: sum of squared errors to centroid of cluster
    '''
    def calc_sse(self, feature_list):
        centroid = np.average(feature_list, axis=0)
        sse = np.sum(np.power(feature_list - centroid[None, :], 2))
        return sse

    '''
        Input:
            feature_list - list of lists containing the features of objects in the considered cluster

        Output:
            mse - float: mean squared error of cluster
    '''
    def calc_mse(self, feature_list):
        sse = self.calc_sse(feature_list)
        return sse / len(feature_list)

    '''
        Input:
            feature_list - list of lists containing the features of objects in the considered cluster

        Output:
            mae - float: mean average errors to centroid of cluster
    '''
    def calc_mae(self, feature_list):
        centroid = np.average(feature_list, axis=0)
        mae = np.average(np.abs(feature_list - centroid[None, :]))
        return mae

    '''
        Input:
            feature_list - list of lists containing the features of objects in the considered cluster

        Output:
            max_dist - float: maximal distance of cluster member to centroid of cluster
    '''
    def calc_max_dist(self, feature_list):
        max_dist = 0
        for i in range(len(feature_list) - 1):
            for j in range(i + 1, len(feature_list)):
                cur_dist = euclidean(np.array(feature_list[i]), np.array(feature_list[j]))
                if cur_dist > max_dist:
                    max_dist = cur_dist
        max_dist /= 2 ** (1 / 2)
        return max_dist

    '''
        Input:
            feature_list - list of lists containing the features of objects in the considered cluster

        Output:
            avg_dist - float: average distance of cluster members to their minPts neighbor
    '''
    def calc_min_pts(self, feature_list):
        avg_dist = 0
        for i in range(len(feature_list)):
            dist_list = [10] * self._minPts
            for j in range(len(feature_list)):
                if i == j:
                    continue
                cur_dist = euclidean(np.array(feature_list[i]), np.array(feature_list[j]))
                for k in range(len(dist_list)):
                    if cur_dist < dist_list[k]:
                        dist_list.insert(k, cur_dist)
                        dist_list.pop(self._minPts)
            avg_dist += dist_list[self._minPts - 1]
        avg_dist /= len(feature_list)
        return avg_dist

    '''
        Output:
            rand_index - float: average rand_index for all time clusterings
    '''
    def calc_rand_rating(self):
        timestamps = self._data[self._time_column_name].unique()
        timestamps.sort()
        gr_time = self._data.groupby(self._time_column_name)
        rand_index = 0
        counter = 0
        for i in range(len(timestamps) - 1):
            for j in range(i, len(timestamps)):
                cluster_bel = gr_time[[self._object_column_name, self._cluster_column_name]].get_group(timestamps[i])
                bel_next = gr_time[[self._object_column_name, self._cluster_column_name]].get_group(timestamps[j])
                bel_next = bel_next.rename(columns={self._cluster_column_name: 'clu2'})
                cluster_bel = cluster_bel.set_index(self._object_column_name)\
                                .join(bel_next.set_index(self._object_column_name))
                true_labels = cluster_bel[self._cluster_column_name].tolist()
                pred_labels = cluster_bel['clu2'].tolist()
                rand_index += rand_score(true_labels, pred_labels)
                counter += 1
        return rand_index / counter
