import numpy as np
import h5py
import os


def load_graph(dataset_name):
    if dataset_name == 'cora' or dataset_name == 'citeseer' or dataset_name == 'facebook':
        dataset_filename = 'datasets/' + dataset_name + "/{}.h5".format(dataset_name)

        # if os.path.exists(dataset_filename):
        if False:
            print("Reading dataset '{}' from file".format(dataset_name))
            dataset_file = h5py.File(dataset_filename, 'r')
            A = dataset_file['A'][:]
            X = dataset_file['X'][:]
            L = dataset_file['L'][:]

            print("A shape: ", A.shape)
            print("X shape: ", X.shape)
            print("L shape: ", L.shape)

            dataset_file.close()
        else:
            print("Bulding dataset '{}'".format(dataset_name))
            if dataset_name != 'facebook':
                cites_txt = open('datasets/{}/{}.cites'.format(dataset_name, dataset_name), 'r')
                content_txt = open('datasets/{}/{}.content'.format(dataset_name, dataset_name), 'r')
            else:
                cites_txt = open('datasets/{}/{}.txt'.format(dataset_name, dataset_name), 'r')

            # convert cites into a dictionary
            edges = []
            all_ids = set()

            for line in cites_txt.readlines():
                if dataset_name == 'facebook':
                    line_ = line.rstrip().split(" ")
                else:
                    line_ = line.rstrip().split("\t")


                cited_paper = line_[0]
                citing_paper = line_[1]

                edges += [(cited_paper, citing_paper)]

                all_ids.add(citing_paper)
                all_ids.add(cited_paper)

            all_ids = list(all_ids)

            edge_ids = []

            # transforms id into indices
            for ii, edge in enumerate(edges):
                idx = all_ids.index(edge[0])
                idy = all_ids.index(edge[1])

                edge_ids += [(idx, idy)]

            # makes matrix A
            dim = len(all_ids)
            A = np.zeros((dim, dim)).astype('int32')

            for edge in edge_ids:
                ii = edge[0]
                jj = edge[1]

                # assert A[ii, jj] != 1, "indices: {}, {}; corresponding to ids {}, {}: ".format(ii, jj, all_ids[ii], all_ids[jj])
                # assert ii != jj, print(edge_ids.index(edge))

                A[ii, jj] = 1
                A[jj, ii] = 1

            print("Number of edges: ", A.sum().sum() // 2)
            print("Number of nodes: ", dim)
            print("Is symmetric: ", np.allclose(A, A.T, rtol=1e-05, atol=1e-08))
            # idx = i + j*n_col
            print("max: ", np.max(A))

            if dataset_name == 'facebook':
                X = np.eye(A.shape[0])
                L = np.eye(A.shape[0])
                return A, X, L

            # convert content into features and labels
            features = {}
            labels = {}

            all_labels = set()

            for line in content_txt.readlines():
                line_ = line.rstrip().split("\t")

                id = line_[0]
                label = line_[-1]
                feature = []
                str_feature = line_[1:-1]
                for x in str_feature:
                    feature += [int(x)]

                labels[id] = label
                features[id] = feature

                all_labels.add(label)

            all_labels = list(all_labels)

            dim_feature = len(line_) - 2
            dim_labeled_nodes = len(features.keys())
            dim_labels = len(all_labels)

            X = np.zeros((dim_labeled_nodes, dim_feature)).astype('int32')
            L = np.zeros((dim_labeled_nodes, dim_labels)).astype('int32')

            print("Number of labeled edges: ", dim_labeled_nodes)
            print("Number of labels: ", dim_labels)
            print("Feature dimension: ", dim_feature)

            fake_idx = []

            for ii, key in enumerate(list(features.keys())):
                feat = np.array(features[key])
                idx = all_ids.index(key)

                # handles missing features case for citeseer

                try:
                    X[idx] = feat
                    label_idx = all_labels.index(labels[key])
                    L[idx, label_idx] = 1
                except IndexError:
                    fake_idx += [idx]

            A = np.delete(A, fake_idx, axis=0)
            A = np.delete(A, fake_idx, axis=1)

            # check for isolated nodes
            a_sum = A.sum(axis=1)
            a_isolated = np.where(a_sum == 0.0)

            A = np.delete(A, a_isolated, axis=0)
            A = np.delete(A, a_isolated, axis=1)

            X = np.delete(X, a_isolated, axis=0)
            X = np.delete(X, a_isolated, axis=1)

            dataset_file = h5py.File(dataset_filename, 'w')
            dataset_file.create_dataset('A', data=A)
            dataset_file.create_dataset('X', data=X)
            dataset_file.create_dataset('L', data=L)
            dataset_file.close()

        return A, X, L
    else:
        print("Dataset '{}' not available".format(dataset_name))
        raise ValueError
