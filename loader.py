import numpy as np
import h5py
import os


def load_graph(dataset_name):
    if dataset_name == 'cora' or dataset_name =='citeseer':
        dataset_filename = 'datasets/' + dataset_name + "/{}.h5".format(dataset_name)

        if os.path.exists(dataset_filename):
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
            cites_txt = open('datasets/cora/cora.cites', 'r')
            content_txt = open('datasets/cora/cora.content', 'r')

            # convert cites into a dictionary
            edges = {}
            all_ids = set()

            for line in cites_txt.readlines():
                line_ = line.rstrip().split("\t")

                cited_paper = line_[0]
                citing_paper = line_[1]

                if citing_paper in edges.keys():
                    edges[citing_paper] += [cited_paper]
                else:
                    edges[citing_paper] = [cited_paper]

                all_ids.add(citing_paper)
                all_ids.add(cited_paper)

            # convert content into features and labels
            features = {}
            labels = {}

            # makes matrix A
            all_ids = list(all_ids)

            dim = len(all_ids)
            A = np.zeros((dim, dim)).astype('int32')

            for ii, id in enumerate(edges.keys()):
                cited_ids = edges[id]

                # cites_ids to indices
                for jj, cited in enumerate(cited_ids):
                    index = all_ids.index(cited)
                    cited_ids[jj] = index

                for cited_idx in cited_ids:
                    A[ii, cited_idx] += 1
            A += A.T

            print("Number of edges: ", A.sum().sum()//2)
            print("Number of nodes: ", dim)
            print("Is symmetric: ", np.allclose(A, A.T, rtol=1e-05, atol=1e-08))


            all_labels = set()

            for line in content_txt.readlines():
                line_ = line.rstrip().split("\t")

                id = line_[0]
                label = line_[-1]
                feature = []

                for x in line_[1:-1]:
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

            for ii, key in enumerate(list(features.keys())):
                feat = np.array(features[key])
                X[ii] = feat

                label_idx = all_labels.index(labels[key])
                L[ii, label_idx] = 1


            dataset_file = h5py.File(dataset_filename, 'w')
            dataset_file.create_dataset('A', data=A)
            dataset_file.create_dataset('X', data=X)
            dataset_file.create_dataset('L', data=L)
            dataset_file.close()

        return A, X, L
