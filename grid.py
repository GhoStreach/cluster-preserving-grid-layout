dict = {}
dict2 = {}
dict0 = {}
dict3 = {}
f_dict = {}
f_dict2 = {}
save_file = True

def body(dataset, grid_width, flag, tt):
    tmp_dict = {}
    full_flag = 0

    import numpy as np
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=4)
    import os
    tsne_result = "../tsne_result"

    # feature_path = os.path.join(processed_data, "{}_full.npz".format(dataset))
    tsne_path = os.path.join(tsne_result, dataset, "position_pred.npz")
    # features = np.load(feature_path, allow_pickle=True)['features']
    tsne_file = np.load(tsne_path, allow_pickle=True)
    embeddings = tsne_file['positions']
    labels = tsne_file['gtlabels']
    plabels = tsne_file['plabels']
    label_names = tsne_file['label_names']

    import collections

    print(collections.Counter(labels).keys())
    print(collections.Counter(plabels).keys())

    labels_num = len(collections.Counter(labels).keys())
    if dataset=="StanfordDog":
        flag = 10
    if dataset=="Isolet":
        flag = 8

    print(label_names)
    print(labels.shape[0])
    if labels.shape[0] < grid_width*grid_width:
        return

    size = labels.shape[0]
    # grid_width = 45
    # flag = 7
    path = "./grid_result"
    if not os.path.exists(path):
        os.mkdir(path)

    path = os.path.join(path, dataset)
    if not os.path.exists(path):
        os.mkdir(path)

    path = os.path.join(path, str(grid_width))
    if not os.path.exists(path):
        os.mkdir(path)

    path = os.path.join(path, str(flag) + "-" + str(grid_width) + "-" + str(tt))
    if save_file and not os.path.exists(path):
        os.mkdir(path)

    samples_N = grid_width * grid_width

    if flag == 'all':
        if (grid_width-1)*(grid_width-1) >= size:
            return
        if samples_N > size:
            samples_N = size
        samples = np.random.choice(size, samples_N, replace=False)
        np.save("samples_{}0.npy".format(dataset), samples)
    else:
        if flag >= labels_num:
            return
        if labels_num <= 4:
            return

        import collections
        print(collections.Counter(labels))

        while True:
            chosen_labels = np.random.choice(labels_num, flag, replace=False)
            # chosen_labels = np.array([1, 4, 7, 9])
            print(chosen_labels)

            idx = np.zeros(size, dtype='bool')
            for lb in chosen_labels:
                idx = (idx | (labels == lb))

            idx2 = np.zeros(size, dtype='bool')
            for lb in chosen_labels:
                idx2 = (idx2 | (plabels == lb))
            idx = idx & idx2

            print(idx)
            idx = (np.arange(size))[idx]
            chosen_size = idx.shape[0]
            print('chosen_size', chosen_size)
            if (grid_width - 1) * (grid_width - 1) < chosen_size:
                break

        if (grid_width-1)*(grid_width-1) >= chosen_size:
            return

        if samples_N > chosen_size:
            samples_N = chosen_size

        samples = np.random.choice(chosen_size, samples_N, replace=False)
        samples = idx[samples]
        np.save("samples_{}0.npy".format(dataset), samples)

    samples = np.load("samples_{}0.npy".format(dataset))
    # s_features = features[samples]
    s_embeddings = embeddings[samples]
    print(s_embeddings.shape)
    s_labels = labels[samples]
    print(s_labels.shape)
    # print(samples.shape[0], s_features.shape)

    import numpy as np

    new_labels = s_labels.copy()

    from gridOptimizer import gridOptimizer
    import os

    # type = "T"
    # # showT = ""
    # showT = "-NoneText"

    # for type in ["O", "S", "T", "E", "C"]:
    # for type in ["C", "E", "EC", "CE", "CplusE"]:
    # for type in ["T", "S", "ST", "TS", "C", "E", "EC", "CE"]:
    # for type in ["C", "E", "EC", "CE"]:
    # for type in ["O", "T", "ST", "EC"]:
    for type in ["O", "ST", "EC"]:
    # for type in ["T", "C", "E", "EC", "CE"]:
        use_type = type
        # for showT in ["", "-NoneText"]:
        for showT in ["-NoneText"]:

            # for op_type in ["base", "compact", "global", "full"]:
            for op_type in ["base", "global", "local", "full", "full2"]:
            # for op_type in ["global", "full"]:
                if (type != "O") and ((op_type == "base") or (op_type == 'global')):
                    continue
                if (type == "O") and ((op_type != "base") and (op_type != 'global')):
                    continue
                # if ((type != "EC") and (type != "ST")) and (op_type == "local"):
                #     continue

                swap_op_order = False
                if op_type == "full2":
                    swap_op_order = True

                m1 = 0
                m2 = 3
                if type == "E":
                    m1 = 0
                    m2 = 3
                if type == "C":
                    m1 = 0
                    m2 = 3
                if type == "CplusE":
                    m1 = 0
                    m2 = 3
                if type == "T":
                    m1 = 5
                    m2 = 0
                if type == "Tswap":
                    m1 = 0
                    m2 = 3
                    use_type = "T"
                if type == "ST":
                    m1 = 5
                    m2 = 0
                if type == "S":
                    m1 = 0
                    m2 = 3
                if type == "TS":
                    m1 = 0
                    m2 = 3
                if type == "O":
                    m1 = 0
                    m2 = 0

                use_local = True
                if (op_type == "base") or (op_type == "global") or (op_type == "compact"):
                    m1 = 0
                    m2 = 0
                    use_local = False

                use_global = True
                if (op_type == "base") or (op_type == "local"):
                    use_global = False

                only_compact = False
                if op_type == "compact":
                    only_compact = True

                file_path = os.path.join(path, type + "-" + op_type + showT + ".svg")
                save_path = os.path.join(path, type + "-" + op_type + ".npz")

                Optimizer = gridOptimizer()
                # print("check done", BASolver.checkConvex(np.array(row_asses_c), np.array(s_labels)))
                # row_asses_m, heat = BASolver.grid3(s_embeddings, s_labels, 'E')
                row_asses_m, t1, t2, new_labels, new_cost, cc = Optimizer.grid(s_embeddings, s_labels, use_type, m1, m2,
                                                                           use_global, use_local, only_compact, swap_op_order=swap_op_order, swap_cnt=2147483647, pred_labels=plabels[samples])

                show_labels = new_labels
                show_labels = np.array(show_labels)

                labels_dict = {}
                dict_num = 0
                for i in range(show_labels.shape[0]):
                    if show_labels[i] not in labels_dict:
                        labels_dict.update({show_labels[i]: dict_num})
                        dict_num += 1
                    show_labels[i] = labels_dict[show_labels[i]]

                tmp = np.full(row_asses_m.shape[0] - show_labels.shape[0], dtype='int', fill_value=-1)
                show_labels = np.concatenate((show_labels, tmp), axis=0)
                print(row_asses_m.shape[0], show_labels.shape[0])

                print(t1, t2)
                showText = True
                if showT == "-NoneText":
                    showText = False
                    s_samples = samples
                    if dataset=="OoDAnimals":
                        s_samples = tsne_file['true_id'][samples]
                    if dataset=="OoDAnimals3":
                        s_samples = tsne_file['true_id'][samples]
                    np.savez(save_path, row_asses=row_asses_m, labels=new_labels, samples=s_samples)
                print("new_cost", new_cost)
                name = "\'" + dataset + "\'-" + str(grid_width) + "-" + str(flag) + "-" + type + "-" + op_type
                print(name, tt)

                # new_cost = np.append(new_cost, [t1 + t2, t2], None)
                if op_type == "base":
                    new_cost = np.append(new_cost, [t2], None)
                else:
                    new_cost = np.append(new_cost, [t1+t2], None)

                cflag = 0
                new_cost[0] = np.exp(-new_cost[0] / grid_width / grid_width)
                new_cost[1] = np.exp(-new_cost[1] / grid_width / grid_width)
                new_cost[2] = 1 - new_cost[2] / grid_width / grid_width
                new_cost[3] = 1 - new_cost[3] / grid_width / grid_width
                new_cost[4] = 1 - new_cost[4] / grid_width / grid_width
                new_cost[5] = 1 - new_cost[5] / grid_width / grid_width
                new_cost[6] = 1 - new_cost[6] / grid_width / grid_width
                new_cost[7] = 1 - new_cost[7] / grid_width / grid_width
                new_cost[8] = 1 - new_cost[8] / grid_width / grid_width

                if (op_type != "base")and(op_type != "global")and(op_type != "full2")and(cc > 0):
                    cflag = 1
                    full_flag = 1

                print(cc)
                tmp_dict.update({name: new_cost.copy()})

                if cflag == 0:
                    if name not in dict:
                        dict.update({name: new_cost.copy()})
                        dict2.update({name: 1})
                    else:
                        dict[name] += new_cost
                        dict2[name] += 1
                else:
                    # exit(0)
                    if name not in dict3:
                        dict3.update({name: 1})
                    else:
                        dict3[name] += 1

                # name0 = name+"-"+str(tt)
                # dict0.update({name0: new_cost.copy()})

                print(show_labels.max())
                if show_labels.max() < 30:
                    Optimizer.show_grid(row_asses_m, show_labels, grid_width, file_path, showText, just_save=True)
                # Optimizer.show_grid(row_asses_m, show_labels, grid_width, "E-full-NoneText.svg", showText)
                # Optimizer.show_grid(row_asses_m, show_labels, grid_width, "test"+name+".png", showText)
                # Optimizer.show_grid(row_asses_m, s_labels, grid_width)

    if full_flag==0:
        for name in tmp_dict:
            if name not in f_dict:
                f_dict.update({name: tmp_dict[name].copy()})
                f_dict2.update({name: 1})
            else:
                f_dict[name] += tmp_dict[name]
                f_dict2[name] += 1


# for dataset in ["MNIST", "STL-10", "CIFAR10", "USPS",  "Weather", "Clothes", "FashionMNIST", "Animals", "Indian food", "Wifi"]:
# for dataset in ["Animals", "Indian food", "Wifi"]:
# for dataset in ["CIFAR10"]:
# for dataset in ["StanfordDog-10"]:
# for dataset in ["FashionMNIST"]:
for dataset in ["MNIST", "Isolet", "CIFAR10", "USPS", "Animals", "Weather", "Wifi", "Indian food", "Clothes", "FashionMNIST", "Texture", "StanfordDog-10"]:
    for grid_width in [20, 30, 40]:
    # for grid_width in [30]:
        for flag in ['all']:
        # for flag in [7]:
            max_tt = 20
            # if grid_width == 20:
            #     max_tt = 50
            for tt in range(max_tt):
                body(dataset, grid_width, flag, tt)
    import numpy as np
    import pickle
    f = open("grid_result/"+dataset+"/data.pkl", 'wb+')
    pickle.dump({"dict": dict, "dict2": dict2}, f, 0)
    f.close()
    # np.savez("grid_result/"+dataset+"/data.npz", dict=dict, dict2=dict2)

# for dataset in ["CIFAR10"]:
#     for grid_width in [30, 40]:
#         for flag in ['all']:
#             max_tt = 20
#             for tt in range(max_tt):
#                 body()
# #
# for dataset in ["CIFAR10"]:
#     for grid_width in [20]:
#         for flag in [3, 5, 7]:
#             max_tt = 20
#             for tt in range(max_tt):
#                 body()

# for key in dict:
#     print(key,"----", dict[key])

# f = open("grid_result/" + "MNIST" + "/data.pkl", 'rb+')
# data = pickle.load(f)
# dict = data['dict']
# dict2 =data['dict2']

for key in dict:
    t = -1
    dict[key] /= dict2[key]
    print(key, "----", "%.3lf"%dict[key][0], "&", "%.3lf"%dict[key][1], "&", "%.3lf"%dict[key][2], "&", "%.3lf"%dict[key][3], "&", "%.3lf"%dict[key][4], "&", "%.3lf"%dict[key][5], "&", "%.3lf"%dict[key][6], "&", "%.3lf"%dict[key][7], "&", "%.3lf"%dict[key][8], "&", "%.3lf"%dict[key][t])

for key in dict:
    if key not in dict2:
        dict2.update({key: 0})
    if key not in dict3:
        dict3.update({key: 0})

    dict3[key] /= (dict3[key]+dict2[key])
    print(key, "----", dict3[key])

import pickle
f = open("all_data.pkl", 'wb+')
pickle.dump({"dict": dict}, f, 0)
f.close()