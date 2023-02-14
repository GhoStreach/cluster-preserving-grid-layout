import numpy as np
np.set_printoptions(suppress=True)
np.set_printoptions(precision=5)
import os


dict = {}
save_file = True

# for dataset in ["MNIST", "STL-10", "CIFAR10", "USPS"]:
# for dataset in ["CIFAR10", "USPS"]:
# for dataset in ["Wifi"]:
for dataset in ["Cats-vs-dogs", "Weather", "Indian food", "Wifi"]:
    for grid_width in [20]:
    # for grid_width in [30]:
    #     for flag in [3, 5, 7]:
        for flag in ['all']:
            max_tt = 20
            for tt in range(max_tt):
                processed_data = "./processed_data"
                tsne_result = "./tsne_result"

                # feature_path = os.path.join(processed_data, "{}_full.npz".format(dataset))
                tsne_path = os.path.join(tsne_result, dataset, "position.npz")
                # features = np.load(feature_path, allow_pickle=True)['features']
                tsne_file = np.load(tsne_path, allow_pickle=True)
                embeddings = tsne_file['positions']
                labels = tsne_file['labels']
                label_names = tsne_file['label_names']
                print(label_names)
                print(labels.shape[0])
                # print(features.shape)

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

                if flag == 'all':
                    samples = np.random.choice(size, grid_width*grid_width, replace=False)
                    np.save("samples_{}3.npy".format(dataset), samples)
                else:
                    chosen_labels = np.random.choice(label_names.shape[0], flag, replace=False)
                    print(chosen_labels)
                    idx = np.zeros(size, dtype='bool')
                    for lb in chosen_labels:
                        idx = (idx|(labels == lb))
                    print(idx)
                    idx = (np.arange(size))[idx]
                    chosen_size = idx.shape[0]
                    samples = np.random.choice(chosen_size, grid_width*grid_width, replace=False)
                    samples = idx[samples]
                    np.save("samples_{}3.npy".format(dataset), samples)

                samples = np.load("samples_{}3.npy".format(dataset))
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
                for type in ["T"]:
                    # for showT in ["", "-NoneText"]:
                    for showT in ["-NoneText"]:
                        # alpha_list = [(0, 0), (1, 0), (0, 1), (0.4, 0.1), (0.4, 0.3), (0.4, 0.5)]
                        alpha_list = [(0.33, 0.33)]
                        alpha_list2 = []

                        # alpha_list = []
                        # alpha_list2 = [(0, 0), (1, 0), (0, 1)]
                        # for alpha2 in [1/8, 1/4, 1/2, 1, 2, 4, 8]:
                        #     for beta2 in [1/8, 1/4, 1/2, 1, 2, 4, 8]:
                        #         alpha = alpha2/(1+alpha2+beta2)
                        #         beta = beta2/(1+alpha2+beta2)
                        #         if (alpha2/beta2>8)or(beta2/alpha2>8):
                        #             continue
                        #         alpha_list.append((alpha, beta))
                        # length = len(alpha_list)
                        # for i in range(length):
                        #     for j in range(length):
                        #         if i >= j:
                        #             continue
                        #         if (alpha_list[i][0]>alpha_list[j][0])or((alpha_list[i][0]==alpha_list[j][0])and(alpha_list[i][1]>alpha_list[j][1])):
                        #             alpha_list[i], alpha_list[j] = alpha_list[j], alpha_list[i]
                        #
                        # for alpha2 in [1]:
                        #     for beta2 in [1/8, 1/4, 1/2, 1, 2, 4, 8]:
                        #         alpha = alpha2/(alpha2+beta2)
                        #         beta = beta2/(alpha2+beta2)
                        #         if (alpha2/beta2>8)or(beta2/alpha2>8):
                        #             continue
                        #         alpha_list2.append((alpha, beta))
                        #         alpha_list2.append((0, alpha))
                        #         alpha_list2.append((beta, 0))
                        #
                        # length2 = len(alpha_list2)
                        # for i in range(length2):
                        #     for j in range(length2):
                        #         if i >= j:
                        #             continue
                        #         if (alpha_list2[i][0]>alpha_list2[j][0])or((alpha_list2[i][0]==alpha_list2[j][0])and(alpha_list2[i][1]>alpha_list2[j][1])):
                        #             alpha_list2[i], alpha_list2[j] = alpha_list2[j], alpha_list2[i]

                        for op_type in ["base", "compact", "global", "local", "full"]:
                        # for op_type in ["local"]:
                                m1 = 0
                                m2 = 5
                                # if alpha+beta == 0:
                                #     m1 = 1
                                #     m2 = 0
                                # if type == "S" or type == "C" or type == "TB":
                                #     m1 = 0
                                #     m2 = 5
                                if type == "E":
                                    m1 = 0
                                    m2 = 3
                                if type == "T":
                                    m1 = 5
                                    m2 = 0
                                #     if alpha+beta > 0:
                                #         m2 = 1
                                if type == "O":
                                    m1 = 0
                                    m2 = 0

                                if (op_type == "base") or (op_type == "global") or (op_type == "compact"):
                                    m1 = 0
                                    m2 = 0

                                use_global = True
                                if (op_type == "base") or (op_type == "local"):
                                    use_global = False

                                only_compact = False
                                if op_type == "compact":
                                    only_compact = True


                                file_path = os.path.join(path, type + "-" + op_type + showT + ".png")
                                save_path = os.path.join(path, type + "-" + op_type + ".npz")

                                Optimizer = gridOptimizer()
                                # print("check done", BASolver.checkConvex(np.array(row_asses_c), np.array(s_labels)))
                                # row_asses_m, heat = BASolver.grid3(s_embeddings, s_labels, 'E')
                                row_asses_m, t1, t2, new_labels, new_cost = Optimizer.grid(s_embeddings, s_labels, type, m1, m2, use_global, only_compact)

                                show_labels = s_labels
                                show_labels = np.array(show_labels)
                                tmp = np.full(row_asses_m.shape[0] - show_labels.shape[0], dtype='int', fill_value=flag)
                                show_labels = np.concatenate((show_labels, tmp), axis=0)

                                print(t1, t2)
                                showText = True
                                if showT == "-NoneText":
                                    showText = False
                                    np.savez(save_path, row_asses=row_asses_m, labels=new_labels)
                                print("new_cost", new_cost)
                                name = "\'" + dataset + "\'-" + str(grid_width) + "-" + str(flag) + "-" + type + "-" + op_type
                                print(name, tt)
                                new_cost = np.append(new_cost, [t1, t2], None)

                                new_cost[0] = np.exp(-new_cost[0]/grid_width/grid_width)
                                new_cost[1] = np.exp(-new_cost[1]/grid_width/grid_width)
                                new_cost[2] = 1-new_cost[2]/grid_width/grid_width

                                if name not in dict:
                                    dict.update({name: new_cost/max_tt})
                                else:
                                    dict[name] += new_cost/max_tt

                                Optimizer.show_grid(row_asses_m, show_labels, grid_width, file_path, showText, just_save=True)
                                # Optimizer.show_grid(row_asses_m, show_labels, grid_width, "test13.png", showText)
                                # Optimizer.show_grid(row_asses_m, show_labels, grid_width, "test"+name+".png", showText)
                                # Optimizer.show_grid(row_asses_m, s_labels, grid_width)

# for key in dict:
#     print(key,"----", dict[key])


for key in dict:
    print(key, "----", dict[key])