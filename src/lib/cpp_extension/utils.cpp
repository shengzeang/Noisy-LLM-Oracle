#include <stdlib.h>
#include <aio.h>
#include <omp.h>
#include <unistd.h>
#include <fcntl.h>
#include <torch/extension.h>
#include <Python.h>
#include <pybind11/pybind11.h>
#include <torch/script.h>
#include <errno.h>
#include <cstring>
#include <inttypes.h>
#include <ATen/ATen.h>
#include <pthread.h>
#include <iostream>
#include <vector>


std::tuple<int64_t, torch::Tensor, int64_t> get_max_reliable_influ_node(torch::Tensor high_score_nodes, torch::Tensor activated_node, torch::Tensor reliability_list,
                                                                    float th, torch::Tensor adj_matrix2, bool is_first) {
    auto high_score_nodes_data = high_score_nodes.data_ptr<int64_t>();
    auto activated_node_data = activated_node.data_ptr<float>();
    auto reliability_list_data = reliability_list.data_ptr<float>();
    auto adj_matrix2_data = adj_matrix2.data_ptr<float>();
    auto num_node = adj_matrix2.size(0);

    auto num_high_score_nodes = high_score_nodes.size(0);
    std::vector<int64_t> activated_node_num_list(num_high_score_nodes);
    #pragma omp parallel for num_threads((int)(atoi(getenv("NUM_THREADS"))*1.00))
    for (int64_t i = 0; i < num_high_score_nodes; ++i) {
        auto node = high_score_nodes_data[i];
        float reliable_score = 0.0;
        if (is_first) {
            reliable_score = 1;
        } else {
            reliable_score = reliability_list_data[node];
        }

        int64_t activated_num = 0;
        for (int64_t j = 0; j < num_node; ++j) {
            int64_t pos = node*num_node + j;
            int64_t cur_val = (int64_t)(adj_matrix2_data[pos]*reliable_score > th);
            activated_num += cur_val*(int)activated_node_data[j];
        }
        activated_node_num_list[i] = activated_num;
    }

    int64_t max_ral_node = 0;
    std::vector<float> max_activated_node_list(num_node);
    int64_t max_activated_num = 0;
    for (int64_t i = 0; i < num_high_score_nodes; ++i) {
        auto cur_count = activated_node_num_list[i];
        if (cur_count > max_activated_num) {
            max_activated_num = cur_count;
            max_ral_node = high_score_nodes_data[i];
        }
    }
    float reliable_score = 0.0;
    if (is_first) {
        reliable_score = 1;
    } else {
        reliable_score = reliability_list_data[max_ral_node];
    }
    // #pragma omp parallel for num_threads((int)(atoi(getenv("NUM_THREADS"))*0.25))
    for (int64_t i = 0; i < num_node; ++i) {
        int64_t cur_val = (int64_t)(adj_matrix2_data[max_ral_node*num_node + i]*reliable_score > th);
        max_activated_node_list[i] = cur_val*(int)activated_node_data[i];
    }

    auto max_activated_node = torch::from_blob(max_activated_node_list.data(), {num_node}, torch::kFloat32).clone();
    return std::make_tuple(max_ral_node, max_activated_node, max_activated_num);
}


void update_reliability_single_node(torch::Tensor reliability_score, int64_t num_node, int64_t node, torch::Tensor adj_vec,
                                    torch::Tensor labels, torch::Tensor idx_used_mask, torch::Tensor similarity_feat,
                                    torch::Tensor labels_sim, torch::Tensor sim_label, int64_t num_class, torch::Tensor visited, float sim_label_cor) {
    auto reliability_score_data = reliability_score.data_ptr<float>();
    auto adj_vec_data = adj_vec.data_ptr<float>();
    auto labels_data = labels.data_ptr<int64_t>();
    auto idx_used_mask_data = idx_used_mask.data_ptr<float>();
    auto similarity_feat_data = similarity_feat.data_ptr<float>();
    auto labels_sim_data = labels_sim.data_ptr<float>();
    auto sim_label_data = sim_label.data_ptr<float>();
    auto visited_data = visited.data_ptr<float>();

    auto node_label = labels_data[node];
    auto relative_influence_vec = std::vector<float>();
    float influence_sum = 0.0;
    // #pragma omp parallel for
    for (int64_t n = 0; n < num_node; ++n) {
        if (adj_vec_data[n] == 0 || n == node) {
            relative_influence_vec.push_back(0.0);
            continue;
        }
        auto n_label = labels_data[n];
        float relative_influence = 0.0;
        if (idx_used_mask_data[n] == 1) {
            relative_influence = labels_sim_data[n_label*num_class + n_label] * similarity_feat_data[n] /
                                (labels_sim_data[n_label*num_class + n_label]*similarity_feat_data[n] + labels_sim_data[n_label*num_class + node_label]*(1-similarity_feat_data[n]));
        } else {
            // average the error rate of its row in the mis-label probability matrix, excluding the diagonal element
            // mean(mat[node_label], except mat[node_label][node_label])
            relative_influence = sim_label_cor * similarity_feat_data[n] /
                                (sim_label_cor*similarity_feat_data[n] + sim_label_data[node_label]*(1-similarity_feat_data[n]));
        }
        relative_influence_vec.push_back(relative_influence);
        influence_sum += relative_influence;
    }

    // #pragma omp parallel for
    for (int64_t n = 0; n < num_node; ++n) {
        relative_influence_vec[n] = relative_influence_vec[n] / (influence_sum + 1e-8);
    }

    // #pragma omp parallel for
    for (int64_t n = 0; n < num_node; ++n) {
        if (adj_vec_data[n] == 0 || n == node) {
            continue;
        }
        if (visited_data[n] == 0) {
            reliability_score_data[n] = 0.0;
        }
        reliability_score_data[n] += reliability_score_data[node] * relative_influence_vec[n];
        visited_data[n] = 1;
    }
}



PYBIND11_MODULE(utils, m) {
    m.def("update_reliability_single_node", &update_reliability_single_node, "optimize efficiency", py::call_guard<py::gil_scoped_release>());
    m.def("get_max_reliable_influ_node", &get_max_reliable_influ_node, "optimize efficiency", py::call_guard<py::gil_scoped_release>());
}
