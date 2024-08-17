from src.lib.cpp_extension.wrapper import *


def update_reliability_single_node(reliability_score, num_node, node, adj_vec, labels, idx_used_mask, 
                                       similarity_feat, labels_sim, sim_label, num_class, visited, sim_label_cor):
    utils.update_reliability_single_node(reliability_score, num_node, node, adj_vec, labels, idx_used_mask,
                                         similarity_feat, labels_sim, sim_label, num_class, visited, sim_label_cor);

def get_max_reliable_influ_node(high_score_nodes, activated_node, reliability_list, th, adj_matrix2, is_first):
    return utils.get_max_reliable_influ_node(high_score_nodes, activated_node, reliability_list, th, adj_matrix2, is_first)
