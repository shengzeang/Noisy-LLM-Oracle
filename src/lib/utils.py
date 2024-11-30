from src.lib.cpp_extension.wrapper import *


def update_reliability_single_node(reliability_score, num_node, node, adj_vec, labels, idx_used_mask, 
                                       similarity_feat, labels_sim, sim_label, num_class, visited):
    utils.update_reliability_single_node(reliability_score, num_node, node, adj_vec, labels, idx_used_mask,
                                         similarity_feat, labels_sim, sim_label, num_class, visited);

def get_max_reliable_influ_node(high_score_nodes, activated_node, reliability_list, th, adj_rowptr, adj_col, adj_value, is_first):
    return utils.get_max_reliable_influ_node(high_score_nodes, activated_node, reliability_list, th, adj_rowptr, adj_col, adj_value, is_first)
