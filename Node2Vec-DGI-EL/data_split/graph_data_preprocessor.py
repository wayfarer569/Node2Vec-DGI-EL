
import random
import dgl
import logging
import itertools

class GraphDataPreprocessor:
    def __init__(self, g,Ingredient_nodes,Disease_nodes, hide_percentage):
        self.g = g
        self.nodes_type_A = Ingredient_nodes
        self.nodes_type_C = Disease_nodes
        self.hide_percentage = hide_percentage

    def hide_AC_edges(self):

        src_nodes = self.g.edges()[0].tolist()
        dst_nodes = self.g.edges()[1].tolist()

        nodes_type_A_set = set(self.nodes_type_A)
        nodes_type_C_set = set(self.nodes_type_C)

        AC_edges = []
        for u, v in zip(src_nodes, dst_nodes):
            if (u in nodes_type_A_set and v in nodes_type_C_set) or (u in nodes_type_C_set and v in nodes_type_A_set):
                AC_edges.append((min(u, v), max(u, v)))
        AC_edges = list(set(AC_edges))
        print("AC_edges: ",len(AC_edges)*2)
        num_hide = int(len(AC_edges) * self.hide_percentage)
        hidden_edges = set()
        hidden_pairs = random.sample(AC_edges, num_hide)
        for u, v in hidden_pairs:
            hidden_edges.add((u, v))
            hidden_edges.add((v, u))
        print(f"Selected {len(hidden_edges)} hidden edges between A and C nodes.")
        return list(hidden_edges)


    def generate_negative_edges_batch(self,g,hidden_edges_number):
        if not self.nodes_type_A or not self.nodes_type_C:
            raise ValueError("Error: nodes_type_A or nodes_type_C is empty. Cannot proceed.")

        all_possible_edges = list(itertools.product(self.nodes_type_A, self.nodes_type_C))
        print("all_possible_edges",len(all_possible_edges))

        existing_edges = set(zip(g.edges()[0].tolist(), g.edges()[1].tolist()))

        existing = [edge for edge in all_possible_edges if edge in existing_edges]
        non_existing = [edge for edge in all_possible_edges if edge not in existing_edges]

        print("existing_edges_count", len(existing))
        print("non_existing_edges_count", len(non_existing))

        ratio = len(non_existing)/ len(existing)
        print(f"ratio: {ratio:.4f}")

        num_neg_samples = hidden_edges_number/2*ratio
        if len(non_existing) <= num_neg_samples:
            return non_existing
        neg_edges = random.sample(non_existing, int(num_neg_samples))
        return neg_edges

    def prepare_train_and_test_data(self):
        hidden_edges = self.hide_AC_edges()

        all_edges = list(zip(self.g.edges()[0].tolist(), self.g.edges()[1].tolist()))
        train_edges = list(set(all_edges) - set(hidden_edges))
        train_graph = dgl.graph(train_edges, num_nodes=self.g.number_of_nodes())
        print("train_graph",train_graph)

        hidden_edges_number = len(hidden_edges)
        neg_edges = self.generate_negative_edges_batch(self.g, hidden_edges_number)

        test_positive_edges = hidden_edges
        test_negative_edges = neg_edges
        print("Number of positive and negative edges in the test set:")
        print(f"Positive sample edge: {len(test_positive_edges)}")
        print(f"Negative sample edge: {len(test_negative_edges)}")
        return train_graph, test_positive_edges, test_negative_edges


def load_and_group_nodes_by_type(node_info):
    node_type_map = {}
    for node_id, (_, node_type) in node_info.items():
        try:
            node_type_map[node_type].append(node_id)
        except KeyError:
            node_type_map[node_type] = [node_id]
            print(f"Node type '{node_type}' does not exist, added.")
    for node_type, node_ids in node_type_map.items():
        count = len(node_ids)
        print(f"Node type '{node_type}' contains {count} nodes.")
        print(f"Node ID List: {node_ids}")

    return node_type_map
