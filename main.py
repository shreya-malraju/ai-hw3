import sys
import numpy as np
from collections import Counter

class TreeNode:
    def __init__(self, feature_index=None, threshold=None, result=None, left=None, right=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.result = result
        self.left = left
        self.right = right

def load_data(filename):
    data = np.loadtxt(filename)
    features = data[:, :-1]
    labels = data[:, -1]
    return features, labels

def calculate_entropy(labels):
    class_counts = Counter(labels)
    total_samples = len(labels)
    entropy = -sum((count / total_samples) * np.log2(count / total_samples) for count in class_counts.values())
    return entropy

def calculate_information_gain(data, labels, feature_index, threshold):
    left_mask = data[:, feature_index] <= threshold
    right_mask = ~left_mask

    left_entropy = calculate_entropy(labels[left_mask])
    right_entropy = calculate_entropy(labels[right_mask])

    total_samples = len(labels)
    total_entropy = calculate_entropy(labels)

    gain = total_entropy - ((sum(left_mask) / total_samples) * left_entropy + (sum(right_mask) / total_samples) * right_entropy)
    return gain

def find_best_split(data, labels, option):
    num_features = len(data[0])
    best_gain = -1
    best_feature = None
    best_threshold = None

    if option == "randomized":
        feature_index = np.random.randint(num_features)
        unique_values = np.unique(data[:, feature_index])
        if len(unique_values) < 2:
            return None, None

        threshold = np.random.choice((unique_values[:-1] + unique_values[1:]) / 2)
        return feature_index, threshold

    for feature_index in range(num_features):
        unique_values = np.unique(data[:, feature_index])
        if len(unique_values) < 2:
            continue

        thresholds = (unique_values[:-1] + unique_values[1:]) / 2

        for threshold in thresholds:
            gain = calculate_information_gain(data, labels, feature_index, threshold)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_index
                best_threshold = threshold

    return best_feature, best_threshold


def build_tree(data, labels, option, max_depth=float('inf')):
    if max_depth == 0 or len(set(labels)) == 1:
        return TreeNode(result=Counter(labels).most_common(1)[0][0])

    feature_index, threshold = find_best_split(data, labels, option)

    if feature_index is None:
        return TreeNode(result=Counter(labels).most_common(1)[0][0])

    left_mask = data[:, feature_index] <= threshold
    right_mask = ~left_mask

    left_subtree = build_tree(data[left_mask], labels[left_mask], option, max_depth - 1)
    right_subtree = build_tree(data[right_mask], labels[right_mask], option, max_depth - 1)

    return TreeNode(feature_index=feature_index, threshold=threshold, left=left_subtree, right=right_subtree)

def predict(tree, sample):
    if tree.result is not None:
        return tree.result

    if sample[tree.feature_index] <= tree.threshold:
        return predict(tree.left, sample)
    else:
        return predict(tree.right, sample)

def print_results(predictions, true_labels):
    for i, (predicted_class, true_class) in enumerate(zip(predictions, true_labels)):
        accuracy = 1 if predicted_class == true_class else 0
        print(f"Object Index = {i}, Result = {predicted_class}, True Class = {true_class}, Accuracy = {accuracy}")

    overall_accuracy = np.mean(predictions == true_labels)
    print(f"Classification Accuracy = {overall_accuracy}")

def main():
    if len(sys.argv) != 4:
        print("Usage: dtree training_file test_file option")
        sys.exit(1)

    train_file = sys.argv[1]
    test_file = sys.argv[2]
    option = sys.argv[3]

    train_data, train_labels = load_data(train_file)
    test_data, test_labels = load_data(test_file)

    if option.startswith("forest"):
        num_trees = int(option[6:])
        forest = [build_tree(train_data, train_labels, "randomized") for _ in range(num_trees)]
        predictions = [Counter([predict(tree, sample) for tree in forest]).most_common(1)[0][0] for sample in test_data]
    else:
        tree = build_tree(train_data, train_labels, option)
        predictions = [predict(tree, sample) for sample in test_data]

    print_results(predictions, test_labels)


if __name__ == "__main__":
    main()