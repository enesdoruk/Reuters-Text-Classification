

def calculate_metrics(actual_labels, predicted_labels, class_name):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for actual, predicted in zip(actual_labels, predicted_labels):
        if actual == class_name and predicted == class_name:
            true_positives += 1
        elif actual != class_name and predicted == class_name:
            false_positives += 1
        elif actual == class_name and predicted != class_name:
            false_negatives += 1

    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    f_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return recall, precision, f_score


def calculate_metrics_for_all_classes(actual_labels, predicted_labels, classes):
    metrics = {}

    for class_name in classes:
        recall, precision, f_score = calculate_metrics(actual_labels, predicted_labels, class_name)
        metrics[class_name] = {'recall': recall, 'precision': precision, 'f_score': f_score}

    return metrics


def calculate_macro_average(metrics):
    total_recall = 0
    total_precision = 0
    total_f_score = 0
    num_classes = len(metrics)

    for class_name, class_metrics in metrics.items():
        total_recall += class_metrics['recall']
        total_precision += class_metrics['precision']
        total_f_score += class_metrics['f_score']

    macro_recall = total_recall / num_classes
    macro_precision = total_precision / num_classes
    macro_f_score = total_f_score / num_classes
    return macro_recall, macro_precision, macro_f_score
