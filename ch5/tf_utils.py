import tensorflow as tf


def accuracy(y_true, logits):
    """
    Computes Accuracy for TensorFlow Supernet
    """
    return {'enas_acc': reward_accuracy(y_true, logits)}


def reward_accuracy(y_true, logits):
    batch_size = y_true.shape[0]
    y_true = tf.squeeze(y_true)
    y_pred = tf.math.argmax(logits, axis = 1)
    y_pred = tf.cast(y_pred, y_true.dtype)
    equal = tf.cast(y_pred == y_true, tf.int32)
    return tf.math.reduce_sum(equal).numpy() / batch_size


def get_best_model(mutator):
    """
    Sampling mutator for 100 times and getting most probable (best) subnet
    """
    subnet_counter = {}

    for _ in range(100):
        sample = mutator.sample_final()
        subnet = {}

        for k, v in sample.items():
            v_list = v.numpy().tolist()
            params = [i for i, x in enumerate(v_list) if x]
            subnet[k] = tuple(params)

        subnet_signature = tuple(sorted(subnet.items()))
        if subnet_signature not in subnet_counter:
            subnet_counter[subnet_signature] = 0

        # incrementing subnet counter
        subnet_counter[subnet_signature] += 1

    # extracting best subnet signature
    best_subnet_signature = max(subnet_counter, key = subnet_counter.get)

    best_subnet = dict(best_subnet_signature)
    return best_subnet
