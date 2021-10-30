from __future__ import division
from __future__ import print_function
from models import GCN, MLP, MLP_NORM
from utils import *
import sys
import tensorflow as tf
import time
import os
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append('/data/code/gnn-lx/gcn-master')


# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
# 'cora', 'citeseer', 'pubmed'
# 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')
# 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_string('model', 'gcn', 'Model string.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 2000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4,
                   'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 40,
                     'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')


flags.DEFINE_float('alpha', 1, 'weight for frobenius norm on Z')
flags.DEFINE_float('beta', 1, 'weight for frobenius norm on Z-A')
flags.DEFINE_float('gamma', 0.2, 'weight for skip connection in kept')
flags.DEFINE_integer('split_part', 0, 'dataset split part')
flags.DEFINE_integer('dense_layers', 2, 'Number of mlp layers')
flags.DEFINE_integer('norm_layers', 2, 'Number of graphnorm layers')

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(
    FLAGS.dataset, FLAGS.split_part)

classes = y_train.shape[1]
print('features classes: ', classes)

# Some preprocessing
features = preprocess_features(features)
if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
elif FLAGS.model == 'mlp_norm':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP_NORM
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    # helper variable for sparse dropout
    'num_features_nonzero': tf.placeholder(tf.int32),
}

# Create model
model = model_func(placeholders, input_dim=features[2][1], logging=True)

print('model layers', model.layers)

# Initialize session
sess = tf.Session()


# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(
        features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []
acc_val = []

# Train model
for epoch in range(10):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(
        features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy, model.outputs],
                    feed_dict=feed_dict)

    # Validation
    cost, acc, duration = evaluate(
        features, support, y_val, val_mask, placeholders)
    cost_val.append(cost)
    acc_val.append(acc)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(
              outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

    print('outputs: ', outs[3])

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

# Testing
test_cost, test_acc, test_duration = evaluate(
    features, support, y_test, test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

# print(type(test_cost), test_cost)
# print(type(test_acc), test_acc)
# print(type(test_duration), test_duration)
results_dict = {}
results_dict['test_cost'] = float(test_cost)
results_dict['test_acc'] = float(test_acc)
results_dict['test_duration'] = test_duration
with open(os.path.join('runs', f'{FLAGS.dataset}_{int(FLAGS.split_part)}_results.txt'), 'w') as outfile:
    outfile.write(json.dumps(results_dict))
