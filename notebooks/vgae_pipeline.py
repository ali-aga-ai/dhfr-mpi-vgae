#!/usr/bin/env python3
"""
VGAE Link Prediction with Feature Comparison
Usage: /usr/bin/python3 vgae_pipeline.py \
    --network features/mpi_network/mpi_Homo_sapiens.pkl \
    --train-features features/mpi_features/feature_df_Homo_sapiens.pkl \
    --inference-features features/mpi_features/feature_df_Homo_sapiens.pkl \
    --epochs 500 \
    --cutoff 0.67 \
    --output-prefix pipeline_test
"""

import argparse
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns
import time
import tensorflow.compat.v1 as tf
import os

# Import VGAE modules
from vgae.optimizer import OptimizerVAE
from vgae.model import GCNModelVAE
from vgae.preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges_seed
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc, precision_recall_curve

tf.disable_v2_behavior()


def load_network(network_path):
    """Load the MPI network from pickle file"""
    print(f"\n{'='*80}")
    print(f"Loading network from: {network_path}")
    print(f"{'='*80}")
    with open(network_path, "rb") as f:
        g = pickle.load(f)
    print(f"Graph loaded: {g}")
    print(f"  - Nodes: {g.number_of_nodes()}")
    print(f"  - Edges: {g.number_of_edges()}")
    return g


def load_features(features_path):
    """Load node features from pickle file"""
    print(f"\nLoading features from: {features_path}")
    node_feats = pd.read_pickle(features_path)
    print(f"Features loaded: {len(node_feats)} nodes")
    print(f"  - Feature dimension: {len(node_feats['features'][0])}")
    return node_feats


def prepare_features(node_feats, g):
    """Convert node features to numpy array and create adjacency matrix"""
    print(f"\n{'='*80}")
    print("Preparing features and adjacency matrix")
    print(f"{'='*80}")
    
    t = []
    for fp in node_feats['features'].tolist():
        temp = [i for i in fp]
        t.append(temp)
    
    features = np.asarray(t)
    print(f"Feature array shape: {features.shape}")
    
    adj = nx.adjacency_matrix(g, nodelist=node_feats['node'].tolist())
    print(f"Adjacency matrix created: {adj.shape}")
    
    x = sp.lil_matrix(features)
    features_tuple = sparse_to_tuple(x)
    features_shape = features_tuple[2]
    
    return features, adj, features_tuple, features_shape


def split_edges(adj):
    """Split edges into train/validation/test sets"""
    print(f"\n{'='*80}")
    print("Splitting edges into train/val/test")
    print(f"{'='*80}")
    
    np.random.seed(0)
    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
        test_edges, test_edges_false = mask_test_edges_seed(adj, test_frac=.1, val_frac=.1, seed=12345)
    
    print(f"Total edges: {int(adj.nnz/2)}")
    print(f"  - Training edges (positive): {len(train_edges)}")
    print(f"  - Training edges (negative): {len(train_edges_false)}")
    print(f"  - Validation edges (positive): {len(val_edges)}")
    print(f"  - Validation edges (negative): {len(val_edges_false)}")
    print(f"  - Test edges (positive): {len(test_edges)}")
    print(f"  - Test edges (negative): {len(test_edges_false)}")
    
    return adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false


def get_roc_score(edges_pos, edges_neg, pos_weight, norm, sess, model, placeholders, feed_dict, adj_orig, emb=None):
    """Calculate ROC AUC score"""
    if emb is None:
        feed_dict.update({placeholders['dropout']: 0})
        emb = sess.run(model.z_mean, feed_dict=feed_dict)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    adj_rec = np.dot(emb, emb.T)
    preds_pos = []
    pos = []
    for e in edges_pos:
        preds_pos.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds_pos, preds_neg])
    labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    fpr, tpr, _ = roc_curve(labels_all, preds_all)
    precision, recall, _ = precision_recall_curve(labels_all, preds_all)
    pr_score = auc(recall, precision)
    ap_score = average_precision_score(labels_all, preds_all)
    
    val_loss = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
        labels=tf.constant(labels_all), logits=tf.constant(preds_all), pos_weight=pos_weight))

    return roc_score, ap_score, fpr, tpr, pr_score, precision, recall, val_loss


def train_model(adj_train, adj_orig, features_tuple, features_shape, train_edges, train_edges_false, 
                val_edges, val_edges_false, test_edges, test_edges_false, epochs=500, 
                learning_rate=0.005, hidden1_dim=32, hidden2_dim=16, dropout=0.1, save_path=None):
    """Train the VGAE model"""
    print(f"\n{'='*80}")
    print("Initializing and training VGAE model")
    print(f"{'='*80}")
    print(f"Hyperparameters:")
    print(f"  - Epochs: {epochs}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Hidden1 dim: {hidden1_dim}")
    print(f"  - Hidden2 dim: {hidden2_dim}")
    print(f"  - Dropout: {dropout}")
    
    num_nodes = adj_train.shape[0]
    num_features = features_shape[1]
    features_nonzero = features_tuple[1].shape[0]
    
    adj_orig_no_diag = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig_no_diag.eliminate_zeros()
    
    adj_norm = preprocess_graph(adj_train)
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)
    
    pos_weight = float(adj_train.shape[0] * adj_train.shape[0] - adj_train.sum()) / adj_train.sum()
    norm = adj_train.shape[0] * adj_train.shape[0] / float((adj_train.shape[0] * adj_train.shape[0] - adj_train.sum()) * 2)
    
    tf.reset_default_graph()
    
    device = '/GPU:1'
    with tf.device(device):
        placeholders = {
            'features': tf.sparse_placeholder(tf.float32),
            'adj': tf.sparse_placeholder(tf.float32),
            'adj_orig': tf.sparse_placeholder(tf.float32),
            'dropout': tf.placeholder_with_default(0., shape=())
        }
        
        model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero, hidden1_dim, hidden2_dim)
        
        opt = OptimizerVAE(
            preds=model.reconstructions,
            labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'], validate_indices=False), [-1]),
            model=model, num_nodes=num_nodes,
            pos_weight=pos_weight,
            norm=norm,
            learning_rate=learning_rate
        )
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    
    # Create saver for saving model weights
    saver = tf.train.Saver()
    
    feed_dict = construct_feed_dict(adj_norm, adj_label, features_tuple, placeholders)
    
    cost_val = []
    acc_val = []
    val_roc_score = []
    train_loss_log = []
    val_loss_log = []
    
    print(f"\n{'='*80}")
    print("Training progress:")
    print(f"{'='*80}")
    
    for epoch in range(epochs):
        t = time.time()
        
        feed_dict.update({placeholders['dropout']: dropout})
        outs = sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)
        
        avg_cost = outs[1]
        avg_accuracy = outs[2]
        
        roc_curr, ap_curr, fpr, tpr, pr_score, precision, recall, train_loss_epoch = get_roc_score(
            train_edges, train_edges_false, pos_weight, norm, sess, model, placeholders, feed_dict, adj_orig_no_diag)
        train_loss_val = sess.run(train_loss_epoch)
        train_loss_log.append(train_loss_val)
        
        roc_curr, ap_curr, fpr, tpr, pr_score, precision, recall, val_loss_epoch = get_roc_score(
            val_edges, val_edges_false, pos_weight, norm, sess, model, placeholders, feed_dict, adj_orig_no_diag)
        val_roc_score.append(roc_curr)
        val_loss_val = sess.run(val_loss_epoch)
        val_loss_log.append(val_loss_val)
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch: {epoch+1:04d} | train_loss: {avg_cost:.5f} | train_acc: {avg_accuracy:.5f} | "
                  f"val_roc: {val_roc_score[-1]:.5f} | val_ap: {ap_curr:.5f} | time: {time.time() - t:.5f}")
    
    print(f"\nTraining completed!")
    
    # Save model weights if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        saver.save(sess, save_path)
        print(f"Model weights saved to: {save_path}")
    
    return sess, model, placeholders, feed_dict, adj_orig_no_diag, pos_weight, norm, train_loss_log, val_loss_log


def load_pretrained_model(model_path, adj_train, adj_orig, features_tuple, features_shape, 
                          hidden1_dim=32, hidden2_dim=16):
    """Load a pretrained VGAE model from checkpoint"""
    print(f"\n{'='*80}")
    print("Loading pretrained VGAE model")
    print(f"{'='*80}")
    print(f"Model path: {model_path}")
    
    num_nodes = adj_train.shape[0]
    num_features = features_shape[1]
    features_nonzero = features_tuple[1].shape[0]
    
    adj_orig_no_diag = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig_no_diag.eliminate_zeros()
    
    adj_norm = preprocess_graph(adj_train)
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)
    
    pos_weight = float(adj_train.shape[0] * adj_train.shape[0] - adj_train.sum()) / adj_train.sum()
    norm = adj_train.shape[0] * adj_train.shape[0] / float((adj_train.shape[0] * adj_train.shape[0] - adj_train.sum()) * 2)
    
    tf.reset_default_graph()
    
    device = '/GPU:1'
    with tf.device(device):
        placeholders = {
            'features': tf.sparse_placeholder(tf.float32),
            'adj': tf.sparse_placeholder(tf.float32),
            'adj_orig': tf.sparse_placeholder(tf.float32),
            'dropout': tf.placeholder_with_default(0., shape=())
        }
        
        model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero, hidden1_dim, hidden2_dim)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    
    # Restore model weights
    saver = tf.train.Saver()
    saver.restore(sess, model_path)
    print(f"✓ Model weights restored successfully!")
    
    feed_dict = construct_feed_dict(adj_norm, adj_label, features_tuple, placeholders)
    
    return sess, model, placeholders, feed_dict, adj_orig_no_diag, pos_weight, norm


def plot_training_loss(train_loss_log, val_loss_log, output_path="training_loss.pdf"):
    """Plot training and validation loss"""
    print(f"\nGenerating training loss plot...")
    
    plt.figure(figsize=(10, 10))
    lw = 2
    
    plt.plot(range(len(train_loss_log)), train_loss_log, lw=lw, color='#0070c0', label='Training')
    plt.plot(range(len(val_loss_log)), val_loss_log, lw=lw, color='#bf0001', label='Validation')
    
    plt.xlim([0, len(train_loss_log)])
    plt.ylim([min(min(train_loss_log), min(val_loss_log)) - 1, max(max(train_loss_log), max(val_loss_log)) + 1])
    
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.legend(loc="upper right", fontsize=20)
    
    plt.savefig(output_path)
    print(f"Training loss plot saved to: {output_path}")
    plt.show()


def vae_pred(test, feed_dict, sess, model, placeholders, adj_orig, emb=None):
    """Make predictions using the VGAE model"""
    if emb is None:
        feed_dict.update({placeholders['dropout']: 0})
        emb = sess.run(model.z_mean, feed_dict=feed_dict)
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    adj_rec = np.dot(emb, emb.T)
    preds_pos = []
    pos = []
    for e in test:
        preds_pos.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])
    
    return np.array(preds_pos), np.array(pos)


def compare_predictions(test_edges, train_preds, inference_preds, cutoff=0.67):
    """Compare predictions between training and inference features"""
    print(f"\n{'='*80}")
    print(f"Comparing predictions (cutoff = {cutoff})")
    print(f"{'='*80}")
    
    train_accepted = []
    train_rejected = []
    
    for edge, score in zip(test_edges, train_preds):
        if score >= cutoff:
            train_accepted.append(edge)
        else:
            train_rejected.append(edge)
    
    inference_accepted = []
    inference_rejected = []
    
    for edge, score in zip(test_edges, inference_preds):
        if score >= cutoff:
            inference_accepted.append(edge)
        else:
            inference_rejected.append(edge)
    
    train_accepted = [tuple(e) for e in train_accepted]
    inference_accepted = [tuple(e) for e in inference_accepted]
    train_rejected = [tuple(e) for e in train_rejected]
    inference_rejected = [tuple(e) for e in inference_rejected]
    
    added = [e for e in inference_accepted if e not in train_accepted]
    removed = [e for e in train_accepted if e not in inference_accepted]
    
    print(f"\nPrediction changes:")
    print(f"  - Edges added (now predicted positive): {len(added)}")
    print(f"  - Edges removed (now predicted negative): {len(removed)}")
    
    intersection = list(set(added) & set(removed))
    print(f"  - Intersection (should be 0): {len(intersection)}")
    
    return added, removed


def calculate_metrics(true_preds, false_preds, cutoff=0.67):
    """Calculate TPR, FPR, Precision, Recall"""
    test_TP = true_preds[np.where(true_preds >= cutoff)]
    test_TN = false_preds[np.where(false_preds < cutoff)]
    test_FN = true_preds[np.where(true_preds < cutoff)]
    test_FP = false_preds[np.where(false_preds >= cutoff)]
    
    test_TPR = len(test_TP) / (len(test_TP) + len(test_FN)) if (len(test_TP) + len(test_FN)) > 0 else 0
    test_FPR = len(test_FP) / (len(test_FP) + len(test_TN)) if (len(test_FP) + len(test_TN)) > 0 else 0
    test_Precision = len(test_TP) / (len(test_TP) + len(test_FN)) if (len(test_TP) + len(test_FN)) > 0 else 0
    test_Recall = len(test_TP) / (len(test_TP) + len(test_FP)) if (len(test_TP) + len(test_FP)) > 0 else 0
    
    return {
        'TP': len(test_TP),
        'TN': len(test_TN),
        'FP': len(test_FP),
        'FN': len(test_FN),
        'TPR': test_TPR,
        'FPR': test_FPR,
        'Precision': test_Precision,
        'Recall': test_Recall
    }


def plot_confusion_matrix(metrics, output_path="confusion_matrix.pdf"):
    """Plot confusion matrix"""
    print(f"\nGenerating confusion matrix...")
    
    plt.figure(figsize=(12, 10))
    data = np.array([[metrics['TP'], metrics['FN']], [metrics['FP'], metrics['TN']]])
    df_cm = pd.DataFrame(data, columns=['Positive', 'Negative'], index=['Positive', 'Negative'])
    sns.heatmap(df_cm, cmap="Blues", annot=True, annot_kws={"size": 20}, fmt='g')
    plt.xlabel('Predicted', fontsize=20)
    plt.ylabel('True', fontsize=20)
    plt.savefig(output_path)
    print(f"✓ Confusion matrix saved to: {output_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='VGAE Link Prediction with Feature Comparison')
    parser.add_argument('--network', type=str, required=True, help='Path to MPI network pickle file')
    parser.add_argument('--train-features', type=str, required=True, help='Path to training node features pickle file')
    parser.add_argument('--inference-features', type=str, required=True, help='Path to inference node features pickle file')
    parser.add_argument('--model-weights', type=str, default=None, help='Path to pretrained model checkpoint (skips training if provided)')
    parser.add_argument('--save-weights', type=str, default=None, help='Path to save model weights after training')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--hidden1', type=int, default=32, help='Hidden layer 1 dimension')
    parser.add_argument('--hidden2', type=int, default=16, help='Hidden layer 2 dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--cutoff', type=float, default=0.67, help='Prediction cutoff threshold')
    parser.add_argument('--output-prefix', type=str, default='vgae', help='Prefix for output files')
    
    args = parser.parse_args()
    
    print(f"\n{'#'*80}")
    print(f"# VGAE Link Prediction Pipeline")
    print(f"{'#'*80}")
    
    # Load data
    g = load_network(args.network)
    train_node_feats = load_features(args.train_features)
    inference_node_feats = load_features(args.inference_features)
    
    # Prepare training features
    train_features, adj, train_features_tuple, features_shape = prepare_features(train_node_feats, g)
    
    # Split edges
    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = split_edges(adj)
    
    # Train or load model
    if args.model_weights:
        # Load pretrained model - skip training
        print(f"\n{'='*80}")
        print("INFERENCE MODE: Loading pretrained model")
        print(f"{'='*80}")
        sess, model, placeholders, feed_dict, adj_orig, pos_weight, norm = load_pretrained_model(
            args.model_weights, adj_train, adj, train_features_tuple, features_shape,
            hidden1_dim=args.hidden1, hidden2_dim=args.hidden2
        )
        train_loss_log = None
        val_loss_log = None
    else:
        # Train model from scratch
        print(f"\n{'='*80}")
        print("TRAINING MODE: Training new model")
        print(f"{'='*80}")
        sess, model, placeholders, feed_dict, adj_orig, pos_weight, norm, train_loss_log, val_loss_log = train_model(
            adj_train, adj, train_features_tuple, features_shape, train_edges, train_edges_false,
            val_edges, val_edges_false, test_edges, test_edges_false,
            epochs=args.epochs, learning_rate=args.lr, hidden1_dim=args.hidden1, 
            hidden2_dim=args.hidden2, dropout=args.dropout, save_path=args.save_weights
        )
        
        # Plot training loss only if we trained
        if train_loss_log and val_loss_log:
            plot_training_loss(train_loss_log, val_loss_log, f"{args.output_prefix}_training_loss.pdf")
    
    # Get predictions with training features
    print(f"\n{'='*80}")
    print("Making predictions with training features")
    print(f"{'='*80}")
    train_true_preds, _ = vae_pred(test_edges, feed_dict, sess, model, placeholders, adj_orig)
    train_false_preds, _ = vae_pred(test_edges_false, feed_dict, sess, model, placeholders, adj_orig)
    
    train_metrics = calculate_metrics(train_true_preds, train_false_preds, args.cutoff)
    print(f"\nTraining features metrics:")
    print(f"  - TPR: {train_metrics['TPR']:.3f}")
    print(f"  - FPR: {train_metrics['FPR']:.3f}")
    print(f"  - Precision: {train_metrics['Precision']:.3f}")
    print(f"  - Recall: {train_metrics['Recall']:.3f}")
    
    plot_confusion_matrix(train_metrics, f"{args.output_prefix}_train_confusion_matrix.pdf")
    
    # Prepare inference features
    inference_features, _, inference_features_tuple, _ = prepare_features(inference_node_feats, g)
    
    # Normalize adjacency for inference
    adj_norm = preprocess_graph(adj_train)
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)
    
    feed_dict_inference = construct_feed_dict(adj_norm, adj_label, inference_features_tuple, placeholders)
    
    # Get predictions with inference features
    print(f"\n{'='*80}")
    print("Making predictions with inference features")
    print(f"{'='*80}")
    inference_true_preds, _ = vae_pred(test_edges, feed_dict_inference, sess, model, placeholders, adj_orig)
    inference_false_preds, _ = vae_pred(test_edges_false, feed_dict_inference, sess, model, placeholders, adj_orig)
    
    inference_metrics = calculate_metrics(inference_true_preds, inference_false_preds, args.cutoff)
    print(f"\nInference features metrics:")
    print(f"  - TPR: {inference_metrics['TPR']:.3f}")
    print(f"  - FPR: {inference_metrics['FPR']:.3f}")
    print(f"  - Precision: {inference_metrics['Precision']:.3f}")
    print(f"  - Recall: {inference_metrics['Recall']:.3f}")
    
    plot_confusion_matrix(inference_metrics, f"{args.output_prefix}_inference_confusion_matrix.pdf")
    
    # Compare predictions
    added, removed = compare_predictions(test_edges, train_true_preds, inference_true_preds, args.cutoff)
    
    print(f"\n{'#'*80}")
    print(f"# Pipeline completed successfully!")
    print(f"{'#'*80}")
    print(f"\nOutput files:")
    if train_loss_log:
        print(f"  - {args.output_prefix}_training_loss.pdf")
    print(f"  - {args.output_prefix}_train_confusion_matrix.pdf")
    print(f"  - {args.output_prefix}_inference_confusion_matrix.pdf")
    
    sess.close()


if __name__ == "__main__":
    main()

    