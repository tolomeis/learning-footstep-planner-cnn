import os
import pickle
from argparse import ArgumentParser

import tensorflow as tf
from tensorflow_addons.optimizers import AdamW

from utils import experiment, metrics
import dataset.foothold_generic as ds
import model.mrf as mrf

import matplotlib.pyplot as plt

def main(args):
    tf.keras.backend.clear_session()

    (train_ds, train_size), (val_ds, val_size), (test_ds, test_size), metadata = \
        ds.load_npz(args.dataset_path)

    train_ds = train_ds \
        .batch(args.batch_size) \
        .prefetch(2)

    val_ds = val_ds \
        .batch(args.batch_size) \
        .prefetch(2)

    test_ds = test_ds \
        .batch(args.batch_size) \
        .prefetch(2)

    # 2. Model
    model = mrf.MRF_generic(metadata['number_of_classes'], 1, args.field_size, args.momentum, args.dropout)
    model.build_seq((None,40,40,1),(None,1),(None,5,5,130))
    dummy = [tf.keras.Input(shape=[40,40,1]), tf.keras.Input(shape=[1]), tf.keras.Input(shape=[1])]
    out = model(dummy)
    #x = [tf.keras.Input(shape=(40,40,1)), tf.keras.Input(shape=(1)), tf.keras.Input(shape=(1))]

    #model = model2.get_model(x)
    print(model.summary())
    
    loss_weights = tf.convert_to_tensor(metadata['classes_weights_with_workspace'], tf.float32)
    stat_weights = tf.convert_to_tensor(metadata['classes_weights_stats_with_workspace'], tf.float32)
    colors = ds.get_colors(metadata['number_of_classes'])

    # 3. Optimization
    eta = tf.Variable(args.eta)
    eta_f = tf.keras.optimizers.schedules.ExponentialDecay(
        args.eta,
        decay_steps=int(float(train_size) * args.eta_mul / args.batch_size),
        decay_rate=args.train_beta)

    wd = tf.Variable(args.weight_decay)
    wd_f = tf.keras.experimental.CosineDecay(wd, args.wd_decay_steps, alpha=1e-03)
    optimizer = AdamW(wd, eta)

    # 4. Restore, Log & Save
    itr = iter(train_ds)
    #train_step, val_step = tf.constant(0, tf.int64), tf.constant(0, tf.int64)
    train_step = tf.Variable(0, dtype=tf.int64)
    val_step = tf.Variable(0,   dtype=tf.int64)
    best_iou = tf.Variable(0.0, dtype=tf.float64)
    epoch_chkp = tf.Variable(0, dtype=tf.int64)
    early_stop_counter = tf.Variable(0, dtype=tf.int64)

    experiment_handler = experiment.ExperimentHandler(
        args.working_path, args.out_name,
        step=epoch_chkp,
        model=model,
        optimizer=optimizer,
        iterator=itr,
        eta=eta,
        wd=wd,
        train_step=train_step,
        val_step=val_step,
        best_iou=best_iou,
        estop_counter=early_stop_counter
    )

    experiment_handler.restore_last()
        
    # 5. Run everything
    eta.assign(eta_f(train_step.numpy()))
    wd.assign(wd_f(train_step.numpy()))
    epoch = epoch_chkp.numpy()

    cm = metrics.ConfusionMatrix(metadata['number_of_classes'], weights=stat_weights)
    best_train_cm = None
    best_val_cm = None

    @tf.function
    def segmentation_loss(labels, logits, weights):
        C = tf.shape(logits)[-1]

        labels = tf.reshape(labels, [-1])
        logits = tf.reshape(logits, [-1, C])

        logits = tf.reshape(logits, [-1, C])
        weights = tf.nn.embedding_lookup(weights, labels)
        weights *= tf.cast(tf.abs(tf.argmax(logits, -1, tf.int32) - labels), tf.float32) + 1.0

        l = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
        l = tf.reduce_sum(l * weights) / tf.reduce_sum(weights)

        return l

    @tf.function
    def query(data, labels, training):
        output = model(data, training=training)
        predictions = tf.argmax(output, -1, output_type=tf.int32)
        loss = segmentation_loss(labels, output, loss_weights)
        return loss, predictions, output

    @tf.function
    def train_fn(data, labels):
        with tf.GradientTape() as tape:
            loss, predictions, output = query(data, labels, False)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return loss, predictions, output

    
    #tf.profiler.experimental.server.start(6009)

    while True:

        # 5.1. Training Loop
        experiment_handler.log_training()
        cm.reset_states()
        first = True
        for  i, map, cp, cr, labels  in experiment.ds_tqdm('Train', train_ds, epoch, args.batch_size, train_size):
            # if first:
            #     dummy = {"input_1": tf.keras.Input(shape=(40,40,1)), "input_2": tf.keras.Input(shape=(2))}
            #     d = model(dummy, training=False)
            #     first = False
            #     print("MODEL BUILDED")
            #     print(model.summary())
            data = [map, cp, cr]
            loss, predictions, output = train_fn(data, labels)
            cm(labels, predictions)

            # 5.1.4 Save logs for particular interval
            if train_step.numpy() % args.log_interval == 0:
                log_scalars(loss, train_step.numpy(), eta, wd)

            if train_step.numpy() % args.log_images_interval == 0:
                log_images(map, predictions, labels, model.k_internal, colors, train_step.numpy())

            # 5.1.5 Update meta variables
            eta.assign(eta_f(train_step.numpy()))
            wd.assign(wd_f(train_step.numpy()))
            #train_step += 1
            train_step.assign_add(1)

        # 5.1.6 Take statistics over epoch
        train_cm = cm.to_array()
        log_epoch_scalars(cm, epoch)

        # 5.2. Validation Loop
        experiment_handler.flush()
        experiment_handler.log_validation()
        cm.reset_states()
        for i, map, cp, cr, labels in experiment.ds_tqdm('Validation', val_ds, epoch, args.batch_size, val_size):
            # never pass python primitives, arguments should be only tensorflow objects
            data = [map, cp, cr]
            loss, predictions, output = query(data, labels, False)

            cm(labels, predictions)

            # 5.2.3 Save logs for particular interval
            if val_step.numpy() % args.log_interval == 0:
                log_scalars(loss, val_step.numpy())

            if val_step.numpy() % args.log_images_interval == 0:
                log_images(map, predictions, labels, model.k_internal, colors, val_step.numpy())

            # 5.2.4 Update meta variables
            #val_step += 1
            val_step.assign_add(1)

        # 5.2.5 Take statistics over epoch
        val_cm = cm.to_array()
        log_epoch_scalars(cm, epoch)
        experiment_handler.flush()

        # 5.3 Save last and best
        epoch_iou = cm.mean_iou()
        if epoch_iou > best_iou.numpy():
            experiment_handler.save_best()
            best_iou.assign(epoch_iou)
            early_stop_counter.assign(0)
            print("Best IOU: {0} on epoch {1}".format(best_iou.numpy(), epoch))
            best_train_cm = train_cm
            best_val_cm = val_cm
        elif epoch > args.early_stop_epoch:
            early_stop_counter.assign_add(1)

        if early_stop_counter.numpy() >= args.early_stop_threshold or epoch >= 310:
            print("Early stopping triggered on epoch: {0}".format(epoch))
            break

        epoch += 1
        # Save previous epoch in checkpoint
        experiment_handler.save_last()
        # epoch stored in checkpoint is the epoch to be made
    cm.reset_states()
    for i, map, cp, cr, labels in experiment.ds_tqdm('Test', test_ds, 0, args.batch_size, test_size):
        data = [map, cp, cr]
        loss, predictions, output = query(data, labels, False)
        cm(labels, predictions)

    test_cm = cm.to_array()

    model.save_weights(os.path.join(args.working_path, args.out_name, "last_model"))

    # after all take best model
    experiment_handler.restore_best()
    model.save_weights(os.path.join(args.working_path, args.out_name, "best_model"))

    summary = {
        'model_name': args.out_name,
        'last_checkpoints': experiment_handler.checkpoint_manager_last.latest_checkpoint,
        'best_checkpoints': experiment_handler.checkpoint_manager_best.latest_checkpoint,
        'val_cm': best_val_cm,
        'test_cm': test_cm,
        'train_cm': best_train_cm,
        'metadata': metadata
    }
    print("saved")
    base_path = os.path.join(args.working_path, args.out_name)
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    with open(os.path.join(base_path, 'summary.pickle'), 'wb') as fp:
        pickle.dump(summary, fp)


def log_scalars(loss, step, eta=None, weight_decay=None):
    tf.summary.scalar('metrics/model_loss', loss, step=step)

    if eta is not None:
        tf.summary.scalar('info/eta', eta, step=step)
    if weight_decay is not None:
        tf.summary.scalar('info/weight_decay', weight_decay, step=step)


def log_images(input, prediction, labels, kernel, colors, step):
    tf.summary.image('images/input', input, max_outputs=1, step=step)
    tf.summary.image('images/labels', tf.nn.embedding_lookup(colors, tf.squeeze(labels, -1)), max_outputs=1, step=step)
    tf.summary.image('images/predicted', tf.nn.embedding_lookup(colors, prediction), max_outputs=1, step=step)

    k_shape = tf.shape(kernel).numpy()

    for i in range(k_shape[0]):
        for j in range(k_shape[1]):
            k = kernel[i][j][tf.newaxis, :, :, tf.newaxis]
            k_max = tf.reduce_max(k)
            k_min = tf.reduce_min(k)
            k = (k - k_min) / (k_max - k_min + 1e-4)
            tf.summary.image('kernel/{0}_{1}'.format(i, j), k, max_outputs=1, step=step)


def log_epoch_scalars(cm, epoch):
    tf.summary.scalar('epoch/accuracy', cm.accuracy(), step=epoch)
    tf.summary.scalar('epoch/iou', cm.mean_iou(), step=epoch)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default='/workspaces/dl_foothold/datasets/l1_preproc_allpoints')
    parser.add_argument('--working-path', type=str, default='/workspaces/dl_foothold/ws')
    parser.add_argument('--batch-size', type=int,  default=32)
    parser.add_argument('--log-interval', type=int, default=25)
    parser.add_argument('--log-images-interval', type=int, default=100)
    parser.add_argument('--out-name', type=str, default='l1_gen_allpoints')
    parser.add_argument('--eta', type=float, default=5e-4)
    parser.add_argument('--eta-mul', type=float, default=1.2)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--wd-decay-steps', type=int, default=2000000)
    parser.add_argument('--train-beta', type=float, default=0.99)
    parser.add_argument('--early-stop-threshold', type=int, default=20)
    parser.add_argument('--early-stop-epoch', type=int, default=150)
    parser.add_argument('--allow-memory-growth', action='store_true', default=False)
    parser.add_argument('--field-size', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--momentum', type=float, default=0.7)
    args, _ = parser.parse_known_args()

    if args.allow_memory_growth:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

    main(args)
