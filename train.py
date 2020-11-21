import os
import time
from functools import partial

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from core.deeplabV3plus import Deeplabv3
from core.pointrend import preprocessing, preprocessing_mask, PointRend
from core.utils import _uncertainty, get_uncertain_point_coords_on_grid, grid_nd_sample, _point_scale2img

if __name__ == "__main__":
    # ALPHA = 1.0
    HEIGHT = 512
    WIDTH = 512
    train_steps_per_epoch = 1
    train_batch_size_per_replica = 2
    test_batch_size_per_replica = 2
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    num_replica = max(len(gpus), 1)
    print(num_replica)
    log_dir = "logs"
    if not os.path.exists(log_dir): os.mkdir(log_dir)
    strategy = tf.distribute.MirroredStrategy()
    # strategy = tf.distribute.OneDeviceStrategy("/gpu:0")
    # 保存的方式，3世代保存一次
    now_str = str(int(time.time()))
    callbacks = [ModelCheckpoint(
        os.path.join(log_dir, 'time-' + now_str + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'),
        monitor='loss',
        save_weights_only=True,
        save_best_only=True,
        period=3,
        verbose=1
    ),
        # 学习率下降的方式，val_loss 2次不下降就下降学习率继续训练
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            verbose=1,
            min_lr=1e-8,
        ),
        # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
        # EarlyStopping(
        #     monitor='val_loss',
        #     min_delta=0,
        #     patience=6,
        #     verbose=1
        # ),
    ]
    # host8
    train_ori_image_path_list, train_seg_image_path_list = [os.path.join('./images/ori_image/000039.jpg')] * 1000, [
        os.path.join('./images/mask/000039.png')] * 1000

    num_steps_per_data_circle = len(train_seg_image_path_list) // train_batch_size_per_replica * num_replica
    val_ori_image_path_list, val_seg_image_path_list = [os.path.join('./images/ori_image/000039.jpg')] * 1000, [
        os.path.join('./images/mask/000039.png')] * 1000
    preprocessing = partial(preprocessing, H=HEIGHT, W=WIDTH)
    preprocessing_mask = partial(preprocessing_mask, H=HEIGHT, W=WIDTH)


    # with strategy.scope():
    def batch(x, y):
        return x, (y, y)


    train_db = tf.data.Dataset.from_tensor_slices((train_ori_image_path_list, train_seg_image_path_list)).map(
        tf.autograph.experimental.do_not_convert(
            lambda x, y: (preprocessing(x), preprocessing_mask(y))), num_parallel_calls=-1). \
        batch(batch_size=train_batch_size_per_replica * num_replica, drop_remainder=True).map(batch).repeat().prefetch(
        -1)

    val_db = tf.data.Dataset.from_tensor_slices((val_ori_image_path_list, val_seg_image_path_list)).map(
        tf.autograph.experimental.do_not_convert(
            lambda x, y: (preprocessing(x), preprocessing_mask(y))), num_parallel_calls=-1). \
        batch(batch_size=train_batch_size_per_replica * num_replica, drop_remainder=True).map(batch).repeat().prefetch(
        -1)

    # 获取model
    with strategy.scope():
        # deeplab_add_pointrend = deeplab_add_pointrend_head(weights=None, input_shape=(512, 512, 3), classes=1,
        #                                                    backbone='mobilenetv2',
        #                                                    OS=16, alpha=1., activation=None, )
        deeplabv3plus = Deeplabv3(weights=None, input_shape=(512, 512, 3), classes=1,
                                  backbone='mobilenetv2',
                                  OS=16, alpha=1., activation='sigmoid', )
        # deeplabv3plus.trainable = False
        # deeplab_add_pointrend.trainable = False
        pointrend = PointRend(deeplabv3plus, semantic_segment_model_layer_names=[
            'expanded_conv_3_project_BN',
            # 'expanded_conv_4_add',
            # 'expanded_conv_5_add',
            'expanded_conv_6_project_BN',
            # 'expanded_conv_7_add',
            # 'expanded_conv_8_add',
            # 'expanded_conv_9_add',
            'expanded_conv_10_project_BN',
            # 'expanded_conv_11_add',
            # 'expanded_conv_12_add',
            'expanded_conv_13_project_BN',
            # 'expanded_conv_14_add',
            # 'expanded_conv_15_add',
            'expanded_conv_16_project_BN',
            'logits_semantic'], batch_size=train_batch_size_per_replica, num_class=1)
        model = pointrend.build_model_for_train(important_ratio=0.24, oversample_ratio=1)
        model.summary()
        model.load_weights('./logs/time-1605924594ep249-loss0.338-val_loss0.323.h5', by_name=True,
                           skip_mismatch=True)
        initial_epoch = 213


        # model = Deeplabv3(weights='pascal_voc', input_shape=(HEIGHT, WIDTH, 3), classes=1, backbone='xception')

        def coarse_loss(y_true, y_pred):
            uncertainty_map = _uncertainty(y_pred, cls=0)
            _, coords = get_uncertain_point_coords_on_grid(uncertainty_map, rate=0.1,
                                                           batch_size=train_batch_size_per_replica)
            coords = coords[..., ::-1]
            _, mid_output_H, mid_output_W, _ = y_pred.shape
            y_pred_important = grid_nd_sample(y_pred, _point_scale2img(coords, mid_output_H, mid_output_W))
            y_true_important = grid_nd_sample(y_true, _point_scale2img(coords, mid_output_H, mid_output_W))
            loss1 = tf.keras.losses.binary_crossentropy(y_true_important, y_pred_important)
            loss2 = tf.keras.losses.binary_crossentropy(y_true, y_pred)
            loss = tf.concat([20 * tf.reshape(loss1, shape=(train_batch_size_per_replica, -1)),
                              0.1 * tf.reshape(loss2, shape=(train_batch_size_per_replica, -1))], axis=-1)

            return loss


        model.compile(
            loss=[
                coarse_loss,
                # tf.keras.losses.categorical_crossentropy,
                pointrend.pointrend_loss,
                # accuracy_for_rend_and_coords
                # [accuracy_for_rend_and_coords,pointrend.pointrend_loss,]
            ],
            loss_weights=[1, 1, ],
            optimizer=Adam(lr=1e-4),
            metrics={'tf_op_layer_coarse': tf.keras.metrics.binary_accuracy,
                     'rend_and_coords': [pointrend.rend_and_coords_acc, pointrend.selected_coarse_acc], },
        )

    # initial_epoch = 0
    print('loaded_weights_from_epoch_{}'.format(initial_epoch))

    num_epoch_per_data_cicle = num_steps_per_data_circle // train_steps_per_epoch
    # num_epoch_per_data_cicle = 1
    print(num_epoch_per_data_cicle)


    model.fit(
        train_db,
        steps_per_epoch=max(1, train_steps_per_epoch),
        validation_data=val_db,
        validation_steps=max(1, 20),
        epochs=10 * num_epoch_per_data_cicle + initial_epoch,
        initial_epoch=initial_epoch,
        callbacks=callbacks)

    model.save_weights(os.path.join(log_dir, 'last1.h5'))
