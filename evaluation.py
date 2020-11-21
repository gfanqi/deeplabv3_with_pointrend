import os
from functools import partial

import pandas as pandas
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from core.deeplabV3plus import Deeplabv3, deeplab_add_pointrend_head
from core.pointrend import PointRend
from core.utils_core import get_best_weight, preprocessing_all


# from model.deeplabV3plus import Deeplabv3
# from utils import preprocessing_all, get_best_weight


if __name__ == "__main__":
    # ALPHA = 1.0
    test_pointrend = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    HEIGHT = 512
    WIDTH = 512
    train_steps_per_epoch = 200
    batch_size_per_replica = 4
    # test_batch_size_per_replica = 4
    gpus = tf.config.experimental.list_physical_devices('GPU')

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    num_replica = len(gpus)
    print(num_replica)
    log_dir = "logs/"
    if not os.path.exists(log_dir): os.mkdir(log_dir)
    # strategy = tf.distribute.MirroredStrategy()
    strategy = tf.distribute.OneDeviceStrategy("/gpu:0")
    with strategy.scope():
        if test_pointrend:
            deeplabv3plus = Deeplabv3(weights=None, input_shape=(HEIGHT, WIDTH, 3), classes=1, backbone='mobilenetv2')
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
                'logits_semantic'], batch_size=batch_size_per_replica, num_class=1)
            # model = pointrend.build_model_for_train(important_ratio=0.75)
            model = pointrend.build_model_for_infer(batch_size=batch_size_per_replica,
                                                    weights_path='./logs/time-1605880347ep315-loss0.140-val_loss0.129.h5')
        else:
            model = Deeplabv3(weights=None,input_shape=(HEIGHT, WIDTH, 3), classes=1, backbone='mobilenetv2')
            model.compile(loss=tf.keras.losses.categorical_crossentropy,
                          optimizer=Adam(lr=1e-4),
                          metrics=['accuracy'],
                          )
            model.load_weights('logs/time-1605692926ep213-loss0.032-val_loss0.029.h5',by_name=True)
    # weights_path, initial_epoch = get_best_weight(log_dir)
    # model.load_weights('logs/time-1605692926ep213-loss0.032-val_loss0.029.h5',by_name=True)
    print('weights loaded!')
    data = val_data
    classes = set(data['third_level_label'].tolist())
    # print(classes)
    # input()
    preprocessing_all = partial(preprocessing_all, H=HEIGHT, W=WIDTH)
    total_count = pd.DataFrame()

    for class_name in classes:
        print(class_name)
        data = val_data[val_data['third_level_label'] == class_name]
        data = data.reset_index(drop=True)
        # print(len(data))
        if len(data) == 0:
            res = {'loss': 0, 'accuracy': 0}
            res_df = pandas.DataFrame(res, index=[class_name])
            # res_df.index = class_name
            total_count = total_count.append(res_df)
        else:
            cache_dir = os.path.expanduser('~/Image/data/all_images')
            ori_image_path_list, seg_image_path_list = generate_semantic_segmentation_data(data, cache_dir)
            db = tf.data.Dataset.from_tensor_slices((ori_image_path_list, seg_image_path_list)).map(
                preprocessing_all, num_parallel_calls=-1). \
                batch(batch_size=batch_size_per_replica * num_replica).prefetch(-1)
            # print(next(iter(db)))
            # exit()
            # a = model.predict(next(iter(db)))
            # print(len(a))
            # print(a[0].shape)
            # print(a[1].shape)
            # print(a[2].shape)
            # print(a[3].shape)
            # exit()

            res = model.evaluate(db, steps=len(data) // (batch_size_per_replica * num_replica), verbose=1,
                                 return_dict=True)
        res.update({'num_data': len(data)})
        res_df = pandas.DataFrame(res, index=[class_name])
        # res_df.index = class_name
        total_count = total_count.append(res_df)
        total_count.to_csv('evalution.csv', encoding='GBK')

