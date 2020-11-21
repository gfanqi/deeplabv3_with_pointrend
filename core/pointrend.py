import tensorflow as tf
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops.gen_image_ops import resize_bilinear
import numpy as np
import matplotlib.pyplot as plt
from core.utils import get_uncertain_pt_coords_randomness, _grid_nd_sample, grid_nd_sample, _uncertainty, \
    get_uncertain_point_coords_on_grid, _point_scale2img


class PointRend(object):
    def __init__(self, semantic_segment_model, semantic_segment_model_layer_names, batch_size, num_class,
                 num_suvdivision_steps=3):
        '''

        Args:
            semantic_segment_model:
        '''
        self.semantic_segment_model_layer_names = semantic_segment_model_layer_names
        self.batch_size = batch_size
        self.num_classes = num_class
        self.num_subdivision_steps = num_suvdivision_steps
        self.semantic_segment_model = self.get_mid_outputs(semantic_segment_model, semantic_segment_model_layer_names)
        self.mask_point_head_model = self._create_mask_point_head(num_classes=num_class, is_training=False,
                                                                  last_dim=673)

    @staticmethod
    def get_mid_outputs(model, layer_name_list):
        '''
        最后一个层的名字作为粗分类的层，也就是未经过resize前的层
        Args:
            model:
            layer_name_list:

        Returns:

        '''
        mid_layer_outputs = []
        for layer_name in layer_name_list:
            mid_layer_outputs.append(model.get_layer(name=layer_name).output)
        return tf.keras.models.Model(inputs=model.inputs, outputs=mid_layer_outputs)

    def _create_mask_point_head(self, fine=None, num_classes=None,
                                is_training=False, last_dim=673, resuse_time='', ):
        if fine is None:
            fine = tf.keras.layers.Input(shape=(None, last_dim))
        out = self._mask_point_head(num_classes=num_classes,
                                    fine_gained_feature=fine,
                                    is_training=is_training,
                                    reuse_time=resuse_time,
                                    )
        return tf.keras.models.Model(inputs=fine, outputs=out)

    def _mask_point_head(self, num_classes, fine_gained_feature, is_training=True, reuse_time=''):
        '''

        Args:
            num_classes:
            fine_gained_feature:
            is_training:
            reuse_time:
        Returns:
        '''
        net = fine_gained_feature
        # net = tf.concat(fine_gained_feature, axis=-1)  # (b*p, sample_points, c+cls)
        net = tf.keras.layers.Conv1D(256, kernel_size=1, activation=tf.nn.relu, use_bias=True,
                                     kernel_initializer="glorot_normal",
                                     trainable=is_training, name=reuse_time + "lin0")(net)  # (b*p, sample_points, C)
        net = tf.keras.layers.Conv1D(256, kernel_size=1, activation=tf.nn.relu, use_bias=True,
                                     kernel_initializer="glorot_normal",
                                     trainable=is_training, name=reuse_time + "lin1")(net)  # (b*p, sample_points, C)
        net = tf.keras.layers.Conv1D(256, kernel_size=1, activation=tf.nn.relu, use_bias=True,
                                     kernel_initializer="glorot_normal",
                                     trainable=is_training, name=reuse_time + "lin2")(net)  # (b*p, sample_points, C)
        net = tf.keras.layers.Conv1D(num_classes, kernel_size=1, activation=tf.identity, use_bias=True,
                                     kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.001),
                                     bias_initializer=tf.keras.initializers.Zeros(),
                                     trainable=is_training, name=reuse_time + "lin_out")(net)
        return net

    # oversample_ratio = 1,
    def build_model_for_train(self, oversample_ratio=1, important_ratio=0.6):
        out = self.semantic_segment_model.output
        coarse = out[-1]  # 最后一层的输出为粗分类层
        points = get_uncertain_pt_coords_randomness(coarse, batch_size=self.batch_size,
                                                    oversample_ratio=oversample_ratio,
                                                    important_ratio=important_ratio, )
        fine = []
        for mid_output in out:
            _, mid_output_H, mid_output_W, _ = mid_output.shape
            fine.append(grid_nd_sample(mid_output, _point_scale2img(points, mid_output_H, mid_output_W)))
        fine_concated = tf.keras.layers.concatenate(fine, axis=-1)
        coarse_selected_points = fine[-1]
        rend = self._mask_point_head(num_classes=self.num_classes,
                                     fine_gained_feature=fine_concated)
        rend_and_coords = tf.keras.layers.concatenate([coarse_selected_points, rend, points], axis=-1,
                                                      name='rend_and_coords')
        coarse = gen_image_ops.resize_bilinear(coarse, size=(512, 512), align_corners=True, name='coarse')
        # coarse = tf.keras.layers.Lambda(lambda x:gen_image_ops.resize_bilinear(x, size=(512, 512), align_corners=True, )
        #                                 ,name='coarse')
        # coarse = tf.keras.layers.Activation(activation='sigmoid',name='coarse')(coarse)
        model = tf.keras.Model(inputs=self.semantic_segment_model.inputs, outputs=[coarse,
                                                                                   rend_and_coords]
                               )
        return model

    def pointrend_loss(self, y_true, y_pred):
        y_true = tf.expand_dims(y_true, axis=-1)
        _, H, W, _ = y_true.shape
        coords = y_pred[..., -2:]
        coords = _point_scale2img(coords, H, W)
        rend = y_pred[..., -2 - self.num_classes:-2]
        y_true_ = _grid_nd_sample(in_tensor=y_true, indices=coords, batch_dims=1)
        #
        # num_positive1 = tf.reduce_sum(y_true)
        # num_negtive1 = tf.reduce_sum(1 - y_true)
        # tf.print(num_positive1, num_negtive1)
        # num_positive = tf.reduce_sum(y_true_)
        # num_negtive = tf.reduce_sum(1 - y_true_)
        # tf.print(num_positive, num_negtive)
        return tf.keras.losses.binary_crossentropy(y_true=y_true_, y_pred=rend)

    def rend_and_coords_acc(self, y_true, y_pred):
        y_true = tf.expand_dims(y_true, axis=-1)
        _, H, W, _ = y_true.shape
        coords = y_pred[..., -2:]
        coords = _point_scale2img(coords, H, W)
        rend = y_pred[..., -2 - self.num_classes:-2]
        y_true_ = _grid_nd_sample(in_tensor=y_true, indices=coords, batch_dims=1)
        return tf.keras.metrics.binary_accuracy(y_true=y_true_, y_pred=rend)

    def selected_coarse_acc(self, y_true, y_pred):
        y_true = tf.expand_dims(y_true, axis=-1)
        _, H, W, _ = y_true.shape
        coords = y_pred[..., -2:]
        coords = _point_scale2img(coords, H, W)
        selected_coarse = y_pred[..., :-2 - self.num_classes]
        y_true_ = _grid_nd_sample(in_tensor=y_true, indices=coords, batch_dims=1)
        return tf.keras.metrics.binary_accuracy(y_true=y_true_, y_pred=selected_coarse)

    def build_model_for_infer(self, weights_path=None, batch_size=1):
        self.semantic_segment_model.load_weights(weights_path, by_name=True)
        self.mask_point_head_model.load_weights(weights_path, by_name=True)

        inputs = self.semantic_segment_model.inputs
        out = self.semantic_segment_model.output
        mask_logit = out[-1]
        # fine_gained_features = out[:-1]
        # origin_output = resize_bilinear(mask_logit, (512, 512), align_corners=True,
        #                                 )
        # _, H, W, C = mask_logit.shape
        ResizeShape_list = [128, 256, 512, 1024, 2048]
        grid_rate = [0.10, 0.09, 0.08, 0.07,0.05]
        for subdivision_step, (size, rate) in enumerate(zip(ResizeShape_list,grid_rate)):
            ResizeShape = (size, size)
            # ResizeShape = list(map(lambda a: int(a) * 2, mask_logit.shape[1:3]))
            mask_logit = resize_bilinear(mask_logit, ResizeShape, align_corners=True,
                                         name='upsampleing')
            # R, sH, sW, C = map(int, mask_logit.shape)
            R, sH, sW, C = mask_logit.shape
            R = batch_size
            uncertainty_map = _uncertainty(mask_logit, cls=0)
            point_indices, point_coords = get_uncertain_point_coords_on_grid(uncertainty_map,
                                                                             # self.num_subdivision_points
                                                                             rate=rate, batch_size=batch_size)
            point_coords = point_coords[..., ::-1]
            # coarse_coords = _point_scale2img(point_coords, sH, sW)  # local feat
            # coarse_features = _grid_nd_sample(mask_logit, coarse_coords, batch_dims=1)
            # show_points_image(mask_logit, coarse_coords)
            fine_gained_feature = []
            for mid_output in out:
                _, fine_H, fine_W, _ = mid_output.shape
                fine_gained_feature_coords = _point_scale2img(point_coords, fine_H, fine_W)
                fine_gained_feature.append(_grid_nd_sample(mid_output, fine_gained_feature_coords, batch_dims=1))
            fine_gained_feature = tf.concat(fine_gained_feature, axis=-1)

            point_logits = self.mask_point_head_model(
                fine_gained_feature
            )
            inds = tf.cast(point_coords * tf.constant((sH, sW), tf.float32), tf.int32)
            expdim = tf.tile(tf.range(0, R, dtype=tf.int32)[..., None], [1, inds.shape[1]])[..., None]
            inds = tf.concat([expdim, inds], -1)
            mask_logit = tf.tensor_scatter_nd_update(mask_logit, indices=inds, updates=point_logits)
        model = tf.keras.models.Model(inputs=inputs, outputs=mask_logit)
        return model

    def inference(self, inputs, weights_path, batch_size=1):
        self.semantic_segment_model.load_weights(weights_path, by_name=True)
        self.mask_point_head_model.load_weights(weights_path, by_name=True)

        # inputs = self.semantic_segment_model.inputs
        out = self.semantic_segment_model(inputs)
        mask_logit = out[-1]

        ResizeShape_list = [128, 256, 320, 480, 512]
        for subdivision_step, size in enumerate(ResizeShape_list):
            ResizeShape = (size, size)
            # ResizeShape = list(map(lambda a: int(a) * 2, mask_logit.shape[1:3]))
            mask_logit = resize_bilinear(mask_logit, ResizeShape, align_corners=True,
                                         name='upsampleing')
            # R, sH, sW, C = map(int, mask_logit.shape)
            R, sH, sW, C = mask_logit.shape
            R = batch_size
            uncertainty_map = _uncertainty(mask_logit, cls=0)
            point_indices, point_coords = get_uncertain_point_coords_on_grid(uncertainty_map,
                                                                             # self.num_subdivision_points
                                                                             5000, batch_size=batch_size)
            point_coords = point_coords[..., ::-1]
            coarse_coords = _point_scale2img(point_coords, sH, sW)  # local feat
            # coarse_features = _grid_nd_sample(mask_logit, coarse_coords, batch_dims=1)
            show_points_image(mask_logit, coarse_coords)
            fine_gained_feature = []
            for mid_output in out:
                _, fine_H, fine_W, _ = mid_output.shape
                fine_gained_feature_coords = _point_scale2img(point_coords, fine_H, fine_W)
                fine_gained_feature.append(_grid_nd_sample(mid_output, fine_gained_feature_coords, batch_dims=1))
            fine_gained_feature = tf.concat(fine_gained_feature, axis=-1)

            point_logits = self.mask_point_head_model(
                fine_gained_feature
            )
            inds = tf.cast(point_coords * tf.constant((sH, sW), tf.float32), tf.int32)
            expdim = tf.tile(tf.range(0, R, dtype=tf.int32)[..., None], [1, inds.shape[1]])[..., None]
            inds = tf.concat([expdim, inds], -1)
            mask_logit = tf.tensor_scatter_nd_update(mask_logit, indices=inds, updates=point_logits)
        return mask_logit


def preprocessing(image_path, H=512, W=512):
    x = tf.io.read_file(image_path)
    x = tf.cond(
        tf.image.is_jpeg(x),
        lambda: tf.image.decode_jpeg(x, channels=3),
        lambda: tf.image.decode_png(x, channels=3))

    x = tf.cast(x, tf.float32)
    x = x / 255.
    x = tf.image.resize(x, size=(H, W))
    # x = tf.expand_dims(x,axis=0)
    return x


def preprocessing_mask(image_path, H, W):
    x = tf.io.read_file(image_path)
    x = tf.cond(
        tf.image.is_jpeg(x),
        lambda: tf.image.decode_jpeg(x, channels=1),
        lambda: tf.image.decode_png(x, channels=1))
    # x = tf.image.is
    x = tf.cast(x > 0, tf.float32)
    x = tf.image.resize(x, size=(H, W))
    # x = tf.expand_dims(x,axis = 0)
    return x


def preprocessing_all(x, y, H, W):
    return preprocessing(x, H, W), preprocessing_mask(y, H, W)


def show_points_image(array, coord):
    array = tf.nn.sigmoid(array)

    array = array.numpy()
    coord = coord.numpy()
    array = array / np.max(array)
    array = array * 255
    zeros = np.zeros_like(array)
    array = np.concatenate([zeros, zeros, array], axis=-1)
    array = array.astype(np.uint8)
    array = np.squeeze(array)
    coord = coord.astype(np.int)
    coord = np.squeeze(coord)
    x = coord[..., 0]
    y = coord[..., 1]

    array[x, y] = np.array([255, 0, 0])
    array = array.astype(np.uint8)
    plt.imshow(array)
    plt.show()
    plt.close()
    # plt.show()
    # exit()

    #
    # array = np.zeros(shape=(512, 512, 3)) * 255
    # coords = np.random.random(size=(1000, 2)) * 512
    # # y = np.random.random(size=(1000, 1)) * 512
    # # x = x.astype(np.int)
    # # y = y.astype(np.int)
    # # zeros = np.zeros(shape=(2,50))
    # # coord = np.concatenate([ coord,zeros,], axis=-1)
    # show_points_image(array, coords)
