import tensorflow as tf
import numpy as np

def _point_scale2img(points, _H, _W):
    """ map normalized [0,1]x[0,1] points to image [0,H]x[0,W]
    args:
        points -> [..., 2]
    """
    # with tf.variable_scope("_point_scale2img", reuse=False):
    points = points * tf.constant([_H - 1, _W - 1], "float32")
    return points


def _uncertainty(logits, cls):
    """
    gt_class_logits 参数的最后一维的范围为0-1之间
    logits: (num_boxes, H, W, Class)
    """
    gt_class_logits = logits[..., cls:cls + 1]
    return -tf.abs(gt_class_logits)


def _grid_nd_sample(in_tensor, indices, batch_dims=1):
    """ gather_nd with interpolation as torch.grid_sample func
    Args:
        in_tensor: N-d tensor, NHWC
        indices: N-d tensor with last dim equals rank(in_tensor) - batch_dims
            assuming shape [..., [..., x, y]]
        batch_dims: number of batch dimensions
    """
    # with tf.variable_scope("grid_nd_sample", reuse=False):
    interpolation_indices = indices[..., -2:]
    rounded_indices = indices[..., :-2]
    inter_floor = tf.floor(interpolation_indices)
    inter_ceil = tf.math.ceil(interpolation_indices)
    p1_indices = tf.concat([rounded_indices, inter_floor], axis=-1, name="p1_ind")
    p2_indices = tf.concat([rounded_indices, inter_ceil[..., :1], inter_floor[..., 1:2]], axis=-1,
                           name="p2_ind")
    p3_indices = tf.concat([rounded_indices, inter_floor[..., :1], inter_ceil[..., 1:2]], axis=-1,
                           name="p3_ind")
    p4_indices = tf.concat([rounded_indices, inter_ceil], axis=-1, name="p4_ind")
    mu = interpolation_indices - inter_floor

    # with tf.name_scope("gather_corners"):
    p1v = tf.gather_nd(in_tensor, tf.cast(p1_indices, tf.int32), batch_dims=batch_dims, name="gather_p1")
    p2v = tf.gather_nd(in_tensor, tf.cast(p2_indices, tf.int32), batch_dims=batch_dims, name="gather_p2")
    p3v = tf.gather_nd(in_tensor, tf.cast(p3_indices, tf.int32), batch_dims=batch_dims, name="gather_p3")
    p4v = tf.gather_nd(in_tensor, tf.cast(p4_indices, tf.int32), batch_dims=batch_dims, name="gather_p4")
    mu_x, mu_y = tf.split(mu, 2, axis=-1)
    with tf.name_scope("interpolate_p12"):
        p12_interp = p1v * (1 - mu_x) + p2v * mu_x
    with tf.name_scope("interpolate_p34"):
        p34_interp = p3v * (1 - mu_x) + p4v * mu_x
    with tf.name_scope("interpolate_y"):
        vertical_interp = p12_interp * (1 - mu_y) + p34_interp * mu_y
    return vertical_interp


class Grid_nd_sample(tf.keras.layers.Layer):
    def __init__(self, batch_dims):
        super(Grid_nd_sample, self).__init__()
        self.batch_dims = batch_dims

    def call(self, inputs, **kwargs):
        in_tensor, indices = inputs
        return _grid_nd_sample(in_tensor, indices, batch_dims=self.batch_dims)


def grid_nd_sample(in_tensor, indices, batch_dims=1):
    sample = Grid_nd_sample(batch_dims=batch_dims)
    return sample([in_tensor, indices])


def _get_uncertain_pt_coords_randomness(mask_coarse_logits, oversample_ratio=3, important_ratio=0.75, cls=0,
                                        batch_size=None):
    '''
    algo steps:
        1. gen random_pts's coord as index for gather logits
        2. as torch func grid_sample => _grid_nd_sample, to get the interp value of logits
        3. calculate step1&2's random interp points uncertainty from logits
        4. get the topK uncertainty points
        5. complete remain position

    Args:
        mask_coarse_logits:
        oversample_ratio:
        important_ratio:

    Returns:
        random_points: (b*p, num_sampled, 2)
            value [0, 1] on bbox proposal local coord
    '''

    B, H, W, C = mask_coarse_logits.shape
    num_sampled = W * H * oversample_ratio
    num_uncertain_points = int(important_ratio * num_sampled)
    mask_coarse_logits = tf.stop_gradient(mask_coarse_logits)
    B = batch_size if batch_size is not None else B

    random_coords = tf.random.uniform(shape=(B, num_sampled, 2), minval=0, maxval=1)
    unnorm_coords = _point_scale2img(random_coords, H, W)


    point_logits = grid_nd_sample(mask_coarse_logits, unnorm_coords, batch_dims=1)

    uncertainty_points = _uncertainty(point_logits, cls)
    uncertainty_points = tf.reshape(uncertainty_points, (B, -1))

    ### 4.
    _, idx = tf.math.top_k(uncertainty_points, k=num_uncertain_points)  #
    random_points = tf.gather_nd(random_coords, idx[..., None], batch_dims=1)  # (100, 441, 2)

    ### 5.
    num_random_points = num_sampled - num_uncertain_points
    # if num_random_points > 0:
    random_points = tf.concat(
        [
            random_points,
            tf.random.uniform(shape=(B, num_random_points, 2), minval=0, maxval=1)
        ], axis=1)
    # random_points
    random_points = tf.stop_gradient(random_points)
    return random_points

class Get_uncertain_pt_coords_randommness(tf.keras.layers.Layer):
    def __init__(self, oversample_ratio, important_ratio, cls_index,
                 batch_size=None):
        super(Get_uncertain_pt_coords_randommness, self).__init__()
        self.oversample_ratio = oversample_ratio
        self.important_ratio = important_ratio
        self.cls = cls_index
        self.batch_size = batch_size

    def call(self, inputs, **kwargs):
        mask_coarse_logits = inputs
        res = _get_uncertain_pt_coords_randomness(mask_coarse_logits,
                                                  oversample_ratio=self.oversample_ratio,
                                                  important_ratio=self.important_ratio,
                                                  cls=self.cls,
                                                  batch_size=self.batch_size)
        return res


def get_uncertain_pt_coords_randomness(mask_coarse_logits, oversample_ratio=3, important_ratio=0.75, cls_index=0,
                                       batch_size=None):
    pt_coords = Get_uncertain_pt_coords_randommness(oversample_ratio=oversample_ratio,
                                                    important_ratio=important_ratio,
                                                    cls_index=cls_index,
                                                    batch_size=batch_size)(mask_coarse_logits)
    return pt_coords

def gen_regular_grid_coord(W, H, N):
    """
    gen 8x8 regular grid
    """
    ### regular grid
    x = np.array(list(range(0, W))) / (W - 1)
    y = np.array(list(range(0, H))) / (H - 1)
    X, Y = tf.meshgrid(x, y)
    indices = tf.stack([X, Y])
    indices = tf.transpose(indices, (1, 2, 0))[None, ...]
    regular_coord_point = tf.tile(indices, (N, 1, 1, 1))
    regular_coord_point = tf.cast(regular_coord_point, tf.float32)
    return regular_coord_point


def get_point_coords_wrt_image(boxes, point_coords):
    """
    Convert box-normalized [0, 1] x [0, 1] point cooordinates to image-level coordinates.
    Args:
        boxes (Tensor): A tensor of shape (R, 4) that contains bounding boxes.
           normalized coordinates (y,x,y,x).
        point_coords (Tensor): A tensor of shape (R, P, 2) that contains
            [0, 1] x [0, 1] box-normalized coordinates of the P sampled points.
    Returns:
        point_coords_wrt_image (Tensor): A tensor of shape (R, P, 2) that contains
            image-normalized coordinates of P sampled points.
    """
    # with tf.variable_scope("get_point_coords_wrt_image", reuse=False):
    boxes = tf.stop_gradient(boxes)
    point_coords = tf.stop_gradient(point_coords)
    h = boxes[:, None, 2] - boxes[:, None, 0]
    w = boxes[:, None, 3] - boxes[:, None, 1]
    y1 = boxes[:, None, 0]
    x1 = boxes[:, None, 1]
    scale = tf.stack([h, w], axis=-1)
    trans = tf.stack([y1, x1], axis=-1)
    point_coords = point_coords * scale
    point_coords = point_coords + trans
    return point_coords

def get_uncertain_point_coords_on_grid(uncertainty_map, num_points):
    """
    args:
        uncertainty_map: (B*P, H, W, 1)
            H&W_size: 28 -> 56 -> 112 -> 224
        num_points: 28*28
    return:
        points indices
        value[0, 1] points on local proposal
        shape (b*p, num_points, 2)
    """
    # print("get_uncertain_point_coords_on_grid")
    # print(uncertainty_map.shape)
    # print(num_points)
    R, H, W, C = map(int, uncertainty_map.shape)
    h_step = 1.0 / float(H)
    w_step = 1.0 / float(W)
    uncertainty_map = tf.reshape(uncertainty_map, (R, H * W))
    _, point_indices = tf.math.top_k(uncertainty_map, k=num_points)  # (R, 784)
    point_coords_y = w_step / 2.0 + tf.cast(point_indices % W, tf.float32) * w_step
    point_coords_x = h_step / 2.0 + tf.cast(point_indices // W, tf.float32) * h_step
    point_coords = tf.stack([point_coords_y, point_coords_x], 2)
    # point_coords = tf.concat([point_coords_y[..., None], point_coords_x[..., None]], -1)
    return point_indices, point_coords

