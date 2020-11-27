import tensorflow as tf
tf.enable_eager_execution()

a = tf.random_normal((2, 3, 4, 3))
shape = a.get_shape().as_list()
b = tf.squeeze(tf.reshape(tf.transpose(a, perm=[0, 3, 1, 2]), (-1, 1, *shape[1:3])))

for l in range(shape[0] * shape[-1]):
    lf = l // shape[-1]
    lb = l % shape[-1]

    print(f'Compare b[{l}, :, :] with a[{lf}, :, :, {lb}], error: {tf.reduce_sum(tf.abs(b[l, :, :] - a[lf, :, :, lb]))}')

c = tf.transpose(tf.reshape(tf.expand_dims(b, axis=1), (shape[0], shape[-1], *shape[1:3])), perm=[0, 2, 3, 1])
print(f'Compare c with a, error: {tf.reduce_sum(tf.abs(c - a))}')