import tensorflow as tf
from keras.layers import Input, Dense, LayerNormalization, Dropout
from keras import Model


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_size = embed_size
        self.head_dim = embed_size // num_heads

        assert (
            self.head_dim * num_heads == embed_size
        ), "Embedding size needs to be divisible by num heads"

        self.wq = Dense(embed_size)
        self.wk = Dense(embed_size)
        self.wv = Dense(embed_size)
        self.out = Dense(embed_size)

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        seq_len = tf.shape(q)[1]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        score = tf.matmul(q, k, transpose_b=True)
        score /= tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
        if mask is not None:
            score += mask * -1e9

        attention_weights = tf.nn.softmax(score, axis=-1)
        out = tf.matmul(attention_weights, v)

        out = tf.transpose(out, perm=[0, 2, 1, 3])
        out = tf.reshape(out, (batch_size, seq_len, self.embed_size))
        return self.out(out)

    def split_heads(self, x):
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])


def create_transformer_model(
    input_dim,
    seq_len,
    embed_size=128,
    num_heads=4,
    ff_dim=512,
    num_transformer_blocks=2,
):
    inputs = Input(shape=(input_dim,))
    x = Dense(embed_size)(inputs)
    x = tf.reshape(x, (-1, 1, embed_size))

    # Transformer blocks
    for _ in range(num_transformer_blocks):
        # Multi-head attention
        attention = MultiHeadSelfAttention(embed_size, num_heads)
        x1 = attention(x, x, x, None)
        x1 = Dropout(0.1)(x1)
        x = LayerNormalization(epsilon=1e-6)(x + x1)

        # Feed-forward network
        ff = Dense(ff_dim, activation="relu")(x)
        ff = Dense(embed_size)(ff)
        ff = Dropout(0.1)(ff)
        x = LayerNormalization(epsilon=1e-6)(x + ff)

    x = tf.tile(x, [1, seq_len, 1])
    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


model = create_transformer_model(input_dim=128, seq_len=10)
model.compile(optimizer="adam", loss="mse")
print(model.summary())
