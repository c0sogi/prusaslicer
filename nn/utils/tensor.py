from typing import Optional
import tensorflow as tf

tf.float32


def create_tensor_spec(
    dtype: tf.DType,
    shape: tf.TensorShape,
    name: Optional[str] = None,
) -> tf.TensorSpec:
    return tf.TensorSpec(  # type: ignore
        dtype=tf.dtypes.as_dtype(dtype),
        shape=tf.TensorShape(shape) if shape is not None else None,
        name=name,
    )


if __name__ == "__main__":
    print(create_tensor_spec(tf.float32, tf.TensorShape((2,)), "test"))
