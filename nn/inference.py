# flake8: noqa
from pathlib import Path
from typing import Union

import numpy as np
from tensorflow import keras

from nn.typings import DataLike

from .ann import ANN


def inference(
    model_path: Union[str, Path], input_data: DataLike
) -> np.ndarray:
    # 모델 초기화
    model = keras.models.load_model(model_path)  # type: ignore
    assert isinstance(model, keras.Model), type(model)
    if model is None:
        raise ValueError(f"Model not found: {model_path}")

    # # 모델 구조 출력
    # model.summary()
    # keras.utils.plot_model(
    #     model,
    #     to_file=str(Path(model_path).with_suffix(".png")),
    #     show_shapes=True,
    #     show_layer_names=True,
    # )

    # Inference 수행
    return model.predict(input_data)


if __name__ == "__main__":
    # 예시 데이터 (추후 실제 데이터로 대체해야 합니다)
    sample_input_data = np.array(
        [[80, 220, 0.1, 50, 1.24, 59, 4, 105, 0.183, 8.5e-05]]
    )

    for model_path in Path("output").glob("*.keras"):
        inference(model_path, sample_input_data)
