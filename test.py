from pathlib import Path
from typing import List
from nn.dataloader import load_pickle_list, PickleHistory


# 파라미터별로 평균 loss를 계산하는 함수
def mean_loss_by_param(pkl: List[PickleHistory], param: str):
    # 파라미터의 가능한 값들을 추출
    values = sorted(
        set(i["train_input"]["hyper_params"][param] for i in pkl)
    )
    # 파라미터별로 loss의 평균을 저장할 딕셔너리
    mean_loss = {}
    # 각 값에 대해 반복
    for v in values:
        # 해당 값으로 훈련된 모델들을 필터링
        models = [
            i for i in pkl if i["train_input"]["hyper_params"][param] == v
        ]
        # 모델들의 loss 리스트를 합침
        losses = sum(
            [i["train_output"]["loss"] for i in models],  # type: ignore
            [],
        )
        # loss의 평균을 계산하고 딕셔너리에 저장
        mean_loss[v] = sum(losses) / len(losses)
    # 딕셔너리를 반환
    return mean_loss


path = "output/ANN_2023_08_30_102253.pkl"
pkl = load_pickle_list(Path(path))

# 각 파라미터에 대해 평균 loss를 계산하고 출력
for param in ["lr", "n1", "n2", "n3"]:
    print(f"{param}에 대한 평균 loss:")
    print(mean_loss_by_param(pkl, param))
