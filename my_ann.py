# flake8: noqa
import itertools
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

from .config import ANNConfig

# from sklearn.preprocessing import MinMaxScaler

InputParams = Literal[
    "weight", "width1", "width2", "width3", "height", "depth", "strength"
]

OutputParams = Literal[
    "weight", "width1", "width2", "width3", "height", "depth", "strength"
]

model_config = ANNConfig()


def train(model: str):
    print(f"\n\n{'#' * 20} Model: {model} {'#' * 20}\n\n")
    result_columns = (
        "Case",
        "L.rate",
        "Nr-HL1",
        "Nr-HL2",
        "RMSE",
        "MAE",
        "MAPE",
    )
    temp_memory = np.zeros((model_config.number_of_cases, len(result_columns)))
    for case, (lr, n1, n2) in enumerate(
        itertools.product(
            model_config.lrs, model_config.n1s, model_config.n2s
        ),
        start=1,
    ):
        print(
            f"\n\nTrial No.{case} | Model: {model} | lr: {lr} | n1: {n1} | n2: {n2}"
        )

        ################ 신경망 구조 재설계 ################

        tf.keras.backend.clear_session()
        model = ANN(model_config, lr=lr, n1=n1, n2=n2)

        ################ 신경망 학습 ################

        hist = model.fit(
            train_data,
            train_labels,
            epochs=Epoch,
            verbose=0,  # type: ignore
            callbacks=[AccuracyPerEpoch()],
            batch_size=100,
        )
        model.save(f"./cases/{model}_{case}.h5")
        print(
            "\n[Final Epochs]    RMSE:{:.5f},   MAE: {:.5f},  MAPE: {:.2f}%".format(
                np.sqrt(hist.history["mse"][-1]),
                hist.history["mae"][-1],
                hist.history["mape"][-1],
            )
        )
        temp_memory[case, :] = (
            case,
            lr,
            n1,
            n2,
            np.sqrt(hist.history["mse"][-1]),
            hist.history["mae"][-1],
            hist.history["mape"][-1],
        )

    Tr_result_temp_pd = pd.DataFrame(
        temp_memory,
        columns=["Case", "L.rate", "Nr-HL1", "Nr-HL2", "RMSE", "MAE", "MAPE"],
    )
    Tr_result_temp_pd.to_csv(
        "./27case_MLmodels_ANN_PRESM/Tr_result_epoch%d.csv" % (Epoch),
        index=None,
    )

    # %%
    Tr_result_temp_pd

    # %%
    Tr_result_rank = Tr_result_temp_pd.loc[
        np.where(Tr_result_temp_pd["MAPE"] != 100)
    ]
    Tr_result_rank = Tr_result_rank.sort_values(["MAPE"], ascending=True)
    Tr_result_rank


train_data.shape, train_labels.shape

# %%
Fold = 5
FoldDataNo = int(train_data.shape[0] / Fold)

# %%
# Validation dataset
for i in range(Fold):
    temp_Valid_Data = train_data.iloc[FoldDataNo * i : FoldDataNo * (i + 1), :]
    s1 = "ValidData_Fold%d = temp_Valid_Data" % (i + 1)
    exec(s1)

    temp_Valid_Label = train_labels.iloc[
        FoldDataNo * i : FoldDataNo * (i + 1), :
    ]
    s2 = "ValidLabel_Fold%d = temp_Valid_Label" % (i + 1)
    exec(s2)

ValidData_Fold5

# %%
# test dataset
testData = TestData
testLabel = TestLabel

testData

# %%
# Training Dataset
for i in range(Fold):
    temp_Train_Data_Front = train_data.iloc[: FoldDataNo * i, :]
    temp_Train_Data_Back = train_data.iloc[FoldDataNo * (i + 1) :, :]
    temp_Train_Data_Total = np.concatenate(
        [temp_Train_Data_Front, temp_Train_Data_Back], axis=0
    )
    s1 = "TrainData_Fold%d  = temp_Train_Data_Total" % (i + 1)
    exec(s1)

    temp_Train_Label_Front = train_labels.iloc[: FoldDataNo * i, :]
    temp_Train_Label_Back = train_labels.iloc[FoldDataNo * (i + 1) :, :]
    temp_Train_Label_Total = np.concatenate(
        [temp_Train_Label_Front, temp_Train_Label_Back], axis=0
    )
    s2 = "TrainLabel_Fold%d  = temp_Train_Label_Total" % (i + 1)
    exec(s2)

TrainData_Fold1.shape, TrainLabel_Fold1.shape

# %%


# %% [markdown]
# ### 최고성능 모델 재학습 및 모델 & 히스토리 저장

# %%
for F in range(Fold):
    s1 = "TrainData  = TrainData_Fold%d" % (F + 1)
    exec(s1)
    s2 = "TrainLabel = TrainLabel_Fold%d" % (F + 1)
    exec(s2)
    s3 = "ValidData  = ValidData_Fold%d" % (F + 1)
    exec(s3)
    s4 = "ValidLabel  = ValidLabel_Fold%d" % (F + 1)
    exec(s4)
    for M in range(1):
        Tr_result_temp = pd.read_csv(
            "./27case_ANN_prediction/Tr_result_epoch2000.csv", sep=","
        )
        learningRate = Tr_result_temp.sort_values(
            ["MAPE"], ascending=True
        ).iloc[0, 1]
        noOfNeuron1 = np.int(
            Tr_result_temp.sort_values(["MAPE"], ascending=True).iloc[0, 2]
        )
        noOfNeuron2 = np.int(
            Tr_result_temp.sort_values(["MAPE"], ascending=True).iloc[0, 3]
        )
        Epoch = 20000

        print("\n\n\nPrediction :" + Model[M])
        print("Learning rate : {:.3}".format(learningRate))
        print("Hidden 1 neuron : %d" % (noOfNeuron1))
        print("Hidden 2 neuron : %d" % (noOfNeuron2))

        #     exec('Label_Trn = TrainLabel_%d'%(M+1))

        ################ 신경망 구조 재설계 ################

        tf.keras.backend.clear_session()

        def ANN_model(input_data):
            model = keras.Sequential()
            model.add(
                keras.layers.Dense(
                    units=noOfNeuron_in,
                    input_shape=(input_data.shape[1],),
                    activation="relu",
                )
            )  # Input  Layer
            model.add(
                keras.layers.Dense(units=noOfNeuron1, activation="relu")
            )  # Hidden Layer 1
            model.add(
                keras.layers.Dense(units=noOfNeuron2, activation="relu")
            )  # Hidden Layer 2
            model.add(
                keras.layers.Dense(
                    units=noOfNeuron_out,
                )
            )  # Output Layer
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=learningRate),
                loss=keras.losses.mean_absolute_error,
                metrics=["mse", "mae", "mape"],
            )
            return model

        model = ANN_model(train_data)

        ################ 신경망 학습 ################

        BestModel_temp = model.fit(
            train_data,
            train_labels,
            epochs=Epoch,
            verbose=0,
            callbacks=[AccuracyPerEpoch()],
            batch_size=100,
            validation_data=(ValidData, ValidLabel),
        )
        print(
            "\n[Final Epochs]    RMSE:{:.5f},   MAE: {:.5f},  MAPE: {:.2f}%".format(
                np.sqrt(BestModel_temp.history["mse"][-1]),
                BestModel_temp.history["mae"][-1],
                BestModel_temp.history["mape"][-1],
            )
        )

        # 모델 저장
        model.save(
            "./27case_ANNmodels_kfold/BestModel_M%d_Fold%d.h5" % (M + 1, F + 1)
        )

        # 히스토리 저장
        RMSE = np.sqrt(np.array(BestModel_temp.history["mse"])[:, np.newaxis])
        MAE = np.array(BestModel_temp.history["mae"])[:, np.newaxis]
        MAPE = np.array(BestModel_temp.history["mape"])[:, np.newaxis]

        History_temp = pd.DataFrame(np.concatenate([RMSE, MAE, MAPE], axis=1))
        History_temp.to_csv(
            "./27case_ANNmodels_kfold/BestModel_M%d_Fold%d_history.csv"
            % (M + 1, F + 1),
            index=None,
        )

# %%
graph = pd.read_csv(
    "./27case_ANNmodels_kfold/BestModel_M1_Fold1_history.csv", sep=","
)
plt.plot(graph.iloc[:, 2])
plt.ylim(0, 20)

# %%
graph = pd.read_csv(
    "./27case_ANNmodels_kfold/BestModel_M1_Fold3_history.csv", sep=","
)
plt.plot(graph.iloc[:, 2])
plt.ylim(0, 20)

# %%
plt.plot(BestModel_temp.history["mape"])
plt.ylim(0, 20)
plt.xlabel("Epoch")
plt.ylabel("mape (%)")

# %%
for F in range(Fold):
    s = (
        "Model_Fold%d = keras.models.load_model('./27case_ANNmodels_kfold/BestModel_M1_Fold%d.h5')"
        % (F + 1, F + 1)
    )
    exec(s)


# %%
Model_Fold1.summary()

# %%
for i in range(Fold):
    s1 = "real2 = ValidLabel_Fold%d" % (i + 1)
    exec(s1)
    real3 = real2.sub(0.5)
    real4 = real3.mul(max_VS - min_VS)
    real = real4 + min_VS

    s2 = "predict2 = Model_Fold%d.predict(ValidData_Fold%d)" % (i + 1, i + 1)
    exec(s2)
    predict2 = pd.DataFrame(predict2)
    predict3 = predict2.sub(0.5)
    predict4 = predict3.mul(max_VS - min_VS)
    predict = predict4 + min_VS

    s3 = (
        "Result_Fold%d =  pd.DataFrame(np.concatenate((real,predict), axis = 1))"
        % (i + 1)
    )
    exec(s3)

# %%
for i in range(Fold):
    a1 = (
        "Error%d = pd.DataFrame(((Result_Fold%d.iloc[:,0]-Result_Fold%d.iloc[:,1])/Result_Fold%d.iloc[:,0])*100)"
        % (i + 1, i + 1, i + 1, i + 1)
    )
    exec(a1)
    a2 = "absError%d = np.abs(Error%d)" % (i + 1, i + 1)
    exec(a2)
    a3 = (
        "Result%d = pd.DataFrame(np.concatenate((Result_Fold%d,Error%d,absError%d),axis = 1))"
        % (i + 1, i + 1, i + 1, i + 1)
    )
    exec(a3)

Result_total = pd.concat([Result1, Result2, Result3, Result4, Result5])
Result_total.columns = ["Real", "Predict", "Error", "absError"]
pd.set_option("display.max_rows", None)
Result_total.reset_index(drop=True)

# %%
print("Average Error : ", np.mean(Result_total.iloc[:, 3]))

# %%
print("Average Error : ", np.mean(Result1.iloc[:, 3]))

# %%
print("Average Error : ", np.mean(Result2.iloc[:, 3]))

# %%
print("Average Error : ", np.mean(Result3.iloc[:, 3]))

# %%
print("Average Error : ", np.mean(Result4.iloc[:, 3]))

# %%
print("Average Error : ", np.mean(Result5.iloc[:, 3]))

# %%


# %%
a = 100 - 6.8180420596193505
a

# %%


# %% [markdown]
# # 전체 학습 후 train data로 검증

# %%
train_data = raw_data.iloc[:, :-2]

train_labels = pd.DataFrame(raw_data.iloc[:, 3])
train_data

# %%
train_labels

# %%
# TrainLabel = TrainLabel_before.add(0.5)
# TrainLabel

# %%
for M in range(1):
    Tr_result_temp = pd.read_csv(
        "./27case_ANN_prediction/Tr_result_epoch2000.csv", sep=","
    )
    learningRate = Tr_result_temp.sort_values(["MAPE"], ascending=True).iloc[
        0, 1
    ]
    noOfNeuron1 = np.int(
        Tr_result_temp.sort_values(["MAPE"], ascending=True).iloc[0, 2]
    )
    noOfNeuron2 = np.int(
        Tr_result_temp.sort_values(["MAPE"], ascending=True).iloc[0, 3]
    )
    Epoch = 10000

    print("\n\n\nPrediction :" + Model[M])
    print("Learning rate : {:.3}".format(learningRate))
    print("Hidden 1 neuron : %d" % (noOfNeuron1))
    print("Hidden 2 neuron : %d" % (noOfNeuron2))

    #     exec('Label_Trn = TrainLabel_%d'%(M+1))

    ################ 신경망 구조 재설계 ################

    tf.keras.backend.clear_session()

    def ANN_model(input_data):
        model = keras.Sequential()
        model.add(
            keras.layers.Dense(
                units=noOfNeuron_in,
                input_shape=(input_data.shape[1],),
                activation="relu",
            )
        )  # Input  Layer
        model.add(
            keras.layers.Dense(units=noOfNeuron1, activation="relu")
        )  # Hidden Layer 1
        model.add(
            keras.layers.Dense(units=noOfNeuron2, activation="relu")
        )  # Hidden Layer 2
        model.add(
            keras.layers.Dense(
                units=noOfNeuron_out,
            )
        )  # Output Layer
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learningRate),
            loss=keras.losses.mean_absolute_error,
            metrics=["mse", "mae", "mape"],
        )
        return model

    model = ANN_model(train_data)

    ################ 신경망 학습 ################

    BestModel_temp = model.fit(
        train_data,
        train_labels,
        epochs=Epoch,
        verbose=0,
        callbacks=[AccuracyPerEpoch()],
        batch_size=100,
    )
    print(
        "\n[Final Epochs]    RMSE:{:.5f},   MAE: {:.5f},  MAPE: {:.2f}%".format(
            np.sqrt(BestModel_temp.history["mse"][-1]),
            BestModel_temp.history["mae"][-1],
            BestModel_temp.history["mape"][-1],
        )
    )

    # 모델 저장
    model.save("./27case_ANNmodels_AllData/BestModel_M%d.h5" % (M + 1))

    # 히스토리 저장
    RMSE = np.sqrt(np.array(BestModel_temp.history["mse"])[:, np.newaxis])
    MAE = np.array(BestModel_temp.history["mae"])[:, np.newaxis]
    MAPE = np.array(BestModel_temp.history["mape"])[:, np.newaxis]

    History_temp = pd.DataFrame(np.concatenate([RMSE, MAE, MAPE], axis=1))
    History_temp.to_csv(
        "./27case_ANNmodels_AllData/BestModel_M%d_history.csv" % (M + 1),
        index=None,
    )

# %%


# %%
graph = pd.read_csv(
    "./27case_ANNmodels_AllData/BestModel_M1_history.csv", sep=","
)
plt.plot(graph.iloc[:, 2])
plt.ylim(0, 20)

# %%


# %%
Model_test = keras.models.load_model(
    "./27case_ANNmodels_AllData/BestModel_M1.h5"
)

# %%
real = TestLabel
predict2 = Model_test.predict(TestData)
predict3 = pd.DataFrame(predict2)
predict4 = predict3.sub(0.5)
predict5 = predict4.mul(max_VS - min_VS)
predict = predict5 + min_VS
Result_test = pd.DataFrame(np.concatenate((real, predict), axis=1))

# %%
Error = pd.DataFrame(
    (
        (Result_test.iloc[:, 0] - Result_test.iloc[:, 1])
        / Result_test.iloc[:, 0]
    )
    * 100
)
absError = np.abs(Error)
Result = pd.DataFrame(np.concatenate((Result_test, Error, absError), axis=1))

Result.columns = ["Real", "Predict", "Error", "absError"]
pd.set_option("display.max_rows", None)
Result

# %%
print("Average Error : ", np.mean(Result.iloc[:, 3]))

# %%
b = 100 - 0.2505445699074594
b

# %%


# %%


# %%
