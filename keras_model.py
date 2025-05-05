import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.sparse import coo_matrix
import tensorflow as tf
from tensorflow import keras

### 1-1 데이터 불러오기
fires = pd.read_csv("./csv_data/sanbul2district-divby100.csv", sep=",")

### 1-2 데이터 요약 출력
print(fires.head())
print(fires.info())
print(fires.describe())
print(fires['month'].value_counts())
print(fires['day'].value_counts())

### 1-3 데이터 시각화
fires.hist(bins=50, figsize=(20,15))
plt.show()

### 1-4 로그 변환
fires['burned_area'] = np.log(fires['burned_area'] + 1)
fires.hist(bins=50, figsize=(20,15))
plt.show()

### 1-5 train/test split 
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(fires, test_size=0.2, random_state=42)

test_set.head()
fires["month"].hist()

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(fires, fires["month"]):
    strat_train_set = fires.loc[train_index]
    strat_test_set = fires.loc[test_index]

print("\nMonth category proportion: \n",
      strat_test_set["month"].value_counts() / len(strat_test_set))
print("\nOverall month category proportion: \n",
      fires["month"].value_counts() / len(fires))


### 1-6 scatter matrix
attributes = ["burned_area", "max_temp", "avg_temp", "max_wind_speed"]
scatter_matrix(fires[attributes], figsize=(12,8))
plt.show()

### 1-7 지역별 burned_area에 대해 plot 하기
fires.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
           s=fires["max_temp"], label="max_temp",
           c="burned_area", cmap=plt.get_cmap("jet"), colorbar=True)
plt.legend()
plt.show()

### 1-8 OneHotEncoder 실험 (평가 요소)
fires = strat_train_set.drop(["burned_area"], axis=1) #정답 데이터를 제외한 데이터
fires_labels = strat_train_set["burned_area"].copy() #정답 데이터
fires_num = fires.drop(["month", "day"], axis=1) #숫자형 데이터

fires_month = strat_train_set[["month"]]
fires_day = strat_train_set[["day"]]

cat_month_encoder = OneHotEncoder()
cat_day_encoder = OneHotEncoder()

fires_month_1hot = cat_month_encoder.fit_transform(fires_month)
fires_day_1hot = cat_day_encoder.fit_transform(fires_day)

print("\n=== 1-8 OneHotEncoder results ===")
print("cat_month_encoder.categories_:")
print(fires_month_1hot.toarray())
print(cat_month_encoder.categories_)

print("cat_day_encoder.categories_:")
print(fires_day_1hot.toarray())
print(cat_day_encoder.categories_)

# COO format
fires_month_1hot_coo = coo_matrix(fires_month_1hot)
fires_day_1hot_coo = coo_matrix(fires_day_1hot)

print("\nfires_month_1hot (COO):")
for row, col, value in zip(fires_month_1hot_coo.row, fires_month_1hot_coo.col, fires_month_1hot_coo.data):
    print(f"({row}, {col}) {value}")

print("\nfires_day_1hot (COO):")
for row, col, value in zip(fires_day_1hot_coo.row, fires_day_1hot_coo.col, fires_day_1hot_coo.data):
    print(f"({row}, {col}) {value}")


### 1-9 Pipeline + StandardScaler
print("\n\n#########################################################")
print("Now let's build a pipline for preprocessing the numerical attributes:")
num_attribs = list(fires_num)
cat_attribs = ["month", "day"]

num_pipeline = Pipeline([('std_scaler', StandardScaler())])
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs)
])


#훈련용 데이터 파이프라인 적용
fires_prepared = full_pipeline.fit_transform(fires) #훈련데이터 입력값 X들

#테스트용 데이터 파이프라인 적용
test_fires = strat_test_set.drop(["burned_area"], axis=1)
fires_test_prepared = full_pipeline.transform(test_fires) #성능 평가용 입력값
fires_test_labels = strat_test_set["burned_area"].copy()








### STEP 2: Keras 모델
#keras model 개발
X_train, X_valid, y_train, y_valid = train_test_split(
    fires_prepared, fires_labels, test_size=0.2, random_state=42)
X_test, y_test = fires_test_prepared, fires_test_labels

np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.Dense(30, activation='relu', input_shape=X_train.shape[1:]),
    keras.layers.Dense(30, activation='relu', input_shape=X_train.shape[1:]),
    keras.layers.Dense(30, activation='relu', input_shape=X_train.shape[1:]),
    keras.layers.Dense(1)
])

model.summary()

model.compile(loss='mean_squared_error', optimizer=keras.optimizers.SGD(learning_rate=1e-3))
history = model.fit(X_train, y_train, epochs=200, validation_data=(X_valid, y_valid))

#keras 모델 저장
model.save("fires_model.keras")

#evaluate model
X_new = X_test[:3]
predictions = np.round(model.predict(X_new), 2)
print("\nnp.round(model.predict(X_new), 2) : \n", 
      predictions)
#모델 예측 비교를 위한 실제 정답 확인
print("\nActual values (log scale)\n:", y_test[:3])



### m² 형식으로 비교하기
#m² 형식으로 변환해서 출력
preds_log = predictions.flatten()
actuals_log = y_test[:3].values

# np.expm1(): log(x + 1) 변환 복원
preds_m2 = np.expm1(preds_log)
actuals_m2 = np.expm1(actuals_log)

print("\n=== Predictions vs Actuals (m² scale) ===")
for pred, actual in zip(preds_m2, actuals_m2):
    print(f"Predicted: {pred:.2f} m², Actual: {actual:.2f} m²")
