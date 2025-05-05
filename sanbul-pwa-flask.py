import tensorflow as tf
from tensorflow import keras

print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)

import numpy as np
import pandas as pd
from flask import Flask, render_template

from flask_bootstrap import Bootstrap5
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

STRING_FIELD = StringField('max_wind_speed', validators=[DataRequired()])
np.random.seed(42)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'
bootstrap5 = Bootstrap5(app)

class LabForm(FlaskForm):
    longitude = StringField('longitude(1-7)', validators=[DataRequired()])
    latitude = StringField('latitude(1-7)', validators=[DataRequired()])
    month = StringField('month(01-Jan ~ Dec-12)', validators=[DataRequired()])
    day = StringField('day(00-sun ~ 06-sat, 07-hol)', validators=[DataRequired()])
    avg_temp = StringField('avg_temp', validators=[DataRequired()])
    max_temp = StringField('max_temp', validators=[DataRequired()])
    max_wind_speed = StringField('max_wind_speed', validators=[DataRequired()])
    avg_wind = StringField('avg_wind', validators=[DataRequired()])
    submit = SubmitField('Submit')

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['GET', 'POST'])
def lab():
    form = LabForm()
    prediction = None

    if form.validate_on_submit():
        # 입력값 DataFrame으로 준비
        input_data = pd.DataFrame([[  
            float(form.longitude.data),
            float(form.latitude.data),
            form.month.data,
            form.day.data,
            float(form.avg_temp.data),
            float(form.max_temp.data),
            float(form.max_wind_speed.data),
            float(form.avg_wind.data)
        ]], columns=['longitude', 'latitude', 'month', 'day', 'avg_temp', 'max_temp', 'max_wind_speed', 'avg_wind'])

        # pipeline용 데이터셋 불러오기
        data = pd.read_csv("./csv_data/sanbul2district-divby100.csv", sep=",")
        num_attribs = ['longitude', 'latitude', 'avg_temp', 'max_temp', 'max_wind_speed', 'avg_wind']
        cat_attribs = ['month', 'day']

        full_pipeline = ColumnTransformer([
            ('num', StandardScaler(), num_attribs),
            ('cat', OneHotEncoder(), cat_attribs)
        ])
        full_pipeline.fit(data[num_attribs + cat_attribs])

        # 입력 데이터 transform
        X_prepared = full_pipeline.transform(input_data)

        # 모델 불러오기
        model = keras.models.load_model("fires_model.keras")

        # 예측 + 로그 역변환
        pred_log = model.predict(X_prepared)
        pred_value = np.expm1(pred_log)[0][0]
        prediction = round(pred_value, 2)

        return render_template('result.html', res=prediction)

    return render_template('prediction.html', form=form)


if __name__ == '__main__':
    app.run()