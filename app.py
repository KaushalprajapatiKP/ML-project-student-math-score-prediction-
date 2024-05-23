from flask import Flask, request, render_template

from src.components.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

@app.route('/', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        data =  CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )

        pred_df = data.get_data_as_dataframe()
        print(pred_df)
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('index.html', results=results)
    return render_template('index.html')



if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True)