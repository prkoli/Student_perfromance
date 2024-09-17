from flask import Flask, request, render_template
from model import predict_performance

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        gender = request.form.get('gender')
        ethnicity = request.form.get('ethnicity')
        parental_education = request.form.get('parental_education')
        lunch = request.form.get('lunch')
        test_preparation = request.form.get('test_preparation')

        result = predict_performance(gender, ethnicity, parental_education, lunch, test_preparation)
        
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
