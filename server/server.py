from flask import Flask, request, jsonify
import utils

app = Flask(__name__)

@app.route("/stroke_likelihood", methods = ['POST'])
def stroke_likelihood():
    age = float(request.form['age'])
    bmi = float(request.form['bmi'])
    glucose = float(request.form['avg_glucose_level'])
    gender = int(request.form['gender'])
    rt = int(request.form['residenceType'])
    em = int(request.form['everMarried'])
    hypertension = int(request.form['hypertension'])
    hi = int(request.form['heartIssues'])
    wt = int(request.form['workType'])
    smoking = int(request.form['smoking'])
    data = utils.get_stroke_likelihood(age = age, hypertension = hypertension, heart_disease = hi, avg_glucose_level = glucose, bmi= bmi, gender = gender, married = em, work_type=wt, residence=rt, smoking=smoking)
    response = jsonify({
        'likelihood' : data
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route("/jaijawan", methods = ['GET', 'POST'])
def jaijawan():
    print("Aa gaya hu mai")
    return jsonify('HHIII')


if __name__ == "__main__":
    print("Starting Python Server")
    utils.load_saved_artifacts()
    app.run(debug=True)