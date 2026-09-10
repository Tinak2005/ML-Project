import sys

from flask import Flask, request, render_template

from src.exception import CustomException
from src.pipelines.predict_pipeline import (
    CustomData,
    PredictPipeline
)


application = Flask(__name__)
app = application


# ==========================================
# HOME PAGE
# ==========================================

@app.route('/')
def index():

    return render_template('home.html')


# ==========================================
# PREDICTION ROUTE
# ==========================================

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():

    try:

        # ----------------------------------
        # GET REQUEST
        # ----------------------------------

        if request.method == 'GET':

            return render_template('home.html')


        # ----------------------------------
        # POST REQUEST
        # ----------------------------------

        else:

            # Create CustomData object
            data = CustomData(

                # ==========================
                # NUMERICAL FEATURES
                # ==========================

                hours_studied=float(
                    request.form.get('hours_studied')
                ),

                attendance=float(
                    request.form.get('attendance')
                ),

                previous_scores=float(
                    request.form.get('previous_scores')
                ),

                tutoring_sessions=float(
                    request.form.get('tutoring_sessions')
                ),

                sleep_hours=float(
                    request.form.get('sleep_hours')
                ),

                physical_activity=float(
                    request.form.get('physical_activity')
                ),


                # ==========================
                # CATEGORICAL FEATURES
                # ==========================

                motivation_level=request.form.get(
                    'motivation_level'
                ),

                access_to_resources=request.form.get(
                    'access_to_resources'
                ),

                parental_involvement=request.form.get(
                    'parental_involvement'
                ),

                internet_access=request.form.get(
                    'internet_access'
                ),

                teacher_quality=request.form.get(
                    'teacher_quality'
                ),

                parental_education_level=request.form.get(
                    'parental_education_level'
                ),

                school_type=request.form.get(
                    'school_type'
                ),

                extracurricular_activities=request.form.get(
                    'extracurricular_activities'
                ),

                peer_influence=request.form.get(
                    'peer_influence'
                ),

                family_income=request.form.get(
                    'family_income'
                )
            )


            # ==================================
            # CONVERT USER INPUT TO DATAFRAME
            # ==================================

            pred_df = data.get_data_as_data_frame()


            print("\nInput Data:")
            print(pred_df)


            # ==================================
            # INITIALIZE PREDICTION PIPELINE
            # ==================================

            predict_pipeline = PredictPipeline()


            # ==================================
            # GET ORIGINAL PREDICTION
            # ==================================

            results = predict_pipeline.predict(
                pred_df
            )


            prediction = round(
                float(results[0]),
                2
            )


            # ==================================
            # GET MODEL-BASED RECOMMENDATIONS
            # ==================================

            recommendations = (
                predict_pipeline.get_recommendations(
                    pred_df
                )
            )


            # ==================================
            # RETURN JSON RESPONSE
            # ==================================

            return {

                "prediction": prediction,

                "recommendations": recommendations

            }


    except Exception as e:

        print(
            "Backend Error:",
            str(e)
        )


        return {

            "prediction": None,

            "recommendations": [],

            "error": str(e)

        }, 500


# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        debug=True

    )