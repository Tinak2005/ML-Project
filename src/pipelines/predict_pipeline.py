import os
import sys
import pandas as pd

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:

    def __init__(self):
        pass


    def load_model_and_preprocessor(self):

        model_path = os.path.join(
            "artifacts",
            "model.pkl"
        )

        preprocessor_path = os.path.join(
            "artifacts",
            "preprocessor.pkl"
        )

        model = load_object(
            file_path=model_path
        )

        preprocessor = load_object(
            file_path=preprocessor_path
        )

        return model, preprocessor


    def predict(self, features):

        try:

            model, preprocessor = (
                self.load_model_and_preprocessor()
            )

            # Transform input data
            transformed_data = preprocessor.transform(
                features
            )

            # Prediction
            preds = model.predict(
                transformed_data
            )

            return preds

        except Exception as e:

            raise CustomException(e, sys)


    def get_recommendations(self, features):

        """
        Performs realistic what-if analysis on all features.

        Each feature is changed individually and the trained
        ML model predicts the possible score impact.
        """

        try:

            # Baseline prediction

            current_prediction = float(
                self.predict(features)[0]
            )


            recommendations = []


            # ==========================================
            # REALISTIC SCENARIOS FOR NUMERICAL FEATURES
            # ==========================================

            numerical_scenarios = {

                "Hours_Studied": [
                    features.iloc[0]["Hours_Studied"] + 1,
                    features.iloc[0]["Hours_Studied"] + 2
                ],

                "Attendance": [
                    min(
                        features.iloc[0]["Attendance"] + 5,
                        100
                    ),

                    min(
                        features.iloc[0]["Attendance"] + 10,
                        100
                    )
                ],

                "Previous_Scores": [
                    min(
                        features.iloc[0]["Previous_Scores"] + 5,
                        100
                    ),

                    min(
                        features.iloc[0]["Previous_Scores"] + 10,
                        100
                    )
                ],

                "Tutoring_Sessions": [
                    features.iloc[0]["Tutoring_Sessions"] + 1,
                    features.iloc[0]["Tutoring_Sessions"] + 2
                ],

                # Sleep is tested around realistic values
                "Sleep_Hours": [
                    7,
                    8
                ],

                # Physical activity tested realistically
                "Physical_Activity": [
                    1,
                    2,
                    3
                ]
            }


            # ==========================================
            # CHECK NUMERICAL FEATURES
            # ==========================================

            for feature, values in numerical_scenarios.items():

                original_value = (
                    features.iloc[0][feature]
                )

                best_prediction = current_prediction
                best_value = original_value


                for value in values:

                    # Don't test same value

                    if value == original_value:
                        continue


                    # Create copy

                    test_data = features.copy()


                    # Change only one feature

                    test_data.loc[
                        test_data.index[0],
                        feature
                    ] = value


                    # Model prediction

                    new_prediction = float(
                        self.predict(test_data)[0]
                    )


                    # Keep best scenario

                    if new_prediction > best_prediction:

                        best_prediction = new_prediction

                        best_value = value


                improvement = (
                    best_prediction
                    - current_prediction
                )


                # Only meaningful improvements

                if improvement >= 0.5:

                    recommendations.append({

                        "factor": feature,

                        "current_value":
                            float(original_value),

                        "recommended_value":
                            float(best_value),

                        "impact":
                            round(improvement, 2)

                    })


            # ==========================================
            # CATEGORICAL FEATURES
            # ==========================================

            categorical_scenarios = {

                "Motivation_Level":
                    ["Low", "Medium", "High"],

                "Access_to_Resources":
                    ["Low", "Medium", "High"],

                "Parental_Involvement":
                    ["Low", "Medium", "High"],

                "Internet_Access":
                    ["Yes", "No"],

                "Teacher_Quality":
                    ["Low", "Medium", "High"],

                "Parental_Education_Level":
                    [
                        "High School",
                        "College",
                        "Postgraduate"
                    ],

                "School_Type":
                    ["Public", "Private"],

                "Extracurricular_Activities":
                    ["Yes", "No"],

                "Peer_Influence":
                    [
                        "Negative",
                        "Neutral",
                        "Positive"
                    ],

                "Family_Income":
                    ["Low", "Medium", "High"]
            }


            # ==========================================
            # CHECK CATEGORICAL FEATURES
            # ==========================================

            for feature, values in categorical_scenarios.items():

                original_value = (
                    features.iloc[0][feature]
                )

                best_prediction = current_prediction
                best_value = original_value


                for value in values:

                    if value == original_value:
                        continue


                    test_data = features.copy()


                    # Change one factor

                    test_data.loc[
                        test_data.index[0],
                        feature
                    ] = value


                    new_prediction = float(
                        self.predict(test_data)[0]
                    )


                    if new_prediction > best_prediction:

                        best_prediction = new_prediction

                        best_value = value


                improvement = (
                    best_prediction
                    - current_prediction
                )


                # Meaningful impact only

                if improvement >= 0.5:

                    recommendations.append({

                        "factor": feature,

                        "current_value":
                            original_value,

                        "recommended_value":
                            best_value,

                        "impact":
                            round(improvement, 2)

                    })


            # ==========================================
            # SORT BY HIGHEST IMPACT
            # ==========================================

            recommendations = sorted(

                recommendations,

                key=lambda x: x["impact"],

                reverse=True

            )


            return recommendations


        except Exception as e:

            raise CustomException(e, sys)



# ==================================================
# CUSTOM DATA CLASS
# ==================================================

class CustomData:

    def __init__(

        self,

        # Numerical Features

        hours_studied,
        attendance,
        previous_scores,
        tutoring_sessions,
        sleep_hours,
        physical_activity,


        # Categorical Features

        motivation_level,
        access_to_resources,
        parental_involvement,
        internet_access,
        teacher_quality,
        parental_education_level,
        school_type,
        extracurricular_activities,
        peer_influence,
        family_income

    ):


        # Numerical

        self.Hours_Studied = hours_studied
        self.Attendance = attendance
        self.Previous_Scores = previous_scores
        self.Tutoring_Sessions = tutoring_sessions
        self.Sleep_Hours = sleep_hours
        self.Physical_Activity = physical_activity


        # Categorical

        self.Motivation_Level = motivation_level
        self.Access_to_Resources = access_to_resources
        self.Parental_Involvement = parental_involvement
        self.Internet_Access = internet_access
        self.Teacher_Quality = teacher_quality

        self.Parental_Education_Level = (
            parental_education_level
        )

        self.School_Type = school_type

        self.Extracurricular_Activities = (
            extracurricular_activities
        )

        self.Peer_Influence = peer_influence

        self.Family_Income = family_income


    def get_data_as_data_frame(self):

        try:

            custom_data_input_dict = {

                "Hours_Studied":
                    [self.Hours_Studied],

                "Attendance":
                    [self.Attendance],

                "Previous_Scores":
                    [self.Previous_Scores],

                "Tutoring_Sessions":
                    [self.Tutoring_Sessions],

                "Sleep_Hours":
                    [self.Sleep_Hours],

                "Physical_Activity":
                    [self.Physical_Activity],


                "Motivation_Level":
                    [self.Motivation_Level],

                "Access_to_Resources":
                    [self.Access_to_Resources],

                "Parental_Involvement":
                    [self.Parental_Involvement],

                "Internet_Access":
                    [self.Internet_Access],

                "Teacher_Quality":
                    [self.Teacher_Quality],

                "Parental_Education_Level":
                    [self.Parental_Education_Level],

                "School_Type":
                    [self.School_Type],

                "Extracurricular_Activities":
                    [self.Extracurricular_Activities],

                "Peer_Influence":
                    [self.Peer_Influence],

                "Family_Income":
                    [self.Family_Income]

            }


            return pd.DataFrame(
                custom_data_input_dict
            )


        except Exception as e:

            raise CustomException(e, sys)