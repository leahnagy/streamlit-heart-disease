import streamlit as st
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

st.write("""
	# Heart Disease Prediction

	#### This app predicts the probability of the presence of heart disease.

	### Instructions:
	Answer the questions in the sidebar. Your probability of heart disease will be displayed. 

	""")

def user_input_features():
	HighBP = st.sidebar.selectbox("Have you been told by a health practitioner that you have high blood pressure?", ["yes", "no"])
	if HighBP == "yes":
		HighBP = 1
	else:
		HighBP = 0 

	HighChol = st.sidebar.selectbox("Have you been told by a health practitioner that you have high cholesterol?", ["yes", "no"])
	if HighChol == "yes":
		HighChol = 1
	else:
		HighChol = 0 

	CholCheck = st.sidebar.selectbox("Have you had your cholesterol checked within the last 5 years?", ["yes", "no"])
	if CholCheck == "yes":
		CholCheck = 1
	else:
		CholCheck = 0 

	BMI = st.sidebar.slider("What is your BMI?", 1, 99)


	Smoker = st.sidebar.selectbox("Have you smoked at least 100 cigarettes in your entire life?", ["yes", "no"])
	if Smoker == "yes":
		Smoker = 1
	else:
		Smoker = 0 

	Stroke = st.sidebar.selectbox("Have you been told by a health practitioner that you had a stroke?", ["yes", "no"])
	if Stroke == "yes":
		Stroke = 1
	else:
		Stroke = 0 

	Diabetes = st.sidebar.selectbox("Have you been told by a health practitioner that you have diabetes?", ["yes", "prediabetes", "no, or only during pregnancy"])
	if Diabetes == "yes":
		Diabetes = 2
	if Diabetes == "prediabetes":
		Diabetes =1
	else:
		Diabetes = 0 

	PhysActivity = st.sidebar.selectbox("Have you performed any physical activities or exercise in the past 30 days \
		other than your regular job?", ["yes", "no"])
	if PhysActivity == "yes":
		PhysActivity = 1
	else:
		PhysActivity = 0 

	Fruits = st.sidebar.selectbox("Do you regularly consume fruit 1 or more times per day?", ["yes", "no"])
	if Fruits == "yes":
		Fruits = 1
	else:
		Fruits = 0 

	Veggies = st.sidebar.selectbox("Do you regularly consume vegetables 1 or more times per day?", ["yes", "no"])
	if Veggies == "yes":
		Veggies = 1
	else:
		Veggies = 0 

	HvyAlcoholConsump = st.sidebar.selectbox("Do you regularly consume more than 14 alcoholic drinks per week (males) or \
		more than 7 alcoholic drinks per week (females)?", ["yes", "no"])
	if HvyAlcoholConsump == "yes":
		HvyAlcoholConsump = 1
	else:
		HvyAlcoholConsump = 0 

	AnyHealthcare = st.sidebar.selectbox("Do you have any kind of health coverage, including health insurance,\
		 or goveernment plans such as Medicare?", ["yes", "no"])
	if AnyHealthcare == "yes":
		AnyHealthcare = 1
	else:
		AnyHealthcare = 0 

	NoDocbcCost = st.sidebar.selectbox("Was there a time in the past 12 months when you needed to see a doctor but \
		could not because of the cost?", ["yes", "no"])
	if NoDocbcCost == "yes":
		NoDocbcCost = 1
	else:
		NoDocbcCost = 0 

	GenHlth = st.sidebar.selectbox("How would you rate your health in general?", ["Excellent", "Very Good", \
		"Good", "Fair", "Poor"])
	if GenHlth == "Excellent":
		GenHlth = 1
	elif GenHlth == "Very Good":
		GenHlth = 2
	elif GenHlth == "Good":
		GenHlth = 3 
	elif GenHlth == "Fair":
		GenHlth = 4 
	else:
		GenHlth = 5 

	MentHlth = st.sidebar.slider("How many days during the past 30 days was your mental health not good \
		where mental health includes stress, depression and problems with emotions?", 0, 30)

	PhysHlth =st.sidebar.slider("How many days during the past 30 days was your physical health not good \
		where physical health includes ilnness and injury?", 0, 30)

	DiffWalk = st.sidebar.selectbox("Do you have difficulty walking or climbing stairs?", ["yes", "no"]) 
	if DiffWalk == "yes":
		DiffWalk = 1
	else:
		DiffWalk = 0 

	Sex = st.sidebar.selectbox("What is your gender", ["Male", "Female"])
	if Sex == "Male":
		Sex = 1
	else:
		Sex = 0 

	Age = st.sidebar.slider("What is your age", 18, 120)

	Education = st.sidebar.selectbox("What is the highest grade or year of school you completed?",\
		["Never attended school or only kindergarten",\
		 "Grades 1 through 8",\
		 "Grades 9 through 11",\
		 "Grades 12 or GED",\
		 "College 1 year to 3 years",\
		 "College 4 years or more"])
	if Education == "Never attended school or only kindergarten":
		Education = 1
	elif Education == "Grades 1 through 8":
		Education = 2 
	elif Education == "Grades 9 through 11":
		Education = 3 
	elif Education == "Grades 12 or GED":
		Education = 4
	elif Education == "College 1 year to 3 years":
		Education = 5 
	else:
		Education = 6 

	Income = st.sidebar.selectbox("what is your annual household income from all sources?",\
		 ["Less than $10,000",\
		 "$10,000 - $14,999",\
		 "$15,000 - $19,999",\
		 "$20,000 - $24,999",\
		 "$25,000 - $34,999",\
		 "$35,000 - $49,999",\
		 "$50,000 - $74,999",\
		 "$75,000 +"])
	if Income == "Less than $10,000":
		Income = 1
	elif Income == "$10,000 - $14,999":
		Income = 2 
	elif Income == "$15,000 - $19,999":
		Income = 3 
	elif Income == "$20,000 - $24,999":
		Income = 4
	elif Income == "$25,000 - $34,999":
		Income = 5 
	elif Income == "$35,000 - $49,999":
		Income = 6 
	elif Income == "$50,000 - $74,999":
		Income = 7 
	else:
		Income = 8 

	data = {'HighBP': HighBP,
            'HighChol': HighChol,
            'CholCheck': CholCheck,
            'BMI': BMI,
            'Smoker' : Smoker,
            'Stroke' : Stroke,
            'Diabetes' : Diabetes,
            'PhysActivity' : PhysActivity,
            'Fruits' : Fruits,
            'Veggies' : Veggies,
            'HvyAlcoholConsump' : HvyAlcoholConsump,
            'AnyHealthcare' : AnyHealthcare,
            'NoDocbcCost' : NoDocbcCost,
            'GenHlth' : GenHlth,
            'MentHlth' : MentHlth,
            'PhysHlth' : PhysHlth,
            'DiffWalk' : DiffWalk,
            'Sex' : Sex,
            'Age' : Age,
            'Education' : Education,
            'Income' : Income}
	features = pd.DataFrame(data, index=[0])
	return features

st.sidebar.header('User Health Input')
df = user_input_features()

dataset = pd.read_csv('heart_disease_2015.csv')
X, y = dataset.drop('HeartDiseaseorAttack', axis=1), dataset['HeartDiseaseorAttack']

####################
### Scaling Data ###
####################

# Continuous Variables
numeric_features = ['MentHlth', 'PhysHlth', 'BMI']
numeric_transformer = Pipeline(
    steps=[('imputer', SimpleImputer(strategy="median")), 
           ('scaler', StandardScaler())])

# Categorical Variables
categorical_features = ['Age', 'Education', 'Income', 
                        'GenHlth', 'Diabetes']

# instantiate encoder
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])


###################
###### Model ######
###################

model = Pipeline(steps=[('preprocessor', preprocessor), 
                      ('classifier', LogisticRegression(class_weight={0: 0.08457286432160804, 1: 0.915427135678392}, 
                                                        C=10))])
model.fit(X, y)
prediction = model.predict(df)
prediction_proba = model.predict_proba(df)[:, 1]

###################
### Predictions ###
###################

prediction = (model.predict_proba(df)[:, 1] >= 0.5).astype(int)
if prediction == 1:
	predict_text = "You may be at risk for heart disease. Please speak to your healthcare \
	practitioner to discuss how to reduce your risk."
else:
	predict_text = "You are not at risk for heart disease. Please speak to your healthcare \
	practitioner to learn more."

st.subheader('Prediction')
# st.write(predict_text)
st.write(predict_text)

st.subheader('Prediction Probability')
st.write(prediction_proba)

st.write("""
	## *****Important*****
	*The probability is not indicative of a diagnosis.* This number is used as part of a Heart Disease
	prevention program to be established with your healthcare practitioner."
	""")

