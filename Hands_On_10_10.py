from transformers import pipeline
from rdflib import Graph, URIRef, Literal, Namespace
import spacy

# Load pre-trained Named Entity Recognition (NER) model
nlp_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

# Example medical report text related to diabetes
medical_text = "John's blood sugar level is 180 mg/dL and his BMI is 32. He should exercise more to manage his condition."

# Extract entities from the medical report
entities = nlp_pipeline(medical_text)
print("Extracted Entities:")
for entity in entities:
    print(entity)

# Initialize RDF graph and define namespaces
g = Graph()
snomed = Namespace("http://snomed.info/id/")
loinc = Namespace("http://loinc.org/")
ex = Namespace("http://example.org/")

# Example of mapping extracted entities to SNOMED CT ontology
john = URIRef(ex.John)
g.add((john, snomed['hasCondition'], Literal("High blood sugar")))
g.add((john, loinc['BMI'], Literal("32")))
g.add((john, snomed['needsExercise'], Literal("Yes")))

# Serialize the graph in Turtle format
# Serialize the graph in Turtle format
print("Knowledge Graph in Turtle format:")
print(g.serialize(format='turtle'))

# Define a SPARQL query to retrieve patients with high BMI and blood sugar levels
query = """
    SELECT ?person ?condition ?bmi WHERE {
        ?person <http://snomed.info/id/hasCondition> ?condition .
        ?person <http://loinc.org/BMI> ?bmi .
        FILTER (?bmi > 30)
    }
"""
# Execute the query and print results
print("Query Results:")
results = g.query(query)
for row in results:
    print(f"Person: {row.person}, Condition: {row.condition}, BMI: {row.bmi}")

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# Example patient data (BMI and blood sugar levels)
patient_data = pd.DataFrame({
    'BMI': [28, 35, 30, 33],  # BMI values
    'BloodSugar': [120, 190, 145, 170]  # Blood sugar levels in mg/dL
})

# Labels for diabetes risk (1 = high risk, 0 = low risk) - dummy data
diabetes_risk_labels = [0, 1, 0, 1]

# Train a logistic regression model to predict diabetes risk
model = LogisticRegression()
model.fit(patient_data[['BMI', 'BloodSugar']], diabetes_risk_labels)

# Predict diabetes risk for a new patient
new_patient_data = pd.DataFrame({'BMI': [32], 'BloodSugar': [180]})
predicted_risk = model.predict(new_patient_data)[0]
print(f"Predicted Diabetes Risk: {'High' if predicted_risk == 1 else 'Low'}")

from owlready2 import *

# Create a new ontology
onto = get_ontology("http://example.org/diabetes_prevention.owl")

# Define classes and data properties
with onto:
    class Person(Thing):
        pass

    # Define data properties
    class hasCondition(Person >> str, DataProperty):
        pass

    class hasBMI(Person >> float, DataProperty):
        pass

    class recommendExercise(Person >> str, DataProperty):
        pass

# Create an individual and assign data property values using append()
john = Person("John_Doe")
john.hasCondition.append("High blood sugar")
john.hasBMI.append(32.0)
john.recommendExercise.append("Yes")

# Save the ontology to an .owl file
onto.save(file="diabetes_prevention.owl", format="rdfxml")

print("Ontology saved successfully!")



# Load the ontology
onto = get_ontology("diabetes_prevention.owl").load()

# Define the rule
swrl_rule = """
Person(?p) ^ hasCondition(?p, "High blood sugar") ^ hasBMI(?p, ?bmi) ^ 
greaterThan(?bmi, 30) -> recommendExercise(?p)
"""

# Add the rule to the ontology
with onto:
    new_rule = Imp()
    new_rule.set_as_rule(swrl_rule)

# Save or query ontology
onto.save(file="updated_ontology.owl", format="rdfxml")

# Query the ontology for individuals who need exercise recommendations
for person in onto.individuals():
    if person.recommendExercise:
        print(f"Recommend exercise for {person.name}")


