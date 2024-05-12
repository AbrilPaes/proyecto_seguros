import pandas as pd
import numpy as np
import datetime
import random
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib


"""
Se genera un conjunto de datos simulados que represente la actividad de una 
aseguradora de coches
"""
# Función para generar fechas aleatorias dentro de un rango
def generate_random_date(start_date, end_date):
    days_difference = (end_date - start_date).days
    random_days = random.randint(0,days_difference)
    fech = start_date + datetime.timedelta(days=random_days)
    return fech

# Generar datos simulados
def generate_insurance_data(n_cases):
    # Semilla para reproducibilidad
    random.seed(42)

    # Fechas de inicio y vencimiento de la cobertura desde 2022 a 2025
    start_date_coverage = datetime.datetime(2022, 1, 1)
    end_date_coverage = datetime.datetime(2025, 12, 31)

    # Tipos de cobertura
    coverage_types = ['Responsabilidad civil', 'Cobertura total', 'Cobertura de colisión', 'Cobertura amplia', 'Cobertura de robo']

    # Modelos de coches y probabilidades de ocurrencia
    car_models = ['Toyota Corolla', 'Honda Civic', 'Ford Focus', 'Chevrolet Cruze', 'Nissan Sentra',
                  'Hyundai Elantra', 'Volkswagen Jetta', 'Kia Forte', 'Mazda 3', 'Subaru Impreza']

    probabilities = [0.2, 0.15, 0.12, 0.1, 0.08, 0.1, 0.08, 0.07, 0.05, 0.05]

    # Generar datos simulados
    data = {
        'Número de póliza': ['P' + str(i).zfill(5) for i in range(1, n_cases + 1)],
        'Fecha de inicio': [generate_random_date(start_date_coverage, end_date_coverage) for _ in range(n_cases)],
        'Fecha de vencimiento': [generate_random_date(start_date_coverage, end_date_coverage) for _ in range(n_cases)],
        'Tipo de cobertura': np.random.choice(coverage_types, n_cases),
        'Modelo del coche': np.random.choice(car_models, n_cases, p=probabilities),
        'Año del coche': np.random.randint(2010, 2023, n_cases),
        'Valor asegurado': np.random.randint(10000, 50001, n_cases),
        'Deducible': np.random.choice([500, 600, 700], n_cases),
        'Estado del seguro': np.random.choice(['Al día', 'Vencido'], n_cases),
        'Gastos médicos': np.random.choice(2, size=n_cases),
        'Daños a terceros': np.random.choice(2, size=n_cases)
    }
    #
    # Crear DataFrame
    df = pd.DataFrame(data)
    # Agregar columna 'Año' a partir de 'Fecha de inicio'
    df['Año'] = df['Fecha de inicio'].dt.year
    return df

# Función para mostrar el DataFrame
def show_data_frame(df):
    st.subheader('Datos Simulados')
    st.write(df.head(6))

#Gráfico de barras para mostrar la cantidad de siniestros por año.
# Función para mostrar el gráfico de barras
"""

"""
def show_bar_plot(df):
    siniestros_por_año = df.groupby('Año').size().reset_index(name='Cantidad de siniestros')
    siniestros_por_año = siniestros_por_año.sort_values(by='Cantidad de siniestros', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.countplot(y='Año',hue='Año', data=df, palette='magma', dodge=False, order=siniestros_por_año['Año'],legend=False)
    plt.title('Tendencia temporal de la cantidad de siniestros')
    plt.xlabel('Cantidad de siniestros')
    plt.ylabel('Año')
    st.pyplot(plt)

def main():
    st.title('Análisis de Seguros de Automóviles')

    # Generar los datos simulados
    num_policies = st.sidebar.number_input("Número de pólizas", 10000, 50001)
    df = generate_insurance_data(num_policies)

    # Mostrar el DataFrame
    show_data_frame(df)

    # Mostrar el gráfico de barras
    st.subheader('Análisis: Tipos de coches con más siniestros')
    siniestros_por_coche = df.groupby('Modelo del coche').size().reset_index(name='Cantidad de siniestros')
    siniestros_por_coche = siniestros_por_coche.sort_values(by='Cantidad de siniestros', ascending=False)
    st.write(siniestros_por_coche)

    plt.figure(figsize=(12, 6))
    sns.countplot(x='Modelo del coche', data=df, order=df['Modelo del coche'].value_counts().index, color='red')
    plt.title('Tipos de coches con más siniestros')
    plt.xlabel('Modelo')
    plt.ylabel('Cantidad de siniestros')
    plt.xticks(rotation=45)
    st.pyplot(plt)

    st.subheader('Análisis: Tendencia temporal de la cantidad de siniestros')
    show_bar_plot(df)

    # Análisis por mes
    st.subheader('Análisis: Meses con más siniestros')
    df['Fecha de inicio'] = pd.to_datetime(df['Fecha de inicio'])
    df['Mes'] = df['Fecha de inicio'].dt.month
    siniestros_por_mes = df.groupby('Mes').size().reset_index(name='Cantidad de siniestros')

    plt.figure(figsize=(10, 6))
    sns.countplot(x='Mes',hue='Mes' ,data=df, palette='plasma',legend=False)
    plt.title('Tendencia temporal de la cantidad de siniestros por mes')
    plt.xlabel('Mes')
    plt.ylabel('Cantidad de siniestros')
    st.pyplot(plt)

    # Análisis por tipo de cobertura
    st.subheader('Análisis: Cantidad de siniestros por tipo de cobertura')
    siniestros_por_cobertura = df['Tipo de cobertura'].value_counts()
    analisis = pd.DataFrame(columns=['Tipo de cobertura', 'Cantidad de siniestros'])
    analisis['Tipo de cobertura'] = siniestros_por_cobertura.index
    analisis['Cantidad de siniestros'] = siniestros_por_cobertura.values
    st.write(analisis)

    plt.figure(figsize=(10, 6))
    siniestros_por_cobertura.plot(kind='line', marker='o', color='#F08080')
    plt.title('Cantidad de siniestros por tipo de cobertura')
    plt.xlabel('Tipo de cobertura')
    plt.ylabel('Cantidad de siniestros')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)

    # Modelo predictivo
    st.title('Predicción de Gastos Médicos')
    st.write('Ingrese los detalles del seguro para predecir si habrá gastos médicos.')

    # Codificación de variables categóricas
    df_encoded = pd.get_dummies(df.drop(['Número de póliza', 'Fecha de inicio', 'Fecha de vencimiento'], axis=1))

    # División de datos en entrenamiento y prueba
    X = df_encoded.drop('Gastos médicos', axis=1)
    y = df_encoded['Gastos médicos']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenamiento del modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Guardar el modelo en un archivo .pkl
    joblib.dump(model, 'modelo_seguro.pkl')

    # Evaluación del modelo
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"\nPrecisión del modelo: {accuracy:.2f}")

    #Entrada de datos
    form = st.form(key='insurance_form')
    coverage_type = form.selectbox('Tipo de cobertura', df['Tipo de cobertura'].unique())
    car_model = form.selectbox('Modelo del coche', df['Modelo del coche'].unique())
    car_year = form.number_input('Año del coche', min_value=2010, max_value=2022)
    insured_value = form.number_input('Valor asegurado', min_value=10000, max_value=50000)
    deductible = form.select_slider('Deducible', options=[500, 600, 700])
    insurance_state = form.selectbox('Estado del seguro', ['Al día', 'Vencido'])
    third_party_damage = form.select_slider('Daños a terceros', options=[0, 1])
    submit_button = form.form_submit_button(label='Predecir')

    # Preprocesamiento de datos de entrada
    # Codificar variables categóricas
    label_encoder = LabelEncoder()
    df['Tipo de cobertura'] = label_encoder.fit_transform(df['Tipo de cobertura'])
    df['Modelo del coche'] = label_encoder.fit_transform(df['Modelo del coche'])

    # Preprocesamiento de datos de entrada
    if coverage_type == 'Responsabilidad civil':
        coverage_type_encoded = 0
    elif coverage_type == 'Cobertura total':
        coverage_type_encoded = 1
    elif coverage_type == 'Cobertura de colisión':
        coverage_type_encoded = 2
    elif coverage_type == 'Cobertura amplia':
        coverage_type_encoded = 3
    else:
        coverage_type_encoded = 4

    if insurance_state == 'Al día':
        insurance_state_encoded = 0
    else:
        insurance_state_encoded = 1

    # Convertir modelo de coche a código numérico
    car_model_encoded = label_encoder.transform([car_model])[0]

    #características
    input_data = [[coverage_type_encoded, car_model_encoded, car_year, insured_value, deductible, insurance_state_encoded, third_party_damage, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    #Predicción
    if submit_button:
        #Predicción utilizando el modelo entrenado
        prediction = model.predict(input_data)
        #Mostrar predicción
        if prediction[0] == 1:
            st.write('Es probable que haya gastos médicos.')
        else:
            st.write('Es poco probable que haya gastos médicos.')

if __name__ == '__main__':
    main()