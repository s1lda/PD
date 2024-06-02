import streamlit as st
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from AntAlgorithm import AntAlgorithm
import seaborn as sns
import time

st.title('Граф городов и Муравьиный алгоритм')

# Ввод размера матрицы
m = st.number_input('Введите размерность массива (m x m):', min_value=2, step=1, value=5)

# Генерация новой матрицы
if st.button('Сгенерировать новую матрицу'):
    L = np.random.randint(1, 50, size=(m, m))
    np.fill_diagonal(L, 0)
    st.session_state['L'] = L
    st.session_state['new_matrix_generated'] = True

# Использование существующей матрицы или генерация новой, если еще не сгенерирована
if 'L' not in st.session_state or not st.session_state.get('new_matrix_generated', False):
    L = np.random.randint(1, 50, size=(m, m))
    np.fill_diagonal(L, 0)
    st.session_state['L'] = L
    st.session_state['new_matrix_generated'] = True
else:
    L = st.session_state['L']

st.write('Сгенерированная матрица L:')
st.write(L)

# Построение графа городов
st.subheader('Граф городов')
G = nx.DiGraph(L)
pos = nx.spring_layout(G)
labels = {i: i + 1 for i in range(m)}

try:
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, labels=labels, with_labels=True, node_color='lightblue', edge_color='gray', node_size=700, font_size=10)
    st.pyplot(plt)
except KeyError as e:
    st.error(f"Ошибка при построении графа: {e}")

# Ввод гиперпараметров
st.sidebar.subheader('Гиперпараметры модели')
a = st.sidebar.number_input('Коэффициент запаха', min_value=0.0, step=1.0, value=1.0)
b = st.sidebar.number_input('Коэффициент расстояния', min_value=0.0, step=1.0, value=5.0)
rho = st.sidebar.number_input('Коэффициент высыхания', min_value=0.0, step=0.1, value=0.7)
e = st.sidebar.number_input('Количество элитных муравьев', min_value=0.0, step=1.0, value=5.0)
Q = st.sidebar.number_input('Количество выпускаемого феромона', min_value=0.0, step=0.1, value=1.0)

# Запуск модели
aa = AntAlgorithm(a=a, b=b, rho=rho, e=e, Q=Q, random_seed=7)

if st.button('Запустить алгоритм'):
    st.subheader('Результаты алгоритма')
    start_time = time.time()
    tao = aa.fit(L)
    best_route = aa.get_best_route()
    best_dist = aa.get_best_dist()
    end_time = time.time()

    formatted_route = [(int(city) + 1) for city in best_route]
    pairs = [(formatted_route[i], formatted_route[i + 1]) for i in range(len(formatted_route) - 1)]
    pairs.append((formatted_route[-1], formatted_route[0]))

    st.success(f"**Лучший путь:** {formatted_route}")
    st.info(f"**Общая длина пути:** {best_dist}")
    st.write(f"**Путь в виде пар (a; b):** {pairs}")
    st.write(f"**Время выполнения:** {end_time - start_time:.2f} секунд")
