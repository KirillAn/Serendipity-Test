import itertools
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import clean_doc
from pyvis.network import Network
import random
import tempfile
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


def viz_topic_bubbles(
        topic_model,
        projected_topics,
        texts
        ):

    x = projected_topics[:, :1]
    y = projected_topics[:, 1:]
    topic_freq = topic_model.get_topic_freq()
    doc_info = topic_model.get_document_info(texts)
    df = topic_freq.merge(doc_info, on='Topic', how='left')
    df = df.groupby(['Topic', 'Top_n_words', 'Count', 'Name']).agg({'Probability': 'mean'}).reset_index()
    df['x'] = x
    df['y'] = y

    fig = px.scatter(
        df,
        x='x',
        y='y',
        hover_data={
            "Topic": True,
            "Top_n_words": True,
            "Count": True,
            "x": False,
            "y": False
        },
        text='Topic',
        size='Count',
        color='Name',
        size_max=100,
        template='plotly_white',
    )

    fig.update_traces(marker=dict(line=dict(width=1, color='Gray')))

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    return fig


def viz_scatter_texts(
        topic_model,
        texts,
        projected_texts
        ):

    topic_freq = topic_model.get_topic_freq()
    doc_info = topic_model.get_document_info(texts)
    df = topic_freq.merge(doc_info, on='Topic', how='left')
    x = projected_texts[:, :1]
    y = projected_texts[:, 1:]
    df['x'] = x
    df['y'] = y
    texts_c = df.groupby(['Topic']).agg({'Document': 'nunique'}).reset_index()
    texts_c = texts_c.rename(columns={'Document': 'Document_qty'})
    df = df.merge(texts_c, on='Topic', how='left')
    df.Document = df.Document.apply(lambda x: x[:100] + '...')

    fig = px.scatter(
        df,
        x='x',
        y='y',
        hover_data={
            "Topic": False,
            "Name": True,
            "Document": False,
            "Document_qty": False,
            "x": False,
            "y": False
        },
        hover_name='Document',
        color='Name',
        size_max=60,
        template='plotly_white',
    )

    fig.update_traces(marker=dict(line=dict(width=1, color='Gray')))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    return fig


def viz_word_scores(
        topic_model,
        top_n_topics=8,
        n_words=5,
        custom_labels=False,
        title="<b>Вероятности слов по темам</b>",
        width=250,
        height=250
):

    colors = itertools.cycle(["#D55E00", "#0072B2", "#CC79A7", "#E69F00", "#56B4E9", "#009E73", "#F0E442"])

    freq_df = topic_model.get_topic_freq()
    freq_df = freq_df.loc[freq_df.Topic != -1, :]
    if top_n_topics is not None:
        topics = sorted(freq_df.Topic.to_list()[:top_n_topics])
    else:
        topics = sorted(freq_df.Topic.to_list()[0:6])

    if isinstance(custom_labels, str):
        subplot_titles = [[[str(topic), None]] + topic_model.topic_aspects_[custom_labels][topic] for topic in topics]
        subplot_titles = ["_".join([label[0] for label in labels[:4]]) for labels in subplot_titles]
        subplot_titles = [label if len(label) < 30 else label[:27] + "..." for label in subplot_titles]
    elif topic_model.custom_labels_ is not None and custom_labels:
        subplot_titles = [topic_model.custom_labels_[topic + topic_model._outliers] for topic in topics]
    else:
        subplot_titles = [f"Тема {topic}" for topic in topics]
    columns = 4
    rows = int(np.ceil(len(topics) / columns))
    fig = make_subplots(
        rows=rows,
        cols=columns,
        shared_xaxes=False,
        horizontal_spacing=.1,
        vertical_spacing=.4 / rows if rows > 1 else 0,
        subplot_titles=subplot_titles
    )

    row = 1
    column = 1
    for topic in topics:
        words = [word + "  " for word, _ in topic_model.get_topic(topic)][:n_words][::-1]
        scores = [score for _, score in topic_model.get_topic(topic)][:n_words][::-1]

        fig.add_trace(
            go.Bar(x=scores,
                   y=words,
                   orientation='h',
                   marker_color=next(colors)),
            row=row, col=column)

        if column == columns:
            column = 1
            row += 1
        else:
            column += 1

    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        title={
            'text': f"{title}",
            'x': .5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black")
        },
        width=width * 4,
        height=height * rows if rows > 1 else height * 1.3,
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        ),
    )

    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    return fig


def viz_topic_heatmap(
        topic_model,
        topics=None,
        top_n_topics=None,
        n_clusters=None,
        custom_labels=False,
        title="<b>Матрица семантической близости тем</b>",
        width=800,
        height=800
):

    if topic_model.topic_embeddings_ is not None:
        embeddings = np.array(topic_model.topic_embeddings_)[topic_model._outliers:]
    else:
        embeddings = topic_model.c_tf_idf_[topic_model._outliers:]

    freq_df = topic_model.get_topic_freq()
    freq_df = freq_df.loc[freq_df.Topic != -1, :]

    if top_n_topics is not None:
        topics = sorted(freq_df.Topic.to_list()[:top_n_topics])
    else:
        topics = sorted(freq_df.Topic.to_list())

    sorted_topics = topics

    if n_clusters:
        distance_matrix = cosine_similarity(embeddings[topics])
        Z = linkage(distance_matrix, 'ward')
        clusters = fcluster(Z, t=n_clusters, criterion='maxclust')

        mapping = {cluster: [] for cluster in clusters}
        for topic, cluster in zip(topics, clusters):
            mapping[cluster].append(topic)
        mapping = [cluster for cluster in mapping.values()]
        sorted_topics = [topic for cluster in mapping for topic in cluster]

    indices = np.array([topics.index(topic) for topic in sorted_topics])
    embeddings = embeddings[indices]
    distance_matrix = cosine_similarity(embeddings)

    if isinstance(custom_labels, str):
        new_labels = [[[str(topic), None]] + topic_model.topic_aspects_[custom_labels][topic] for topic in
                      sorted_topics]
        new_labels = ["_".join([label[0] for label in labels[:4]]) for labels in new_labels]
        new_labels = [label if len(label) < 30 else label[:27] + "..." for label in new_labels]
    elif topic_model.custom_labels_ is not None and custom_labels:
        new_labels = [topic_model.custom_labels_[topic + topic_model._outliers] for topic in sorted_topics]
    else:
        new_labels = [[[str(topic), None]] + topic_model.get_topic(topic) for topic in sorted_topics]
        new_labels = ["_".join([label[0] for label in labels[:4]]) for labels in new_labels]
        new_labels = [label if len(label) < 30 else label[:27] + "..." for label in new_labels]

    fig = px.imshow(
        distance_matrix,
        labels=dict(color="Оценка близости"),
        x=new_labels,
        y=new_labels,
        color_continuous_scale='GnBu'
    )

    fig.update_layout(
        title={
            'text': f"{title}",
            'y': .95,
            'x': 0.55,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black"
            )
        },
        width=width,
        height=height,
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        ),
    )

    fig.update_layout(showlegend=True)
    fig.update_layout(legend_title_text='Trend')

    return fig


def viz_classes_corpus(classes):
    df = pd.DataFrame({'classes': classes})
    df = df.value_counts().rename_axis('classes').reset_index(name='counts')
    fig = px.bar(df, x='classes', y='counts', color='classes')

    return fig


def viz_classes_per_topic(classes, topics,topic=1):

    df = pd.DataFrame({'classes': classes, 'topics': topics})
    df = df[df['topics'] == topic].drop(['topics'], axis=1)
    df = df.value_counts().rename_axis('classes').reset_index(name='counts')
    fig = px.bar(df, x='classes', y='counts', color='classes')

    return fig


def viz_ner_per_topic(ents, ner_topics, topic=1):
    df = pd.DataFrame({'ents': ents, 'topics': ner_topics})
    df = df[df['topics'] == topic]
    df.drop(['topics'], inplace=True, axis=1)
    df['ents'] = df['ents'].apply(lambda x: x.strip())
    df = df.value_counts().rename_axis('entity').reset_index(name='counts').head(10)
    fig = px.bar(df, x='entity', y='counts')

    return fig


def viz_n_grams_per_topic(texts, topic_model, topic=1, n=3):
    ngram_freq_df = pd.DataFrame()
    vectorizer = CountVectorizer(ngram_range=(n,n))
    df = topic_model.get_document_info(texts)
    df = df[df['Topic'] == topic]
    df['Document'] = df['Document'].apply(clean_doc)

    ngrams = vectorizer.fit_transform(df['Document'])
    count_values = ngrams.toarray().sum(axis=0)
    ngram_freq = pd.DataFrame(
        sorted([(count_values[i], k) for k, i in vectorizer.vocabulary_.items()],
        reverse=True),
        columns=["частота", "n-gram"]
        )

    ngram_freq_df = pd.concat([ngram_freq_df, ngram_freq])
    top_ngram = ngram_freq_df.sort_values(by='частота', ascending=False).head(10)

    fig = px.bar(
        top_ngram,
        x='частота',
        y='n-gram',
        orientation='h',
        title=f'Top-10 {n}-грамм для темы "{df.Name.iloc[0]}"'
        )

    return fig

# Функция для визуализации графа знаний
# def viz_knowledge_graph(ents, kb_relations):
#     net = Network(
#         directed=True,
#         height='700px',
#         width='100%',
#         bgcolor='#222222',
#         font_color='white'
#     )
#
#     # Добавление узлов
#     for index, entity in ents.iterrows():
#         net.add_node(entity['Entity'], title=entity['Entity'])
#
#     # Добавление рёбер
#     for index, relation in kb_relations.iterrows():
#         net.add_edge(relation['Head'], relation['Tail'], title=relation['Type'])
#
#     # Настройки физики сети для анимации
#     net.repulsion(node_distance=420, central_gravity=0.33,
#                   spring_length=110, spring_strength=0.10,
#                   damping=0.95)
#
#     # Генерация HTML и возврат как объекта fig
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmpfile:
#         net.show(tmpfile.name)
#         # Чтение HTML-кода из временного файла и сохранение в переменную
#         tmpfile.seek(0)
#         html_content = tmpfile.read().decode('utf-8')
#
#     # Использование Plotly объекта Figure для встраивания HTML
#     fig = go.Figure(data=[go.Scatter(x=[], y=[])], layout=go.Layout())
#     fig.update_layout(
#         template=None,
#         title="Knowledge Graph Visualization",
#         margin=dict(l=0, r=0, t=30, b=0)
#     )
#     fig.add_trace(
#         go.Scatter(
#             x=[0], y=[0],
#             mode='markers+text',
#             text=['<iframe srcdoc="'+html_content+'" style="width:100%; height:700px;"></iframe>'],
#             textposition="bottom center"
#         )
#     )
#
#     return fig
# def viz_knowledge_graph(kb):
#     net = Network(
#         directed=True,
#         height="700px",
#         width="100%",
#         bgcolor="#eeeeee"
#     )
#
#     color_entity = "#00FF00"
#
#     # Сбор всех уникальных сущностей из отношений
#     unique_entities = set()
#     for relation in kb.relations:
#         unique_entities.add(relation['head'])
#         unique_entities.add(relation['tail'])
#
#     # Добавление узлов
#     for entity in unique_entities:
#         if entity in kb.entities:
#             entity_info = kb.entities[entity]
#             net.add_node(entity, title=entity, shape="circle", color=color_entity)
#
#     # Добавление рёбер
#     for relation in kb.relations:
#         if relation['head'] in unique_entities and relation['tail'] in unique_entities:
#             net.add_edge(relation['head'], relation['tail'], title=relation['type'], label=relation['type'])
#
#     net.repulsion(node_distance=200, central_gravity=0.2, spring_length=200, spring_strength=0.05, damping=0.09)
#     net.set_edge_smooth('dynamic')
#
#     # Генерация HTML
#     html = net.generate_html()
#     html = html.replace("'", "\"")
#     iframe_html = f"""<iframe style="width: 100%; height: 700px;margin:0 auto" name="result" allow="midi; geolocation; microphone; camera; display-capture; encrypted-media;" sandbox="allow-modals allow-forms allow-scripts allow-same-origin allow-popups allow-top-navigation-by-user-activation allow-downloads" allowfullscreen="" allowpaymentrequest="" frameborder="0" srcdoc='{html}'></iframe>"""
#
#     return iframe_html
#
#
import random

def determine_color(topic_labels, topic_colors):
    # Убедимся, что topic_labels это список
    if not isinstance(topic_labels, list):
        topic_labels = [topic_labels]

    # Преобразуем числа в строки
    topic_labels_str = [str(topic) for topic in topic_labels]

    # Сортировка тем и создание уникального идентификатора
    topic_key = "-".join(sorted(topic_labels_str))

    # Если тема уже имеет назначенный цвет, возвращаем этот цвет
    if topic_key in topic_colors:
        return topic_colors[topic_key]
    else:
        # Генерируем случайный цвет
        random_color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
        # Добавляем цвет в словарь для этой комбинации тем
        topic_colors[topic_key] = random_color
        return random_color

def viz_knowledge_graph(kb):
    net = Network(
        directed=True,
        height="700px",
        width="100%",
        bgcolor="#eeeeee"
    )

    topic_colors = {}  # Словарь для хранения цветов тем

    # Сбор всех уникальных сущностей из отношений
    unique_entities = set()
    for relation in kb.relations:
        unique_entities.add(relation['head'])
        unique_entities.add(relation['tail'])

    # Добавление узлов
    for entity in unique_entities:
        if entity in kb.entities:
            entity_info = kb.entities[entity]
            color_entity = determine_color(entity_info.get('topic_label', []), topic_colors)
            net.add_node(entity, title=entity, shape="circle", color=color_entity)

    # Добавление рёбер
    for relation in kb.relations:
        if relation['head'] in unique_entities and relation['tail'] in unique_entities:
            net.add_edge(relation['head'], relation['tail'], title=relation['type'], label=relation['type'])

    net.repulsion(node_distance=200, central_gravity=0.2, spring_length=200, spring_strength=0.05, damping=0.09)
    net.set_edge_smooth('dynamic')

    # Генерация HTML
    html = net.generate_html()
    html = html.replace("'", "\"")
    iframe_html = f"""<iframe style="width: 100%; height: 700px;margin:0 auto" name="result" allow="midi; geolocation; microphone; camera; display-capture; encrypted-media;" sandbox="allow-modals allow-forms allow-scripts allow-same-origin allow-popups allow-top-navigation-by-user-activation allow-downloads" allowfullscreen="" allowpaymentrequest="" frameborder="0" srcdoc='{html}'></iframe>"""

    return iframe_html





#%%

#%%
