import pickle

import gradio as gr
import nltk
import numpy as np
import pandas as pd
from bertopic import BERTopic
from nltk.corpus import stopwords
from kg import *
from topic_modelling import dim_reduction_texts_topics
from viz import (
    viz_topic_bubbles,
    viz_scatter_texts,
    viz_word_scores,
    viz_topic_heatmap,
    viz_classes_corpus,
    viz_classes_per_topic,
    viz_ner_per_topic,
    viz_n_grams_per_topic,
    viz_knowledge_graph

)


def make_plot(plot_type):
    fig = None  # Инициализация переменной fig

    if plot_type == "graphs":
        # Возвращает None для gr.Plot и HTML-строку для gr.HTML
        html_content = viz_knowledge_graph(kb)
        return None, html_content
    else:
        # Возвращает Plotly Figure для gr.Plot и None для gr.HTML
        if plot_type == "topic bubbles":
            fig = viz_topic_bubbles(topic_model, projected_topics, texts)
        elif plot_type == "scatter texts":
            fig = viz_scatter_texts(topic_model, texts, projected_texts)
        elif plot_type == "word scores":
            fig = viz_word_scores(topic_model)
        elif plot_type == "topic heatmap":
            fig = viz_topic_heatmap(topic_model)
        elif plot_type == "ner per topic":
            fig = viz_ner_per_topic(all_ents, ner_topics)
        elif plot_type == "n grams per topic":
            fig = viz_n_grams_per_topic(texts, topic_model)
        elif plot_type == "classes_per_corpus":
            fig = viz_classes_corpus(classes)
        elif plot_type == "classes per topic":
            fig = viz_classes_per_topic(classes, topics)

        return fig, None



with gr.Blocks() as demo:
    bbc = pd.read_csv('/Users/kirillanosov/serendipity_hck/data/bbc-news-data.csv', sep='\t')
    texts = bbc['content']
    y_true = bbc['category']
    # topics, topic_model, embeddings = model_topics(texts=texts)
    # ner_topics, all_ents = model_ner_tokens##(topics, texts)
    candidate_labels = list(set(y_true))
    # classes = model_zero_shot_classification(candidate_labels, texts)

    # ЗАГЛУШКА ДЛЯ ДЕМОНСТРАЦИИ
    with open('/Users/kirillanosov/serendipity_hck/cache/zs-classes.pkl', 'rb') as file:
        classes = pickle.load(file)
    with open('/Users/kirillanosov/serendipity_hck/cache/ner_topics.pkl', 'rb') as file:
        ner_topics = pickle.load(file)
    with open('/Users/kirillanosov/serendipity_hck/cache/topics.pkl', 'rb') as file:
        topics = pickle.load(file)
    with open('/Users/kirillanosov/serendipity_hck/cache/embeddings.npy', 'rb') as file:
        embeddings = np.load(file)
    with open('/Users/kirillanosov/serendipity_hck/cache/all_ents.pkl', 'rb') as file:
        all_ents = pickle.load(file)
    with open('/Users/kirillanosov/Desktop/Serendipity/Serendipity-Test/cache/kb1_.pkl', 'rb') as file:
        kb = pickle.load(file)
    # if not isinstance(kb_relations, pd.DataFrame):
    #     kb_relations = pd.DataFrame(kb_relations)


    topic_model = BERTopic.load('/Users/kirillanosov/serendipity_hck/model/topic_model')
    projected_texts, projected_topics = dim_reduction_texts_topics(embeddings, topic_model.topic_embeddings_)

    button = gr.Radio(label="Plot type",
                        choices=['topic bubbles', 'scatter texts', 'word scores',
                                'topic heatmap', 'ner per topic', 'n grams per topic',
                                'classes_per_corpus', 'classes per topic','graphs'], value='scatter_plot')

    #plot = gr.Plot(label="Plot")
    plot_output = gr.Plot(label="Plot")
    html_output = gr.HTML(label="Graph")
    button.change(make_plot, inputs=button, outputs=[plot_output, html_output])

    demo.load(make_plot, inputs=[button], outputs=[plot_output, html_output])


if __name__ == "__main__":
    nltk.download('stopwords')
    stopwords = set(stopwords.words('english'))
    demo.launch(server_name="0.0.0.0", server_port=9999)

#%%

#%%
