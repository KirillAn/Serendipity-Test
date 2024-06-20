from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
from unirel_model import extract_relations_from_model_output  # Импорт функции из модели
from pyvis.network import Network
import logging
import os

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

# Укажите путь к вашей модели
MODEL_PATH = "model/nyt-checkpoint-final"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        logging.info("Received a POST request")
        if 'file' not in request.files:
            logging.error("No file provided in request.")
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            logging.error("No file selected.")
            return jsonify({'error': 'No selected file'}), 400

        if file:
            logging.info(f"Received file: {file.filename}")
            
            # Чтение содержимого файла как CSV
            try:
                df = pd.read_csv(file)
                logging.info("CSV file read successfully")
            except Exception as e:
                logging.error(f"Error reading CSV file: {e}")
                return jsonify({'error': 'Error reading CSV file'}), 400
            
            # Проверка, есть ли столбец 'content'
            if 'content' not in df.columns:
                logging.error("CSV file must contain a 'content' column.")
                return jsonify({'error': 'CSV file must contain a "content" column'}), 400
            
            # Вызов модели unirel для извлечения отношений
            try:
                relations = extract_relations_from_model_output(df, MODEL_PATH)
                logging.info(f"Extracted relations: {relations}")
            except Exception as e:
                logging.error(f"Error extracting relations: {e}")
                return jsonify({'error': 'Error extracting relations'}), 500
            
            # Визуализация графа
            try:
                net = Network(
                    directed=True,
                    width="100%",  # Увеличим размер окна
                    height="800px",  # Увеличим размер окна
                    bgcolor="#ffffff",  # Белый фон
                    font_color="white",
                    select_menu=True,
                    filter_menu=True
                )

                color_entity = "#000000"  # Черный цвет узлов
                for e in relations:
                    net.add_node(e["head"], shape="circle", color=color_entity, label=e["head"], font={"color": "white"})
                for e in relations:
                    net.add_node(e["tail"], shape="circle", color=color_entity, label=e["tail"], font={"color": "white"})

                for rel in relations:
                    net.add_edge(
                        rel["head"],
                        rel["tail"],
                        title=rel["type"],
                        label=rel["type"]
                    )
                    
                net.repulsion(
                    node_distance=200,
                    central_gravity=0.2,
                    spring_length=200,
                    spring_strength=0.05,
                    damping=0.09
                )
                net.set_edge_smooth('dynamic')
  # Добавим кнопки управления физикой
                
                graph_path = os.path.join('static', 'graph.html')
                net.save_graph(graph_path)
                logging.info(f"Graph saved to {graph_path}")

            except Exception as e:
                logging.error(f"Error creating graph: {e}")
                return jsonify({'error': 'Error creating graph'}), 500

            return jsonify({'success': True, 'redirect_url': url_for('visualization')})

    logging.info("Rendering the index page")
    return render_template("index.html")

@app.route("/visualization")
def visualization():
    return render_template("graph.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
