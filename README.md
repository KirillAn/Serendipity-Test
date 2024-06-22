# serendipity
<pre>
  <div id="header" align="center"> <img src=pics/lgo.jpg width="450"/></div>
</pre>


![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) ![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![HTML](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white) ![CSS](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white) ![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)<br/>
  </pre>
  
Open-source tool for EDA (Exploratory Data Analysis) of textual corpora. Built with BERT, transformers, spaCy and pyvis.

This tool would be useful for social science researchers, data journalists, business analysts and PhD students who regularly need to gain insights from text data, but have little or no experience with tuning NLP pipelines/programming complex systems.


## Demo Installation
- open Terminal app
- clone this repo to your local machine
```
git clone https://github.com/KirillAn/Serendipity-Test.git
```
- make sure you have Docker and docker-compose pre-installed
- run the following commands:
```
cd Serendipity-Test/observable-unirel

docker-compose up -d --build
```

- then access the interface at [http://localhost:5001](http://localhost:5001) 
  
<div id="header" align="center"> <img src=pics/main-page.gif width="800"/>
  </div> <br/>

- upload a [.csv file](data/test_data.csv) and wait till the graph is rendered

<div id="header" align="center"> <img src=pics/demo.gif width="800"/>
  </div> <br/>   

## Examples

Please proceed to unirel_code.ipynb if you wish to take a closer look at how things work under the hood and/or use Jupyter as a platform for using the tool.

**Graphs**
<pre>
  <div id="header" align="center"> <img src=pics/exmpl0.jpg width="1000"/>
  </div>
</pre>
<pre>
<div id="header" align="center"> <img src=pics/exmpl1.jpg width="1000"/>
  </div>
</pre>
<pre>
<div id="header" align="center"> <img src=pics/exmpl2.jpg width="1000"/>
  </div>
</pre>
  
## Algorithm
<pre>
<div id="header" align="center"> <img src=pics/model.png width="1000"/>
  </div>
</pre>

[UniRel](https://github.com/wtangdev/UniRel/tree/main) is used for Relational Triple Extraction (RTE) 
- everything is encoded in one representation - both entities and relationships
- with the help of BERT and a special relationship map extracting relationships
- Entity-Entity Interaction - whether there is interaction between entities or not. The function is symmetrical
- Entity-Relation Interaction - separation between an entity and a relationship

  
<pre>
<div id="header" align="center"> <img src=pics/e_e_relations.jpg width="1000"/>
  </div>
</pre>

## Languages and knowledge domains
At the moment, the instrument only supports the English language and one knowledge domain - news.

## TO-DO:
- train the model on other knowledge domains such as Legal, Medicine, AI & Technologies
- add domain selection to web application
- remake Graph visualization with another stack - WebGl and Js/Ts
- deploy web application
- make the application available for anyone
- collect feedback
