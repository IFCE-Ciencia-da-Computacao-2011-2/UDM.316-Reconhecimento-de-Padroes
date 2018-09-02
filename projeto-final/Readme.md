# Trabalho final - Sistema de sugestão de efeitos

## Passos

1. Remover a base antiga de PEDALBOARDS (e só) ```rm data/pedalboard-info.csv```
1. Obter base de dados: ```scrapy runspider scrap.py -o data/pedalboard-info.csv -t csv```
1. Tratar os dados: ```Tratamento de dados 1.ipynb```
1. Tratar os dados: ```Tratamento de dados 2.ipynb```
1. Tratar os dados: ```Tratamento de dados 3 - Categorias.ipynb```