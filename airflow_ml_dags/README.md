```
USER=Адреc отправителя PWD=пароль RECIPIENT=адрес получателя alert 
VOLUME_DIR=путь к монтируемой папки MODEL_PATH=путь к модели docker-compose up --build
```

Самооценка:

№| Задание| Баллы
--- | --- | ---
1 | Реализуйте dag, который генерирует данные | 5 
2 | Реализуйте dag, который обучает модель | 10
3 |  Реализуйте dag, который использует модель ежедневно | 5
3a | Реализуйте сенсоры на то, что данные готовы для дагов тренировки и обучения | 3
4 |  все даги реализованы только с помощью DockerOperator | 10
5 | Протестируйте ваши даги |
6 | В docker compose так же настройте поднятие mlflow |
7 | вместо пути в airflow variables  используйте апи Mlflow Model Registry |
8 | Настройте alert в случае падения дага | 3
9 | Самооценка | 1
 - | Итого | 37