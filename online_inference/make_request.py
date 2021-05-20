import click
import pandas as pd
import requests


@click.command()
@click.option("-h", "--host", default="0.0.0.0")
@click.option("-p", "--port", default=8080)
@click.option("-d", "--data_path")
def predict(host, port, data_path):
    df = pd.read_csv(data_path)
    df.drop(columns=['target'], inplace=True)
    features = ["Id"] + list(df.columns)
    for i, row in enumerate(df.values[-10:]):
        data = [i] + [x.tolist() for x in row]
        response = requests.get(f"http://{host}:{port}/predict",
                                json={"data": [data], "features": features})
        print(response.json())


if __name__ == "__main__":
    predict()
