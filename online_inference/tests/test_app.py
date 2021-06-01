from fastapi.testclient import TestClient
from src.app import app


def test_predict(fake_data):
    with TestClient(app) as client:
        features = fake_data.columns.tolist()
        data = fake_data.values.tolist()
        response = client.get("/predict/", json={"data": data, "features": features})
        assert response.status_code == 200
        assert all([x["heart_disease"] in [0, 1] for x in response.json()])
