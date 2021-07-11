from fastapi.testclient import TestClient
from main import app
from datetime import datetime


now = datetime.now()
current_time = now.strftime("%H:%M")

# test to check the correct functioning of the /ping route
def test_ping():
    with TestClient(app) as client:
        response = client.get("/ping")
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"ping": "pong","Time":current_time}


# test to check if Iris Virginica is classified correctly
def test_pred_virginica():
    # defining a sample payload for the testcase
    payload = {
        "sepal_length": 5.9,
        "sepal_width": 3,
        "petal_length": 5.1,
        "petal_width": 1.8,
    }
    with TestClient(app) as client:
        response = client.post("/predict_flower", json=payload)
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"flower_class": "Iris Virginica","Time":current_time}


# test to check if Iris setosais classified correctly
def test_pred_setosa():
    # defining a sample payload for the testcase
    payload = {
        "sepal_length": 4.8,
        "sepal_width": 3,
        "petal_length": 1.4,
        "petal_width": 0.1,
    }
    with TestClient(app) as client:
        response = client.post("/predict_flower", json=payload)
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"flower_class": "Iris Setosa","Time":current_time}

# test to check if Iris setosais classified correctly
def test_pred_versicolor():
    # defining a sample payload for the testcase
    payload = {
        "sepal_length": 5.7,
        "sepal_width": 2.8,
        "petal_length": 4.1,
        "petal_width": 1.3,
    }
    with TestClient(app) as client:
        response = client.post("/predict_flower", json=payload)
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"flower_class": "Iris Versicolour","Time":current_time}