from collections import OrderedDict

import pandas as pd
import pytest
from faker import Faker


@pytest.fixture
def fake_data() -> pd.DataFrame:
    fake = Faker()
    Faker.seed(42)
    dataset_size = 10
    data = {
        "Id": [i for i in range(dataset_size)],
        "age": [fake.pyint(min_value=29, max_value=77)
                for _ in range(dataset_size)],
        "sex": fake.random_elements(elements=OrderedDict([(0, 0.5),
                                                          (1, 0.5)]),
                                    length=dataset_size),
        "cp": [fake.pyint(min_value=0, max_value=3)
               for _ in range(dataset_size)],
        "trestbps": [fake.pyint(min_value=94, max_value=200)
                     for _ in range(dataset_size)],
        "chol": [fake.pyint(min_value=126, max_value=564)
                 for _ in range(dataset_size)],
        "fbs": fake.random_elements(elements=OrderedDict([(0, 0.5),
                                                          (1, 0.5)]),
                                    length=dataset_size),
        "restecg": [fake.pyint(min_value=0, max_value=2)
                    for _ in range(dataset_size)],
        "thalach": [fake.pyint(min_value=71, max_value=202)
                    for _ in range(dataset_size)],
        "exang": fake.random_elements(elements=OrderedDict([(0, 0.5),
                                                            (1, 0.5)]),
                                      length=dataset_size),
        "oldpeak": [fake.pyfloat(min_value=0, max_value=6)
                    for _ in range(dataset_size)],
        "slope": [fake.pyint(min_value=0, max_value=2)
                  for _ in range(dataset_size)],
        "ca": [fake.pyint(min_value=0, max_value=4)
               for _ in range(dataset_size)],
        "thal": [fake.pyint(min_value=0, max_value=3)
                 for _ in range(dataset_size)],
    }
    df = pd.DataFrame(data=data)
    return df
