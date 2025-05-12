import numpy as np
import pandas as pd
from src.kohonen import KohonenNetwork

def distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)


def test_00():
    df = pd.read_csv("data/europe.csv")
    df = df.set_index("Country")
    df = (df - df.mean()) / df.std(ddof=0) 

    k_net = KohonenNetwork(df.shape[1], dims_out=(5, 5))
    k_net.train(df.values, epochs=100)

    predictions = k_net.predict(df.values)

    distance_to_first = [distance(p, predictions[0]) for p in predictions]
    indices = np.argsort(distance_to_first)

    import pdb; pdb.set_trace()
