import csv


def load_data(path: str) -> tuple[list, list]:
    with open(path) as f:
        data = csv.reader(f)
        next(data)
        km = []
        price = []
        for ikm, iprice in data:
            km.append(float(ikm))
            price.append(float(iprice))
    return km, price


