def load_thetas(path: str) -> tuple[float, float]:
    try:
        with open(path) as f:
            theta0 = float(f.readline())
            theta1 = float(f.readline())
        return theta0, theta1
    except Exception as e:
        return (0.0, 0.0)


def estimate_price(mileage: float, theta0: float, theta1: float) -> float:
    return theta0 + theta1 * mileage


def main(mileage: float):
    theta0, theta1 = load_thetas("thetas.txt")
    price = estimate_price(mileage, theta0, theta1)
    print(price)


if __name__ == "__main__":
    try:
        mileage = input("What is the mileage of your car\n")
        main(float(mileage))
    except:
        print("Error")
