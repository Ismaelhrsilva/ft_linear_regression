from data import load_data

mileage = input("What is the mileage of your car\n")

if mileage.isdigit():
    int(mileage)
    print("Mileage: ", mileage)
    print(load_data("data.csv"))
else:    
    print("Please enter only a number.")
