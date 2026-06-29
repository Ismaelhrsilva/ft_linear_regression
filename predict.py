mileage = input("What is the mileage of your car\n")

if mileage.isdigit():
    int(mileage)
    print("Mileage: ", mileage)
else:    
    print("Please enter only a number.")
