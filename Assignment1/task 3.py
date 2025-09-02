prices = [10, 14, 22, 33, 44, 13, 22, 55, 66, 77]
totalsum = 0

print("Supermarket")
print("===========")

while True:
    choice = int(input("Please select product (1-10) 0 to Quit: "))
    if choice == 0:
        break
    if 1 <= choice <= 10:
        print(f"Product: {choice} Price: {prices[choice-1]}")
        totalsum += prices[choice-1]

print(f"Total: {totalsum}")
payment = int(input("Payment: "))
print(f"Change: {payment - totalsum}")
print('kittos')




















