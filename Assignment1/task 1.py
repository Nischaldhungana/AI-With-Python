def tester(givenstring="Too short, Write more then 10 Letters."):
    print(givenstring)

def main():
    while True:
        user_input = input("Write something (quit ends): ")
        if user_input.lower() == "quit":
            break
        if len(user_input) < 10:
            tester()
        else:
            tester(user_input)

main()
