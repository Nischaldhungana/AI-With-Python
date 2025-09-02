def main():
    shopping_list = []

    while True:
        print("Would you like to")
        print("(1)Add or")
        print("(2)Remove items or")
        print("(3)Quit?: ", end="")
        choice = input().strip()

        if choice == "1":
            item = input("What will be added?: ").strip()
            shopping_list.append(item)

        elif choice == "2":
            if len(shopping_list) == 0:
                print("There are 0 items in the list.")
                print("Incorrect selection.")
                continue

            print(f"There are {len(shopping_list)} items in the list.")
            to_remove = input("Which item is deleted?: ").strip()

            if not to_remove.isdigit():
                print("Incorrect selection.")
                continue

            index = int(to_remove)

            if 0 <= index < len(shopping_list):
                shopping_list.pop(index)
            else:
                print("Incorrect selection.")

        elif choice == "3":
            print("The following items remain in the list:")
            for item in shopping_list:
                print(item)
            break

        else:
            print("Incorrect selection.")


if __name__ == "__main__":
    main()
