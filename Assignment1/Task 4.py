def my_split(sentence, separator):
    result = []
    current = ""
    for char in sentence:
        if char == separator:
            if current:
                result.append(current)
                current = ""
        else:
            current += char
    if current:
        result.append(current)
    return result

def my_join(items, separator):
    result = ""
    for i in range(len(items)):
        result += items[i]
        if i < len(items) - 1:
            result += separator
    return result

# this is the main program
sentence = input("Please enter sentence:")
separated = my_split(sentence, " ")
print(my_join(separated, ","))
print(my_join(separated, " "))


