def my_split(sentence, separator):
    result = []
    current = ""
    for ch in sentence:
        if ch == separator:
            if current != "":
                result.append(current)
                current = ""
        else:
            current += ch
    if current != "":
        result.append(current)
    return result

def my_join(items, separator):
    result = ""
    for i in range(len(items)):
        result += items[i]
        if i < len(items) - 1:
            result += separator
    return result


sentence = input("Please enter sentence:")
separated = my_split(sentence, " ")

print(my_join(separated, ","))
for word in separated:
    print(word)
