import re
def normalized_text(string):
    string = string.lower()
    # ^ matches the start of the string
    # \s matches Unicode whitespace characters (which includes [ \t\n\r\f\v], and also many other characters
    string = re.sub("[^A-Za-z0-9\s]","", string)
    return string

result = normalized_text("Hello, little lion!")
print(result)