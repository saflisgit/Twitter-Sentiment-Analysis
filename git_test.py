import re

text = "ekrem @ solmaz"

print(text)

text = re.sub(r'@[A-Za-z0-9]+', '', text)
text = re.sub(r'@+', '', text)

print(text)