import sys

with open('src/optimizer.py', 'r', encoding='utf-8') as f:
    code = f.read()

code = code.replace("config.get('amplitude', {}).get", "(config.get('amplitude') or {}).get")
code = code.replace("config.get('center', {}).get", "(config.get('center') or {}).get")
code = code.replace("config.get('sigma', {}).get", "(config.get('sigma') or {}).get")
code = code.replace("config.get('wing_window', {}).get", "(config.get('wing_window') or {}).get")

with open('src/optimizer.py', 'w', encoding='utf-8') as f:
    f.write(code)

print('Done fixing optimizer.py')
