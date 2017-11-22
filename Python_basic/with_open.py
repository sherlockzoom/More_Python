# coding=utf-8

hosts = open("/etc/hosts")
try:
    for line in hosts:
        if line.startswith("#"):
            continue
        print(line.strip())
finally:
    hosts.close()
# with 
print("with 上下文管理")
with open("/etc/hosts") as hosts:
    for line in hosts:
        if line.startswith("#"):
            continue
        print(line.strip())
