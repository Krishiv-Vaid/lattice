from lattice.value import Value

a = Value(2)
b = Value(3)

c = a + b
d = a * b

print("a =", a.data)
print("b =", b.data)
print("a + b =", c.data)
print("a * b =", d.data)