from lattice.value import Value

a = Value(2)
b = Value(3)

c = a * b
d = c + a

print("a =", a.data)
print("b =", b.data)
print("c =", c.data)
print("d =", d.data)

print("c operation:", c._op)
print("d operation:", d._op)

print("c has", len(c._prev), "parents")
print("d has", len(d._prev), "parents")