from lattice.value import Value

x = Value(3.0)
y = Value(4.0)

z = x * y
loss = z * z

loss.backward()

print("loss =", loss.data)
print("dx =", x.grad)
print("dy =", y.grad)