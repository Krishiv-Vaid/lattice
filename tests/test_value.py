from lattice.value import Value

# Test addition backward
a = Value(2.0)
b = Value(3.0)

c = a + b

# Pretend a gradient of 1 has arrived at c
c.grad = 1.0
c._backward()

print("Addition:")
print("c =", c.data)
print("a.grad =", a.grad)
print("b.grad =", b.grad)

print()

# Test multiplication backward
x = Value(2.0)
y = Value(3.0)

z = x * y

# Pretend a gradient of 1 has arrived at z
z.grad = 1.0
z._backward()

print("Multiplication:")
print("z =", z.data)
print("x.grad =", x.grad)
print("y.grad =", y.grad)