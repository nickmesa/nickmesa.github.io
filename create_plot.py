import matplotlib.pyplot as plt

# Sample data
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]

# Create a plot
plt.figure(figsize=(8, 6))
plt.plot(x, y, marker='o')
plt.title('Sample Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)

# Save the plot as a PNG file
plt.savefig('plot.png')

print('Plot created and saved as plot.png')
