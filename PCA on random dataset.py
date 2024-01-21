
import numpy as np
import matplotlib.pyplot as plt


#STEP:1
# Set the random seed for reproducibility (optional)
np.random.seed(42)

# Generate a 20x2 matrix from a normal distribution with mean 0 and standard deviation 1
matrix = np.random.normal(0, 1, (20, 2))

# Print the matrix
print(matrix)



#STEP:2
# Extract the x and y coordinates from the matrix
x = matrix[:, 0]
y = matrix[:, 1]
print(x)
# Create a scatter plot
plt.scatter(x, y)

# Add labels and title to the plot
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot')
legend_handles = [
    plt.scatter([], [], label='THA076BEI009\nTHA076BEI036', alpha=0),
   
]

plt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(0.73, 0.999), 
           ncol=len(legend_handles),handlelength=0.3, borderpad=0.08)

# Show the plot
plt.show()

#STEP:3
# Generate a 2x2 matrix from a uniform distribution between 0 and 1
matrix2 = np.random.uniform(0, 1, (2, 2))

# Print the matrix
print(matrix2)

#STEP:4
# Multiply the matrices
product = np.dot(matrix, matrix2)

# Print the result
print(product)

#STEP:5

# Extract the x and y coordinates from the matrix
x = product[:, 0]
y = product[:, 1]

# Create a scatter plot
plt.scatter(x, y)

# Add labels and title to the plot
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot')
legend_handles = [
    plt.scatter([], [], label='THA076BEI009\nTHA076BEI036', alpha=0),
   
]

plt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(0.53, 0.999), 
           ncol=len(legend_handles),handlelength=0.3, borderpad=0.08)

# Show the plot
plt.show()

column1 = product[:, 0]

# Print the first column
print(column1)
column2 = product[:, 1]

# Print the first column
print(column2)

#STEP:6
variance1 = np.var(column1)

# Print the variance
print("Variance of the single column:")
print(variance1)
variance2 = np.var(column2)

# Print the variance
print("Variance of the single column:")
print(variance2)

#STEP:7
# Calculate the covariance matrix
covariance_matrix = np.cov(product.T)

# Print the covariance matrix
print("Covariance matrix:")
print(covariance_matrix)

#STEP:8
# Calculate the eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

# Print the eigenvalues
print("Eigenvalues:")
print(eigenvalues)

# Print the eigenvectors
print("\nEigenvectors:")
print(eigenvectors)

#STEP:9
# Calculate the sum of eigenvalues
total_variance = np.sum(eigenvalues)

# Calculate the proportion of variance explained by each eigenvalue
variance_proportion = eigenvalues / total_variance

# Print the variance proportion
print("Proportion of Variance:")
print(variance_proportion)


# Perform matrix multiplication
Y = np.dot( eigenvectors.T,product.T)

# Print the result
print("Matrix multiplication result:")
print(Y)


# Calculate the covariance matrix
covariance_matrix2 = np.cov(Y)

# Print the covariance matrix
print("Covariance matrix:")
print(covariance_matrix2)


plt.scatter(Y[0,:],Y[1,:])

plt.axis('equal')
plt.grid(True)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

legend_handles = [
    plt.scatter([], [], label='THA076BEI009\nTHA076BEI036', alpha=0),
   
]

plt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(0.73, 0.999), 
           ncol=len(legend_handles),handlelength=0.3, borderpad=0.08)

print(Y)


neweig = eigenvectors[:, 0]
# Convert the first column to a 1x2 matrix
eig = np.reshape(neweig, (1, 2))

Y = np.dot(neweig.T,product.T)
print(Y)
# Calculate the covariance matrix
covariance_matrix2 = np.cov(Y)

# Print the covariance matrix
print("Covariance matrix:")
print(covariance_matrix2)
plt.scatter(Y,np.zeros_like(Y))
plt.xlabel('Principal Component 1')
legend_handles = [
    plt.scatter([], [], label='THA076BEI009\nTHA076BEI036', alpha=0),
   
]

plt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(0.73, 0.999), 
           ncol=len(legend_handles),handlelength=0.3, borderpad=0.08)

plt.grid(True)
print(Y)


neweig = eigenvectors[:, 1]
# Convert the first column to a 1x2 matrix
eig = np.reshape(neweig, (1, 2))

Y = np.dot(neweig.T,product.T)
print(Y)
# Calculate the covariance matrix
covariance_matrix2 = np.cov(Y)

# Print the covariance matrix
print("Covariance matrix:")
print(covariance_matrix2)
plt.scatter(Y,np.zeros_like(Y))
plt.xlabel('Principal Component 1')
legend_handles = [
    plt.scatter([], [], label='THA076BEI009\nTHA076BEI036', alpha=0),
   
]

plt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(0.73, 0.999),
           ncol=len(legend_handles),handlelength=0.3, borderpad=0.08)

plt.grid(True)
print(Y)
