
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

data = load_iris()


X = data.data  
y = data.target  

df = pd.DataFrame(X)

print("Data are")
print(df)
print(y)
scaler = StandardScaler()
scaler.fit(X)
zero_mean_data = scaler.transform(X)

covariance_matrix = np.cov(zero_mean_data.T)
print("covariance_matrix:")
print(covariance_matrix)
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

# Print the eigenvalues
print("Eigenvalues:")
print(eigenvalues)

# Print the eigenvectors
print("\nEigenvectors:")
print(eigenvectors)

# Calculate the sum of eigenvalues
total_variance = np.sum(eigenvalues)

# Calculate the proportion of variance explained by each eigenvalue
variance_proportion = eigenvalues / total_variance

# Print the variance proportion
print("Proportion of Variance:")
print(variance_proportion)

sorted_indices = np.argsort(eigenvalues)[::-1]

#this function dot product eigen vector and data and print covariance matrix
def PCAfun(subset): 
    
    print("************************************************************************",subset)
    print("Num of principle component",len(subset))
    neweig = eigenvectors[:,subset]

    YY = np.dot(neweig.T,zero_mean_data.T)

    covariance_matrix2 = np.cov(YY)

    # Print the covariance matrix
    print("Covariance matrix:")
    print(covariance_matrix2)
    return YY

#for 1 component
Y = PCAfun([sorted_indices[0]])

plt.scatter(Y.T, [0] * len(Y.T), c=y)
plt.axis('equal')
plt.grid(True)
plt.xlabel('Principal Component 1')

plt.title('PCA Scatter Plot')
plt.colorbar()
legend_handles = [
    plt.scatter([], [], label='THA076BEI009\nTHA076BEI036', alpha=0),
   
]

plt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(0.66, 0.999),
           ncol=len(legend_handles),handlelength=0.3, borderpad=0.08)
plt.show()

#for 1 component
Y = PCAfun([sorted_indices[3]])

plt.scatter(Y.T, [0] * len(Y.T), c=y)
plt.axis('equal')
plt.grid(True)
plt.xlabel('Principal Component 1')

plt.title('PCA Scatter Plot')
plt.colorbar()
legend_handles = [
    plt.scatter([], [], label='THA076BEI009\nTHA076BEI036', alpha=0),
   
]

plt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(0.66, 0.999),
           ncol=len(legend_handles),handlelength=0.3, borderpad=0.08)
plt.show()

#for 2 component
Y = PCAfun(sorted_indices[:2])

plt.scatter(Y[0,:], Y[1,:],c=y)
plt.axis('equal')
plt.grid(True)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Scatter Plot')
plt.colorbar()
legend_handles = [
    plt.scatter([], [], label='THA076BEI009\nTHA076BEI036', alpha=0),
   
]

plt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(0.66, 0.999), 
           ncol=len(legend_handles),handlelength=0.3, borderpad=0.08)
plt.show()

#for 2 component
Y = PCAfun([sorted_indices[0],sorted_indices[2]])

plt.scatter(Y[0,:], Y[1,:],c=y)
plt.axis('equal')
plt.grid(True)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 3')
plt.title('PCA Scatter Plot')
plt.colorbar()
legend_handles = [
    plt.scatter([], [], label='THA076BEI009\nTHA076BEI036', alpha=0),
   
]

plt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(0.66, 0.999), 
           ncol=len(legend_handles),handlelength=0.3, borderpad=0.08)
plt.show()

#for 2 component
Y = PCAfun(sorted_indices[-2:])

plt.scatter(Y[0,:], Y[1,:],c=y)
plt.axis('equal')
plt.grid(True)
plt.xlabel('Principal Component 3')
plt.ylabel('Principal Component 4')
plt.title('PCA Scatter Plot')
plt.colorbar()
legend_handles = [
    plt.scatter([], [], label='THA076BEI009\nTHA076BEI036', alpha=0),
   
]

plt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(0.66, 0.999),
           ncol=len(legend_handles),handlelength=0.3, borderpad=0.08)
plt.show()

#for 3 component


Y = PCAfun(sorted_indices[:3])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(Y[0,:], Y[1,:],Y[2,:],c=y)
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')
ax.set_title('3D Scatter Plot')
ax.set_box_aspect([1, 1, 1])
ax.grid(True)
legend_handles = [
    ax.scatter([], [], label='THA076BEI009\nTHA076BEI036', alpha=0),
   
]

ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(0.56, 0.94),
          ncol=len(legend_handles),handlelength=0.3, borderpad=0.08)
plt.show()

#for 3 component


Y = PCAfun([sorted_indices[0],sorted_indices[1],sorted_indices[3]])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(Y[0,:], Y[1,:],Y[2,:],c=y)
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 4')
ax.set_title('3D Scatter Plot')
ax.set_box_aspect([1, 1, 1])
ax.grid(True)
legend_handles = [
    ax.scatter([], [], label='THA076BEI009\nTHA076BEI036', alpha=0),
   
]

ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(0.56, 0.94),
          ncol=len(legend_handles),handlelength=0.3, borderpad=0.08)
plt.show()

#for 3 component


Y = PCAfun([sorted_indices[0],sorted_indices[2],sorted_indices[3]])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(Y[0,:], Y[1,:],Y[2,:],c=y)
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 3')
ax.set_zlabel('PCA 4')
ax.set_title('3D Scatter Plot')
ax.set_box_aspect([1, 1, 1])
ax.grid(True)
legend_handles = [
    ax.scatter([], [], label='THA076BEI009\nTHA076BEI036', alpha=0),
   
]

ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(0.56, 0.94),
          ncol=len(legend_handles),handlelength=0.3, borderpad=0.08)
plt.show()

#for 3 component


Y = PCAfun(sorted_indices[-3:])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(Y[0,:], Y[1,:],Y[2,:],c=y)
ax.set_xlabel('PCA 2')
ax.set_ylabel('PCA 3')
ax.set_zlabel('PCA 4')
ax.set_title('3D Scatter Plot')
ax.set_box_aspect([1, 1, 1])
ax.grid(True)
legend_handles = [
    ax.scatter([], [], label='THA076BEI009\nTHA076BEI036', alpha=0),
   
]

ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(0.56, 0.94),
          ncol=len(legend_handles),handlelength=0.3, borderpad=0.08)
plt.show()
