import os
import numpy as np
from remin.utils import mesh as rm


np.random.seed(0)

# Lame Parameters
mu_a = 0.35
lambda_a = 1.0

def evaluate_mesh(mesh):
	mesh_idx = {
		'boundary' : (
		mesh.find('blr') + mesh.find('top')
		),
		'moving' : (
		mesh.find('top')
		),
		'inner' : mesh.find('pde', exclude='all')
	}
	return mesh_idx

def sinosoidalMove(X_c, Y_c, iter, sgn):
	return X_c, Y_c + sgn * (
		0.05 * iter * np.sin(np.pi * X_c)
	)

def imposeHardBC(U_soft, mesh_idx, mesh, iter):
	U_p = U_soft.copy()
	
	stationary_wall_idx = mesh_idx['boundary'] ^ mesh_idx['moving']

	U_p[stationary_wall_idx] = mesh.vertices[stationary_wall_idx][:,0:2]
	V_c = mesh.vertices[mesh_idx['moving']][:,0:2]
	X_c, Y_c = V_c[:,0:1], V_c[:,1:2]
	U_p[mesh_idx['moving']] = np.hstack(
		sinosoidalMove(X_c, Y_c, iter, -1)
		)

	return U_p

def interpolate(U_p, U_soft, mesh_idx):
	dU = U_p - U_soft
	U_new = U_p.copy()
	U_bound = U_p[mesh_idx['boundary']]
	dU_bound = dU[mesh_idx['boundary']]
	for j in np.arange(U_p.shape[0])[mesh_idx['inner']]:
		dist = np.abs(U_bound - U_p[j])
		distM = (dist*dist).sum(axis=1, keepdims=True)
		# 1/r^2
		coef = np.linalg.norm(distM) / distM
		# [1/r^2 * exp(-x), 1/r^2 * exp(-y)]
		# coef = np.linalg.norm(distM) / distM * np.exp( - dist / np.linalg.norm(dist))
		shift = (
			np.multiply(dU_bound, coef).sum(axis=0)
			/ coef.sum(axis=0)
		)
		U_new[j] += shift
	return U_new

def toXYZ(U):
	return np.hstack((U, np.zeros((len(U), 1))))

def main():

	saveFile = 'results/'

	if not os.path.isdir(saveFile):
		os.makedirs(saveFile)

	mesh = rm.read('../square.msh')
	mesh_idx = evaluate_mesh(mesh)

	vertices = mesh.vertices[:,0:2].copy()

	# Save current hard model
	mesh.writeVTK(saveFile + 'hard_0', toXYZ(vertices))
	
	for i in range(1, 20):
		partName = f'part_{i}'
		hardName = f'hard_{i}'

		# Impose hard BC
		U_p = imposeHardBC(vertices, mesh_idx, mesh, i/5)
		
		# Save current particular model
		mesh.writeVTK(saveFile + partName, toXYZ(U_p))

		# Interpolate
		vertices = interpolate(U_p, vertices, mesh_idx)
		
		# Save current hard model
		mesh.writeVTK(saveFile + hardName, toXYZ(vertices))
				
	return

if __name__ == '__main__':
	main()
