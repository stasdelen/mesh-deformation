import os
import numpy as np
import torch
from remin.func import grad
from remin.utils import mesh as rm
from remin.residual import Residual
from soft_model import trainSoftModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
torch.set_float32_matmul_precision('high')
torch.manual_seed(0)
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

def computeDistance(vertices):
	Vx = vertices[:,0:1]
	Vy = vertices[:,1:2]
	nNodes = len(Vx)
	distance = np.zeros((nNodes,1), dtype=np.float32)

	ub = [min(Vx), min(Vy)]
	lb = [max(Vx), max(Vy)]
	
	for i in range(nNodes):
		x_dist = min(abs((Vx[i, :] - lb[0])), abs((Vx[i, :] - ub[0])))
		y_dist = min(abs((Vy[i, :] - lb[1])), abs((Vy[i, :] - ub[1])))
		distance[i, :] = min(x_dist, y_dist)

	return distance

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

def pde_residual(U, x, y):
	X, Y = U
	X_x, X_y  = grad(X, [x, y])
	Y_x, Y_y = grad(Y, [x, y])
	X_xx = grad(X_x, x)[0]
	X_yy = grad(X_y, y)[0]
	Y_yy = grad(Y_y, y)[0]
	Y_xx = grad(Y_x, x)[0]
	div = X_x + Y_y
	divx, divy = grad(div, [x, y]) 

	fx = (mu_a + lambda_a)*divx + mu_a*(X_xx + X_yy)
	fy = (mu_a + lambda_a)*divy + mu_a*(Y_xx + Y_yy)
	return fx, fy

def stationary_wall(U, x, y):
	X, Y = U
	return X - x, Y - y

def moving_wall_iter(top):
	X_c, Y_c = top[:,0:1], top[:,1:2]
	def func(iter):
		X0, Y0 = sinosoidalMove(X_c, Y_c, iter, -1)
		X0 = torch.from_numpy(X0).to(device)
		Y0 = torch.from_numpy(Y0).to(device)
		def moving_wall(U, x, y):
			X, Y = U
			return X - X0, Y - Y0
		return moving_wall
	return func

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

	moving_wall = moving_wall_iter(
		mesh.vertices[mesh_idx['moving']]
	)
	
	vertices = mesh.vertices[:,0:2].copy()

	# Save current hard model
	mesh.writeVTK(saveFile + 'hard_0', toXYZ(vertices))
	
	distance =  torch.from_numpy(computeDistance(vertices)).to(device)
	stationary_wall_idx = mesh_idx['boundary'] ^ mesh_idx['moving']


	for i in range(1, 4):
		softName = f'soft_{i}'
		partName = f'part_{i}'
		hardName = f'hard_{i}'
		
		# Train soft model
		print(f'Training soft model {i}.')
		soft_residuals = [
			Residual(vertices[mesh_idx['inner']], pde_residual),
			Residual(vertices[stationary_wall_idx], stationary_wall, weight=25),
			Residual(vertices[mesh_idx['moving']], moving_wall(i), weight=25),
		]
		U_soft = trainSoftModel(
			soft_residuals,
			torch.from_numpy(vertices).to(device), 
			softName)

		# Save current soft model
		mesh.writeVTK(saveFile + softName, toXYZ(U_soft))

		# Impose hard BC
		U_p = imposeHardBC(vertices, mesh_idx, mesh, i)
		
		# Save current particular model
		mesh.writeVTK(saveFile + partName, toXYZ(U_p))

		# Interpolate
		vertices = interpolate(U_p, vertices, mesh_idx)
		
		# Save current hard model
		mesh.writeVTK(saveFile + hardName, toXYZ(vertices))
				
	return

if __name__ == '__main__':
	main()
