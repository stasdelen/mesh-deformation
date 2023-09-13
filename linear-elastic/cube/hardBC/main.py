import os
import numpy as np
import torch
from remin.func import grad
from remin.utils import mesh as rm
from remin.residual import Residual
from hard_model import trainHardModel
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
		mesh.find('1') + mesh.find('2') +
		mesh.find('3') + mesh.find('4') +
		mesh.find('5') + mesh.find('6')
		),
		'moving' : (
		mesh.find('2')
		),
		'inner' : mesh.find('pde', exclude='all')
	}
	return mesh_idx

def computeDistance(vertices):
	Vx = vertices[:,0:1]
	Vy = vertices[:,1:2]
	Vz = vertices[:,2:3]
	nNodes = len(Vx)
	distance = np.zeros((nNodes,1), dtype=np.float32)

	ub = [min(Vx), min(Vy), min(Vz)]
	lb = [max(Vx), max(Vy), max(Vz)]
	
	for i in range(nNodes):
		x_dist = min(abs((Vx[i, :] - lb[0])), abs((Vx[i, :] - ub[0])))
		y_dist = min(abs((Vy[i, :] - lb[1])), abs((Vy[i, :] - ub[1])))
		z_dist = min(abs((Vz[i, :] - lb[2])), abs((Vz[i, :] - ub[2])))
		distance[i, :] = min(x_dist, y_dist, z_dist)

	return distance

def sinosoidalMove(X_c, Y_c, Z_c, iter, sgn):
	return X_c, Y_c, Z_c + sgn * (
		0.05 * iter * np.sin(np.pi * X_c) * np.sin(np.pi * Y_c)
	)

def imposeHardBC(U_soft, mesh_idx, mesh, iter):
	U_p = U_soft.copy()
	
	stationary_wall_idx = mesh_idx['boundary'] ^ mesh_idx['moving']

	U_p[stationary_wall_idx] = mesh.vertices[stationary_wall_idx]
	V_c = mesh.vertices[mesh_idx['moving']]
	X_c, Y_c, Z_c = V_c[:,0:1], V_c[:,1:2], V_c[:,2:3]
	U_p[mesh_idx['moving']] = np.hstack(
		sinosoidalMove(X_c, Y_c, Z_c, iter, -1)
		)

	return U_p

def pde_residual(U, x, y, z):
	X, Y, Z = U
	X_x, X_y, X_z  = grad(X, [x, y, z])
	Y_x, Y_y, Y_z = grad(Y, [x, y, z])
	Z_x, Z_y, Z_z = grad(Z, [x, y, z])
	X_xx = grad(X_x, x)[0]
	X_yy = grad(X_y, y)[0]
	X_zz = grad(X_z, z)[0]
	Y_yy = grad(Y_y, y)[0]
	Y_xx = grad(Y_x, x)[0]
	Y_zz = grad(Y_z, z)[0]
	Z_zz = grad(Z_z, z)[0]
	Z_xx = grad(Z_x, x)[0]
	Z_yy = grad(Z_y, y)[0]
	div = X_x + Y_y + Z_z
	divx, divy, divz = grad(div, [x, y, z]) 

	fx = (mu_a + lambda_a)*divx + mu_a*(X_xx + X_yy + X_zz)
	fy = (mu_a + lambda_a)*divy + mu_a*(Y_xx + Y_yy + Y_zz)
	fz = (mu_a + lambda_a)*divz + mu_a*(Z_xx + Z_yy + Z_zz)
	return fx, fy, fz

def stationary_wall(U, x, y, z):
	X, Y, Z = U
	return X - x, Y - y,  Z - z

def moving_wall_iter(top):
	X_c, Y_c, Z_c = top[:,0:1], top[:,1:2], top[:,2:3]
	def func(iter):
		X0, Y0, Z0 = sinosoidalMove(X_c, Y_c, Z_c, iter, -1)
		X0 = torch.from_numpy(X0).to(device)
		Y0 = torch.from_numpy(Y0).to(device)
		Z0 = torch.from_numpy(Z0).to(device)
		def moving_wall(U, x, y, z):
			X, Y, Z = U
			return X - X0, Y - Y0, Z - Z0
		return moving_wall
	return func

def main():

	saveFile = 'results/'

	if not os.path.isdir(saveFile):
		os.makedirs(saveFile)

	mesh = rm.read('../cube.msh')
	mesh_idx = evaluate_mesh(mesh)

	moving_wall = moving_wall_iter(
		mesh.vertices[mesh_idx['moving']]
	)
	
	vertices = mesh.vertices.copy()

	# Save current hard model
	mesh.writeVTK(saveFile + 'hard_0', vertices)
	
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
		mesh.writeVTK(saveFile + softName, U_soft)

		# Impose hard BC, Train hard
		U_p = imposeHardBC(vertices, mesh_idx, mesh, i)
		
		# Save current particular model
		mesh.writeVTK(saveFile + partName, U_p)

		# Train hard model
		print(f'Training hard model {i}.')
		hard_residuals = [Residual(U_p, pde_residual)]
		vertices = trainHardModel(
			hard_residuals,
			distance,
			torch.from_numpy(U_p).to(device), 
			torch.from_numpy(U_p).to(device),
			hardName)
		
		# Save current hard model
		mesh.writeVTK(saveFile + hardName, vertices)
				
	return

if __name__ == '__main__':
	main()
