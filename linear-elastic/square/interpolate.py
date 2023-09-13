from remin.utils import mesh as rm
from remin.solver import Solver, make_trainer
from remin.residual import Residual, make_loader
from remin.solver.residual_loss import EagerLoss
from remin import callbacks
from remin.func import grad
import torch
import numpy as np
from torch import nn
from soft_model import SoftMesh
from hard_model import HardMesh
import matplotlib.pyplot as plt
import matplotlib.tri as tri

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
torch.set_float32_matmul_precision('high')
torch.manual_seed(0)
np.random.seed(0)

# Lame Parameters
mu_a = 0.35
lambda_a = 1.0

def computeDistance(vertices):
	Vx = vertices[:,0:1]
	Vy = vertices[:,1:2]
	nNodes = len(Vx)
	distance = np.zeros((nNodes,1), dtype=np.float32)
	uniform = np.zeros((nNodes,1), dtype=np.float32)
	lb = [min(Vx), min(Vy)]
	ub = [max(Vx), max(Vy)]
	
	for i in range(nNodes):
		x_dist = min(abs((Vx[i, :] - lb[0])), abs((Vx[i, :] - ub[0])))
		y_dist = min(abs((Vy[i, :] - lb[1])), abs((Vy[i, :] - ub[1])))
		distance[i, :] = min(x_dist, y_dist) #* 1e-1
		if distance[i,:] != 0:
			uniform[i,:] = 1
		else:
			uniform[i,:] = 0
	# import matplotlib.pyplot as plt
	# plt.scatter(Vx, Vy, c = distance)
	# plt.colorbar()
	# plt.show()
	# exit()
	return distance, uniform

def imposeHardBC(U_soft, mesh, iter):
	U_p = np.zeros_like(U_soft)
	U_p[mesh.find('pde')] = U_soft[mesh.find('pde')]
	U_p[mesh.find('blr')] = mesh.get('blr')[:,0:2]
	
	V_c = mesh.get('top')[:,0:2]
	X_c, Y_c = V_c[:,0:1], V_c[:,1:2]
	U_p[mesh.find('top'),1:2] = Y_c - 0.05 * (iter + 1) * np.sin(np.pi * X_c)
	U_p[mesh.find('top'),0:1] = X_c

	dU = U_p - U_soft
	U_new = U_p.copy()
	U_bound = U_p[np.logical_or(mesh.find('blr'), mesh.find('top'))]
	dU_bound = dU[np.logical_or(mesh.find('blr'), mesh.find('top'))]
	for j in np.arange(U_p.shape[0])[mesh.find('pde', exclude='all')]:
		dist = np.abs(U_bound - U_p[j])
		distM = (dist*dist).sum(axis=1, keepdims=True)
		# 1/r^2
		coef = np.linalg.norm(distM) / distM
		# [1/r^2 * 1/x, 1/r^2 * 1/y]
		# coef = np.linalg.norm(distM) / distM  * np.linalg.norm(dist) / dist
		# [1/r^2 * exp(-x), 1/r^2 * exp(-y)]
		# coef = np.linalg.norm(distM) / distM * np.exp( - dist / np.linalg.norm(dist))
		shift = (
			np.multiply(dU_bound, coef).sum(axis=0)
			/ coef.sum(axis=0)
		)
		U_new[j] += shift

	return U_new

def pde_residual(U, x, y):
	X, Y = U
	X_x, X_y = grad(X, [x, y])
	Y_x, Y_y = grad(Y, [x, y])
	X_xx, X_xy = grad(X_x, [x, y])
	Y_yx, Y_yy = grad(Y_y, [x, y])
	X_yy = grad(X_y, y)[0]
	Y_xx = grad(Y_x, x)[0]
	
	fx = lambda_a*(X_xx + Y_yx) + mu_a*(2*X_xx + X_yy + Y_yx)
	fy = mu_a*(X_xy + Y_xx + 2*Y_yy) + lambda_a*(X_xy + Y_yy)
	return fx, fy

def stationary_wall(U, x, y):
	X, Y = U
	return X - x, Y - y

def moving_wall_iter(top):
	top = torch.from_numpy(top).to(device)
	X_c, Y_c = top[:,0:1], top[:,1:2]
	def func(iter):
		Y0 = Y_c - 0.05 * (iter + 1) * torch.sin(torch.pi * X_c)
		def moving_wall(U, x, y):
			X, Y = U
			return X - X_c, Y - Y0
		return moving_wall
	return func

def trainHardModel(residuals, distance, U_p, eval_vertices, iter):
	hardModel = HardMesh(distance, U_p)

	loader = make_loader(
		residuals,
		fully_loaded=True
	)

	epochs = 30000
	lr = 1e-2
	gamma = 0.9

	optimizer = torch.optim.Adam(hardModel.parameters(), lr=lr)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=gamma)
	resloss = EagerLoss(nn.HuberLoss())
	trainer = make_trainer(loader,
						   optimizer=optimizer,
						   scheduler=scheduler,
						   residual_loss=resloss)
	
	solver = Solver(hardModel,
					name=f'hardMesh_{iter}',
					save_folder=f'./outputs/hardMesh_{iter}',
					trainer=trainer)
	
	solver.reset_callbacks(
		callbacks.TotalTimeCallback(),
		callbacks.SaveCallback(),
		callbacks.LogCallback(log_epoch=1000, log_progress=100),
		callbacks.PlotCallback(state='residual', name='ressloss.png')
	)

	solver.fit(epochs)
	return hardModel.calc(eval_vertices).cpu().detach().numpy()

def trainSoftModel(residuals, eval_vertices, iter):
	softModel = SoftMesh()

	loader = make_loader(
		residuals,
		fully_loaded=True
	)

	epochs = 15000
	lr = 1e-5
	gamma = 0.99

	optimizer = torch.optim.Adam(softModel.parameters(), lr=lr)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=gamma)
	resloss = EagerLoss(nn.MSELoss())
	trainer = make_trainer(loader,
						   optimizer=optimizer,
						   scheduler=scheduler,
						   residual_loss=resloss)
	
	solver = Solver(softModel,
					name=f'sLMesh_{iter}',
					save_folder=f'./outputs/sLMesh_{iter}',
					trainer=trainer)
	
	solver.reset_callbacks(
		callbacks.TotalTimeCallback(),
		callbacks.SaveCallback(),
		callbacks.CSVCallback(),
		callbacks.LogCallback(log_epoch=1000, log_progress=100),
		callbacks.PlotCallback(state='residual', name='ressloss.png')
	)

	solver.fit(epochs)
	return softModel.calc(eval_vertices).cpu().detach().numpy()

def loadSoftModel(fileName):
	model = SoftMesh()
	mdata = torch.load(fileName)
	model.load_state_dict(mdata['model_state_dict'])
	model.eval()
	return model

def loadHardModel(distance, Up, fileName):
	model = HardMesh(distance, Up)
	mdata = torch.load(fileName)
	model.load_state_dict(mdata['model_state_dict'])
	model.eval()
	return model

def evaluate(iteration, vertices, mesh):
	for i in range(iteration+1):
		model = loadSoftModel(f'outputs/sLMesh_{i}/sLMesh_{i}_best.pt')
		U_soft = model.calc(torch.from_numpy(vertices).to(device)).cpu().detach().numpy()

		# Impose hard BC, save output
		vertices = imposeHardBC(U_soft, mesh, i)
	return vertices

def plot(vertices, mesh):
	triangulation = tri.Triangulation(
		vertices[:,0], 
		vertices[:,1],
		mesh.elements[
			mesh.offsets[mesh.elementTypes == 2][0]:
		].reshape(-1,3)
	)
	plt.cla()
	plt.triplot(triangulation, '-k')
	plt.show()

def toXYZ(U):
	return np.hstack((U, np.zeros((len(U), 1))))

def main():
	mesh = rm.read('square.msh')
	
	moving_wall = moving_wall_iter(mesh.get("top")[:,:2])
	vertices = mesh.vertices[:,0:2].copy()

	#vertices = evaluate(2, vertices, mesh)

	for i in [4]:
		# Train soft model,
		print(f'Training soft model {i}.')
		soft_residuals = [
			Residual(vertices[mesh.find('pde', exclude='all')], pde_residual),
			Residual(vertices[mesh.find('blr')], stationary_wall, weight=25),
			Residual(vertices[mesh.find('top')], moving_wall(i), weight=25)
		]
		U_soft = trainSoftModel(soft_residuals, torch.from_numpy(vertices).to(device), i)
				
		# Load ith Soft Model, compute particular solution
		# model = loadSoftModel(f'outputs/sLMesh_{i}/sLMesh_{i}_best.pt')
		# U_soft = model.calc(torch.from_numpy(vertices).to(device)).cpu().detach().numpy()

		# Save current soft model
		mesh.writeVTK(f'square_soft_{i}', toXYZ(U_soft))

		# Impose hard BC, save output
		vertices = imposeHardBC(U_soft, mesh, i)
		mesh.writeVTK(f'square_int_{i}', toXYZ(vertices))
				
	return

if __name__ == '__main__':
	main()
