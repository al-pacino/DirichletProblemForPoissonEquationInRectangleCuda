#pragma once

///////////////////////////////////////////////////////////////////////////////

// ���������� ������� rij �� ���������� ������.
void CalcR( dim3 gridDim, cudaMatrix p, cudaUniformGrid grid, cudaMatrix r );

// ���������� �������� gij �� ���������� ������.
void CalcG( dim3 gridDim, cudaMatrix r, const NumericType alpha, cudaMatrix g );

// ���������� �������� pij �� ���������� ������, ������������ �������� �����.
NumericType CalcP( dim3 gridDim,
	cudaMatrix g, const NumericType tau, cudaMatrix p,
	cudaMatrix deviceBuffer );

// ���������� alpha.
CFraction CalcAlpha( dim3 gridDim,
	cudaMatrix r, cudaMatrix g, cudaUniformGrid grid,
	cudaMatrix deviceBuffer );

// ���������� tau.
CFraction CalcTau( dim3 gridDim,
	cudaMatrix r, cudaMatrix g, cudaUniformGrid grid,
	cudaMatrix deviceBuffer );

///////////////////////////////////////////////////////////////////////////////
