#pragma once

///////////////////////////////////////////////////////////////////////////////

// Вычисление невязки rij во внутренних точках.
void CalcR( dim3 gridDim, cudaMatrix p, cudaUniformGrid grid, cudaMatrix r );

// Вычисление значений gij во внутренних точках.
void CalcG( dim3 gridDim, cudaMatrix r, const NumericType alpha, cudaMatrix g );

// Вычисление значений pij во внутренних точках, возвращается максимум норма.
NumericType CalcP( dim3 gridDim,
	cudaMatrix g, const NumericType tau, cudaMatrix p,
	cudaMatrix buffer1, cudaMatrix buffer2 );

// Вычисление alpha.
CFraction CalcAlpha( dim3 gridDim,
	cudaMatrix r, cudaMatrix g, cudaUniformGrid grid,
	cudaMatrix buffer1, cudaMatrix buffer2 );

// Вычисление tau.
CFraction CalcTau( dim3 gridDim,
	cudaMatrix r, cudaMatrix g, cudaUniformGrid grid,
	cudaMatrix buffer1, cudaMatrix buffer2 );

///////////////////////////////////////////////////////////////////////////////
