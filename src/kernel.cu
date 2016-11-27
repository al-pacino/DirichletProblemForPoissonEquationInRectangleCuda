#include <Std.h>
#include <CudaSupport.h>
#include <Definitions.h>
#include <CudaObjects.h>
#include <kernel.h>

///////////////////////////////////////////////////////////////////////////////

__device__ void BlockReduceMax( volatile NumericType* shared,
	const size_t threadIndex, const NumericType value )
{
	shared[threadIndex] = value;
	__syncthreads();
	if( threadIndex < 256 ) {
		shared[threadIndex] = max( shared[threadIndex], shared[threadIndex + 256] );
	}
	__syncthreads();
	if( threadIndex < 128 ) {
		shared[threadIndex] = max( shared[threadIndex], shared[threadIndex + 128] );
	}
	__syncthreads();
	if( threadIndex < 64 ) {
		shared[threadIndex] = max( shared[threadIndex], shared[threadIndex + 128] );
	}
	__syncthreads();
	if( threadIndex < 32 ) {
		shared[threadIndex] = max( shared[threadIndex], shared[threadIndex + 32] );
	}
	__syncthreads();
	if( threadIndex < 16 ) {
		shared[threadIndex] = max( shared[threadIndex], shared[threadIndex + 16] );
	}
	__syncthreads();
	if( threadIndex < 8 ) {
		shared[threadIndex] = max( shared[threadIndex], shared[threadIndex + 8] );
	}
	__syncthreads();
	if( threadIndex < 4 ) {
		shared[threadIndex] = max( shared[threadIndex], shared[threadIndex + 4] );
	}
	__syncthreads();
	if( threadIndex < 2 ) {
		shared[threadIndex] = max( shared[threadIndex], shared[threadIndex + 2] );
	}
	__syncthreads();
	if( threadIndex < 1 ) {
		shared[threadIndex] = max( shared[threadIndex], shared[threadIndex + 1] );
	}
}

__device__ void BlockReduceSumTwo( volatile NumericType* shared,
	const size_t threadIndex, const NumericType value1, const NumericType value2 )
{
	shared[threadIndex] = value1;
	shared[threadIndex + 1] = value2;
	__syncthreads();
	if( threadIndex < 512 ) {
		shared[threadIndex] += shared[threadIndex + 512];
		shared[threadIndex + 1] += shared[threadIndex + 512 + 1];
	}
	__syncthreads();
	if( threadIndex < 256 ) {
		shared[threadIndex] += shared[threadIndex + 256];
		shared[threadIndex + 1] += shared[threadIndex + 256 + 1];
	}
	__syncthreads();
	if( threadIndex < 128 ) {
		shared[threadIndex] += shared[threadIndex + 128];
		shared[threadIndex + 1] += shared[threadIndex + 128 + 1];
	}
	__syncthreads();
	if( threadIndex < 64 ) {
		shared[threadIndex] += shared[threadIndex + 64];
		shared[threadIndex + 1] += shared[threadIndex + 64 + 1];
	}
	__syncthreads();
	if( threadIndex < 32 ) {
		shared[threadIndex] += shared[threadIndex + 32];
		shared[threadIndex + 1] += shared[threadIndex + 32 + 1];
	}
	__syncthreads();
	if( threadIndex < 16 ) {
		shared[threadIndex] += shared[threadIndex + 16];
		shared[threadIndex + 1] += shared[threadIndex + 16 + 1];
	}
	__syncthreads();
	if( threadIndex < 8 ) {
		shared[threadIndex] += shared[threadIndex + 8];
		shared[threadIndex + 1] += shared[threadIndex + 8 + 1];
	}
	__syncthreads();
	if( threadIndex < 4 ) {
		shared[threadIndex] += shared[threadIndex + 4];
		shared[threadIndex + 1] += shared[threadIndex + 4 + 1];
	}
	__syncthreads();
	if( threadIndex < 2 ) {
		shared[threadIndex] += shared[threadIndex + 2];
		shared[threadIndex + 1] += shared[threadIndex + 2 + 1];
	}
}

///////////////////////////////////////////////////////////////////////////////

__global__ void ReduceMaxKernel( cudaMatrix arr, cudaMatrix result )
{
	extern __shared__ NumericType shared[];
	const size_t index = blockDim.x * blockIdx.x + threadIdx.x;
	BlockReduceMax( shared, threadIdx.x, arr( index, 0 ) );
	if( threadIdx.x == 0 ) {
		result( blockIdx.x, 0 ) = shared[0];
	}
}

__global__ void ReduceSumTwoKernel( cudaMatrix arr2, cudaMatrix result )
{
	extern __shared__ NumericType shared[];
	const size_t index = blockDim.x * blockIdx.x + threadIdx.x;
	BlockReduceSumTwo( shared, threadIdx.x * 2, arr2( index, 0 ), arr2( index, 1 ) );
	if( threadIdx.x == 0 ) {
		result( blockIdx.x, 0 ) = shared[0];
		result( blockIdx.x, 1 ) = shared[1];
	}
}

///////////////////////////////////////////////////////////////////////////////

__device__ NumericType LaplasOperator( cudaMatrix matrix, cudaUniformGrid grid, size_t x, size_t y )
{
	const NumericType ldx = ( matrix( x, y ) - matrix( x - 1, y ) ) / grid.X.Step( x - 1 );
	const NumericType rdx = ( matrix( x + 1, y ) - matrix( x, y ) ) / grid.X.Step( x );
	const NumericType tdy = ( matrix( x, y ) - matrix( x, y - 1 ) ) / grid.Y.Step( y - 1 );
	const NumericType bdy = ( matrix( x, y + 1 ) - matrix( x, y ) ) / grid.Y.Step( y );
	const NumericType dx = ( ldx - rdx ) / grid.X.AverageStep( x );
	const NumericType dy = ( tdy - bdy ) / grid.Y.AverageStep( y );
	return ( dx + dy );
}

///////////////////////////////////////////////////////////////////////////////

// Вычисление невязки rij во внутренних точках.
__global__ void kernelCalcR( cudaMatrix p, cudaUniformGrid grid, cudaMatrix r )
{
	const size_t x = blockDim.x * blockIdx.x + threadIdx.x + 1;
	const size_t y = blockDim.y * blockIdx.y + threadIdx.y + 1;
	
	if( x < ( p.SizeX() - 1 ) && y < ( p.SizeY() - 1 ) ) {
		r( x, y ) = LaplasOperator( p, grid, x, y ) - F( grid.X[x], grid.Y[y] );
	}
}

// Вычисление значений gij во внутренних точках.
__global__ void kernelCalcG( cudaMatrix r, const NumericType alpha, cudaMatrix g )
{
	const size_t x = blockDim.x * blockIdx.x + threadIdx.x + 1;
	const size_t y = blockDim.y * blockIdx.y + threadIdx.y + 1;

	if( x < ( g.SizeX() - 1 ) && y < ( g.SizeY() - 1 ) ) {
		g( x, y ) = r( x, y ) - alpha * g( x, y );
	}
}

// Вычисление значений pij во внутренних точках, возвращается максимум норма.
__global__ void kernelCalcP( cudaMatrix g, const NumericType tau, cudaMatrix p,
	cudaMatrix differences )
{
	extern __shared__ NumericType shared[];

	const size_t x = blockDim.x * blockIdx.x + threadIdx.x + 1;
	const size_t y = blockDim.y * blockIdx.y + threadIdx.y + 1;
	const size_t threadIndex = threadIdx.y * blockDim.x + threadIdx.x;

	NumericType difference = 0;
	if( x < ( p.SizeX() - 1 ) && y < ( p.SizeY() - 1 ) ) {
		const NumericType newValue = p( x, y ) - tau * g( x, y );
		difference = abs( newValue - p( x, y ) );
		p( x, y ) = newValue;
	}

	BlockReduceMax( shared, threadIndex, difference );

	if( threadIndex == 0 ) {
		const size_t blockIndex = gridDim.x * blockIdx.y + blockIdx.x;
		differences( blockIndex, 0 ) = shared[0];
	}
}

// Вычисление alpha.
__global__ void kernelCalcAlpha( cudaMatrix r, cudaMatrix g, cudaUniformGrid grid,
	cudaMatrix alphas )
{
	extern __shared__ NumericType shared[];

	const size_t x = blockDim.x * blockIdx.x + threadIdx.x + 1;
	const size_t y = blockDim.y * blockIdx.y + threadIdx.y + 1;
	const size_t threadIndex = threadIdx.y * blockDim.x + threadIdx.x;

	NumericType numerator = 0;
	NumericType denominator = 0;
	if( x < ( r.SizeX() - 1 ) && y < ( r.SizeY() - 1 ) ) {
		const NumericType common = g( x, y ) * grid.X.AverageStep( x ) * grid.Y.AverageStep( y );
		numerator = LaplasOperator( r, grid, x, y ) * common;
		denominator = LaplasOperator( g, grid, x, y ) * common;
	}

	BlockReduceSumTwo( shared, threadIndex * 2, numerator, denominator );

	if( threadIndex == 0 ) {
		const size_t blockIndex = gridDim.x * blockIdx.y + blockIdx.x;
		alphas( blockIndex, 0 ) = shared[0];
		alphas( blockIndex, 1 ) = shared[1];
	}
}

// Вычисление tau.
__global__ void kernelCalcTau( cudaMatrix r, cudaMatrix g, cudaUniformGrid grid,
	cudaMatrix taus )
{
	extern __shared__ NumericType shared[];

	const size_t x = blockDim.x * blockIdx.x + threadIdx.x + 1;
	const size_t y = blockDim.y * blockIdx.y + threadIdx.y + 1;
	const size_t threadIndex = threadIdx.y * blockDim.x + threadIdx.x;

	NumericType numerator = 0;
	NumericType denominator = 0;
	if( x < ( r.SizeX() - 1 ) && y < ( r.SizeY() - 1 ) ) {
		const NumericType common = g( x, y ) * grid.X.AverageStep( x ) * grid.Y.AverageStep( y );
		numerator = r( x, y ) * common;
		denominator = LaplasOperator( g, grid, x, y ) * common;
	}

	BlockReduceSumTwo( shared, threadIndex * 2, numerator, denominator );

	if( threadIndex == 0 ) {
		const size_t blockIndex = gridDim.x * blockIdx.y + blockIdx.x;
		taus( blockIndex, 0 ) = shared[0];
		taus( blockIndex, 1 ) = shared[1];
	}
}

///////////////////////////////////////////////////////////////////////////////

const size_t SharedMemSize = BlockDim.x * BlockDim.y * sizeof( NumericType );
const size_t SharedMem2Size = SharedMemSize * 2;
const dim3 LinearBlockDim( 512 );
const size_t LinearSharedMemSize = LinearBlockDim.x * sizeof( NumericType );
const size_t LinearSharedMem2Size = LinearSharedMemSize * 2;

inline NumericType CalcMax( dim3 gridDim, cudaMatrix buffer1, cudaMatrix buffer2 )
{
	//const dim3 linearGridDim( buffer2.SizeX() );
	//ReduceMaxKernel<<<linearGridDim, LinearBlockDim, LinearSharedMemSize>>>( buffer1, buffer2 );
	buffer2 = buffer1;

	vector<NumericType> differences( buffer2.SizeX() );
	buffer2.GetPart( CMatrixPart( 0, buffer2.SizeX(), 0, 1 ), differences );

	NumericType difference = 0;
	for( size_t i = 0; i < buffer2.SizeX(); i++ ) {
		difference = max( difference, differences[i] );
	}

	return difference;
}

inline CFraction CalcFraction( dim3 gridDim, cudaMatrix buffer1, cudaMatrix buffer2 )
{
	//const dim3 linearGridDim( buffer2.SizeX() );
	//ReduceSumTwoKernel<<<linearGridDim, LinearBlockDim, LinearSharedMem2Size>>>( buffer1, buffer2 );
	buffer2 = buffer1;

	vector<NumericType> values( buffer2.SizeX() * 2 );
	buffer2.GetPart( CMatrixPart( 0, buffer2.SizeX(), 0, 2 ), values );

	NumericType numerator = 0;
	NumericType denominator = 0;
	for( size_t i = 0; i < buffer2.SizeX(); i++ ) {
		numerator += values[i];
		denominator += values[i + buffer2.SizeX()];
	}
	return CFraction( numerator, denominator );
}

// Вычисление невязки rij во внутренних точках.
void CalcR( dim3 gridDim, cudaMatrix p, cudaUniformGrid grid, cudaMatrix r )
{
	kernelCalcR<<<gridDim, BlockDim>>>( p, grid, r );
}

// Вычисление значений gij во внутренних точках.
void CalcG( dim3 gridDim, cudaMatrix r, const NumericType alpha, cudaMatrix g )
{
	kernelCalcG<<<gridDim, BlockDim>>>( r, alpha, g );
}

// Вычисление значений pij во внутренних точках, возвращается максимум норма.
NumericType CalcP( dim3 gridDim,
	cudaMatrix g, const NumericType tau, cudaMatrix p,
	cudaMatrix buffer1, cudaMatrix buffer2 )
{
	kernelCalcP<<<gridDim, BlockDim, SharedMemSize>>>( g, tau, p, buffer1 );
	return CalcMax( gridDim, buffer1, buffer2 );
}

// Вычисление alpha.
CFraction CalcAlpha( dim3 gridDim,
	cudaMatrix r, cudaMatrix g, cudaUniformGrid grid,
	cudaMatrix buffer1, cudaMatrix buffer2 )
{
	kernelCalcAlpha<<<gridDim, BlockDim, SharedMem2Size>>>( r, g, grid, buffer1 );
	return CalcFraction( gridDim, buffer1, buffer2 );
}

// Вычисление tau.
CFraction CalcTau( dim3 gridDim,
	cudaMatrix r, cudaMatrix g, cudaUniformGrid grid,
	cudaMatrix buffer1, cudaMatrix buffer2 )
{
	kernelCalcTau<<<gridDim, BlockDim, SharedMem2Size>>>( r, g, grid, buffer1 );
	return CalcFraction( gridDim, buffer1, buffer2 );
}

///////////////////////////////////////////////////////////////////////////////

#if 0

__global__ void CalcRuni( cudaMatrix p, cudaUniformGrid grid, cudaMatrix r )
{
	const size_t xPerThread = 0;
	const size_t yPerThread = 0;

	size_t x = ( BlockSizeX * blockIdx.x + threadIdx.x ) * xPerThread;
	const size_t xEnd = min( x + xPerThread, p.SizeX() );
	size_t y = ( BlockSizeY * blockIdx.y + threadIdx.y ) * yPerThread;
	const size_t yEnd = min( y + yPerThread, p.SizeY() );

	for( ; x < xEnd; x++ ) {
		for( ; y < yEnd; y++ ) {
			r( x, y ) = LaplasOperator( p, grid, x, y ) - F( grid.X[x], grid.Y[y] );
		}
	}
}

#endif
