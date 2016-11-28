#include <Std.h>
#include <CudaSupport.h>
#include <Definitions.h>
#include <CudaObjects.h>
#include <kernel.h>

///////////////////////////////////////////////////////////////////////////////

const dim3 BlockDim( BlockDimX, BlockDimY );
const size_t BlockSize = BlockDimX * BlockDimY;
const size_t SharedMemSize = BlockSize * sizeof( NumericType );
const size_t SharedMem2Size = SharedMemSize * 2;

///////////////////////////////////////////////////////////////////////////////

__device__ void blockReduceMax( volatile NumericType* shared,
	const size_t threadIndex, const NumericType value )
{
	shared[threadIndex] = value;
	__syncthreads();
	if( BlockSize >= 512 && threadIndex < 256 ) {
		shared[threadIndex] = max( shared[threadIndex], shared[threadIndex + 256] );
	}
	__syncthreads();
	if( BlockSize >= 256 && threadIndex < 128 ) {
		shared[threadIndex] = max( shared[threadIndex], shared[threadIndex + 128] );
	}
	__syncthreads();
	if( BlockSize >= 128 && threadIndex < 64 ) {
		shared[threadIndex] = max( shared[threadIndex], shared[threadIndex + 128] );
	}
	__syncthreads();
	if( BlockSize >= 64 && threadIndex < 32 ) {
		shared[threadIndex] = max( shared[threadIndex], shared[threadIndex + 32] );
	}
	__syncthreads();
	if( BlockSize >= 32 && threadIndex < 16 ) {
		shared[threadIndex] = max( shared[threadIndex], shared[threadIndex + 16] );
	}
	__syncthreads();
	if( BlockSize >= 16 && threadIndex < 8 ) {
		shared[threadIndex] = max( shared[threadIndex], shared[threadIndex + 8] );
	}
	__syncthreads();
	if( BlockSize >= 8 && threadIndex < 4 ) {
		shared[threadIndex] = max( shared[threadIndex], shared[threadIndex + 4] );
	}
	__syncthreads();
	if( BlockSize >= 4 && threadIndex < 2 ) {
		shared[threadIndex] = max( shared[threadIndex], shared[threadIndex + 2] );
	}
	__syncthreads();
	if( BlockSize >= 2 && threadIndex < 1 ) {
		shared[threadIndex] = max( shared[threadIndex], shared[threadIndex + 1] );
	}
}

__device__ void blockReduceSumTwo( volatile NumericType* shared,
	const size_t threadIndex, const NumericType value1, const NumericType value2 )
{
	shared[threadIndex] = value1;
	shared[threadIndex + 1] = value2;
	__syncthreads();
	if( BlockSize >= 512 && threadIndex < 512 ) {
		shared[threadIndex] += shared[threadIndex + 512];
		shared[threadIndex + 1] += shared[threadIndex + 512 + 1];
	}
	__syncthreads();
	if( BlockSize >= 256 && threadIndex < 256 ) {
		shared[threadIndex] += shared[threadIndex + 256];
		shared[threadIndex + 1] += shared[threadIndex + 256 + 1];
	}
	__syncthreads();
	if( BlockSize >= 128 && threadIndex < 128 ) {
		shared[threadIndex] += shared[threadIndex + 128];
		shared[threadIndex + 1] += shared[threadIndex + 128 + 1];
	}
	__syncthreads();
	if( BlockSize >= 64 && threadIndex < 64 ) {
		shared[threadIndex] += shared[threadIndex + 64];
		shared[threadIndex + 1] += shared[threadIndex + 64 + 1];
	}
	__syncthreads();
	if( BlockSize >= 32 && threadIndex < 32 ) {
		shared[threadIndex] += shared[threadIndex + 32];
		shared[threadIndex + 1] += shared[threadIndex + 32 + 1];
	}
	__syncthreads();
	if( BlockSize >= 16 && threadIndex < 16 ) {
		shared[threadIndex] += shared[threadIndex + 16];
		shared[threadIndex + 1] += shared[threadIndex + 16 + 1];
	}
	__syncthreads();
	if( BlockSize >= 8 && threadIndex < 8 ) {
		shared[threadIndex] += shared[threadIndex + 8];
		shared[threadIndex + 1] += shared[threadIndex + 8 + 1];
	}
	__syncthreads();
	if( BlockSize >= 4 && threadIndex < 4 ) {
		shared[threadIndex] += shared[threadIndex + 4];
		shared[threadIndex + 1] += shared[threadIndex + 4 + 1];
	}
	__syncthreads();
	if( BlockSize >= 2 && threadIndex < 2 ) {
		shared[threadIndex] += shared[threadIndex + 2];
		shared[threadIndex + 1] += shared[threadIndex + 2 + 1];
	}
}

///////////////////////////////////////////////////////////////////////////////

__device__ NumericType laplasOperator( cudaMatrix matrix, cudaUniformGrid grid, size_t x, size_t y )
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
	const size_t x = BlockDimX * blockIdx.x + threadIdx.x + 1;
	const size_t y = BlockDimY * blockIdx.y + threadIdx.y + 1;

	if( x < ( p.SizeX() - 1 ) && y < ( p.SizeY() - 1 ) ) {
		r( x, y ) = laplasOperator( p, grid, x, y ) - F( grid.X[x], grid.Y[y] );
	}
}

// Вычисление значений gij во внутренних точках.
__global__ void kernelCalcG( cudaMatrix r, const NumericType alpha, cudaMatrix g )
{
	const size_t x = BlockDimX * blockIdx.x + threadIdx.x + 1;
	const size_t y = BlockDimY * blockIdx.y + threadIdx.y + 1;

	if( x < ( g.SizeX() - 1 ) && y < ( g.SizeY() - 1 ) ) {
		g( x, y ) = r( x, y ) - alpha * g( x, y );
	}
}

// Вычисление значений pij во внутренних точках, возвращается максимум норма.
__global__ void kernelCalcP( cudaMatrix g, const NumericType tau, cudaMatrix p,
	cudaMatrix differences )
{
	extern __shared__ NumericType shared[];

	const size_t x = BlockDimX * blockIdx.x + threadIdx.x + 1;
	const size_t y = BlockDimY * blockIdx.y + threadIdx.y + 1;
	const size_t threadIndex = threadIdx.y * BlockDimX + threadIdx.x;

	NumericType difference = 0;
	if( x < ( p.SizeX() - 1 ) && y < ( p.SizeY() - 1 ) ) {
		const NumericType newValue = p( x, y ) - tau * g( x, y );
		difference = abs( newValue - p( x, y ) );
		p( x, y ) = newValue;
	}

	blockReduceMax( shared, threadIndex, difference );

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

	const size_t x = BlockDimX * blockIdx.x + threadIdx.x + 1;
	const size_t y = BlockDimY * blockIdx.y + threadIdx.y + 1;
	const size_t threadIndex = threadIdx.y * BlockDimX + threadIdx.x;

	NumericType numerator = 0;
	NumericType denominator = 0;
	if( x < ( r.SizeX() - 1 ) && y < ( r.SizeY() - 1 ) ) {
		const NumericType common = g( x, y ) * grid.X.AverageStep( x ) * grid.Y.AverageStep( y );
		numerator = laplasOperator( r, grid, x, y ) * common;
		denominator = laplasOperator( g, grid, x, y ) * common;
	}

	blockReduceSumTwo( shared, threadIndex * 2, numerator, denominator );

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

	const size_t x = BlockDimX * blockIdx.x + threadIdx.x + 1;
	const size_t y = BlockDimY * blockIdx.y + threadIdx.y + 1;
	const size_t threadIndex = threadIdx.y * BlockDimX + threadIdx.x;

	NumericType numerator = 0;
	NumericType denominator = 0;
	if( x < ( r.SizeX() - 1 ) && y < ( r.SizeY() - 1 ) ) {
		const NumericType common = g( x, y ) * grid.X.AverageStep( x ) * grid.Y.AverageStep( y );
		numerator = r( x, y ) * common;
		denominator = laplasOperator( g, grid, x, y ) * common;
	}

	blockReduceSumTwo( shared, threadIndex * 2, numerator, denominator );

	if( threadIndex == 0 ) {
		const size_t blockIndex = gridDim.x * blockIdx.y + blockIdx.x;
		taus( blockIndex, 0 ) = shared[0];
		taus( blockIndex, 1 ) = shared[1];
	}
}

///////////////////////////////////////////////////////////////////////////////

namespace { // anonymous namespace

inline NumericType CalcMax( cudaMatrix buffer )
{
	vector<NumericType> differences( buffer.SizeX() );
	buffer.GetPart( CMatrixPart( 0, buffer.SizeX(), 0, 1 ), differences );
	return *max_element( differences.begin(), differences.end() );
}

inline CFraction CalcFraction( cudaMatrix buffer )
{
	vector<NumericType> values( buffer.SizeX() * 2 );
	buffer.GetPart( CMatrixPart( 0, buffer.SizeX(), 0, 2 ), values );
	typedef vector<NumericType>::const_iterator CIterator;
	const CIterator middle = values.begin() + buffer.SizeX();
	const NumericType numerator = accumulate<CIterator, NumericType>( values.begin(), middle, 0 );
	const NumericType denominator = accumulate<CIterator, NumericType>( middle, values.end(), 0 );
	return CFraction( numerator, denominator );
}

} // end of anonymous namespace

///////////////////////////////////////////////////////////////////////////////

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
	cudaMatrix buffer )
{
	kernelCalcP<<<gridDim, BlockDim, SharedMemSize>>>( g, tau, p, buffer );
	return CalcMax( buffer );
}

// Вычисление alpha.
CFraction CalcAlpha( dim3 gridDim,
	cudaMatrix r, cudaMatrix g, cudaUniformGrid grid,
	cudaMatrix buffer )
{
	kernelCalcAlpha<<<gridDim, BlockDim, SharedMem2Size>>>( r, g, grid, buffer );
	return CalcFraction( buffer );
}

// Вычисление tau.
CFraction CalcTau( dim3 gridDim,
	cudaMatrix r, cudaMatrix g, cudaUniformGrid grid,
	cudaMatrix buffer )
{
	kernelCalcTau<<<gridDim, BlockDim, SharedMem2Size>>>( r, g, grid, buffer );
	return CalcFraction( buffer );
}

///////////////////////////////////////////////////////////////////////////////
