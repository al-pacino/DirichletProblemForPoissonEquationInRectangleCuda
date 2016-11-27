#include <Std.h>
#include <Errors.h>
#include <CudaSupport.h>
#include <Definitions.h>
#include <CudaSupport.h>
#include <CudaObjects.h>

///////////////////////////////////////////////////////////////////////////////

cudaMatrix::cudaMatrix()
{
	Matrix.ptr = 0;
	Matrix.pitch = 0;
	Matrix.xsize = 0;
	Matrix.ysize = 0;
}

void cudaMatrix::Allocate( const CMatrix& matrix )
{
	cudaCheck( cudaMallocPitch( &Matrix.ptr, &Matrix.pitch,
		matrix.SizeX() * sizeof( NumericType ), matrix.SizeY() ) );
	Matrix.xsize = matrix.SizeX();
	Matrix.ysize = matrix.SizeY();
	cudaCheck( cudaMemcpy2D( Matrix.ptr, Matrix.pitch,
		matrix.values.data(), matrix.SizeX() * sizeof( NumericType ), 
			matrix.SizeX() * sizeof( NumericType ), matrix.SizeY(),
			cudaMemcpyHostToDevice ) );
}

void cudaMatrix::Dump( CMatrix& matrix ) const
{
	cudaCheck( cudaMemcpy2D( matrix.values.data(), matrix.SizeX() * sizeof( NumericType ),
		Matrix.ptr, Matrix.pitch,
		Matrix.xsize * sizeof( NumericType ), Matrix.ysize,
		cudaMemcpyDeviceToHost ) );
}

void cudaMatrix::SetPart( const CMatrixPart& part, const vector<NumericType>& values )
{
	cudaCheck( cudaMemcpy2D(
		(char*)Matrix.ptr + Matrix.pitch * part.BeginY + part.BeginX * sizeof( NumericType ),
		Matrix.pitch,
		values.data(), part.SizeX() * sizeof( NumericType ),
		part.SizeX() * sizeof( NumericType ), part.SizeY(),
		cudaMemcpyHostToDevice ) );
}

void cudaMatrix::GetPart( const CMatrixPart& part, vector<NumericType>& values ) const
{
	cudaCheck( cudaMemcpy2D(
		values.data(), part.SizeX() * sizeof( NumericType ),
		(char*)Matrix.ptr + Matrix.pitch * part.BeginY + part.BeginX * sizeof( NumericType ),
		Matrix.pitch,
		part.SizeX() * sizeof( NumericType ), part.SizeY(),
		cudaMemcpyDeviceToHost ) );
}

///////////////////////////////////////////////////////////////////////////////

void cudaUniformPartition::Allocate( const CUniformPartition& uniformPartition )
{
	cudaCheck( cudaMalloc( &Points, uniformPartition.Size() * sizeof( NumericType ) ) );
	cudaCheck( cudaMemcpy( Points, uniformPartition.ps.data(),
		uniformPartition.Size() * sizeof( NumericType ), cudaMemcpyHostToDevice ) );
}


///////////////////////////////////////////////////////////////////////////////
