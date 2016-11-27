#pragma once

///////////////////////////////////////////////////////////////////////////////

struct cudaMatrix {
	cudaPitchedPtr Matrix;

	cudaMatrix();

	void Allocate( const CMatrix& matrix );
	void Dump( CMatrix& matrix ) const;

	void SetPart( const CMatrixPart& part, const vector<NumericType>& values );
	void GetPart( const CMatrixPart& part, vector<NumericType>& values ) const;

	__device__ NumericType& operator()( size_t x, size_t y )
	{
		return ((NumericType*)((char*)Matrix.ptr + y * Matrix.pitch))[x];
	}
	__device__ NumericType operator()( size_t x, size_t y ) const
	{
		return ((NumericType*)((char*)Matrix.ptr + y * Matrix.pitch))[x];
	}

	__device__ size_t SizeX() const { return Matrix.xsize; }
	__device__ size_t SizeY() const { return Matrix.ysize; }
};

///////////////////////////////////////////////////////////////////////////////

struct cudaUniformPartition {
	NumericType* Points;

	cudaUniformPartition() :
		Points( 0 )
	{
	}

	void Allocate( const CUniformPartition& uniformPartition );

	__device__ NumericType operator[]( size_t i ) const
	{
		return Point( i );
	}
	__device__ NumericType Point( size_t i ) const
	{
		return Points[i];
	}
	__device__ NumericType Step( size_t i ) const
	{
		return ( Point( i + 1 ) - Point( i ) );
	}
	__device__ NumericType AverageStep( size_t i ) const
	{
		//return ( Step( i ) + Step( i - 1 ) ) / static_cast<NumericType>( 2 );
		return ( Point( i + 1 ) - Point( i - 1 ) ) / static_cast<NumericType>( 2 );
	}
};

///////////////////////////////////////////////////////////////////////////////

struct cudaUniformGrid {
	cudaUniformPartition X;
	cudaUniformPartition Y;
};

///////////////////////////////////////////////////////////////////////////////

class cudaProgram {
public:
	const dim3 BlockDim;

	cudaProgram() :
		BlockDim( BlockSizeX, BlockSizeY )
	{
	}

	cudaMatrix P;
	cudaMatrix R;
	cudaMatrix G;
	CFraction Tau;
	CFraction Alpha;
	cudaUniformGrid Grid;

	void CalcP();
	void CalcR();
	void CalcG();
	void CalcTau();
	void CalcAlpha();

	dim3 GetGridDim() const { return gridDim; }
	void SetGridDim();

private:
	cudaMatrix blockValues;
	dim3 gridDim3;

	cudaProgram( const cudaProgram& );
	cudaProgram& operator=( const cudaProgram& );
};

///////////////////////////////////////////////////////////////////////////////
