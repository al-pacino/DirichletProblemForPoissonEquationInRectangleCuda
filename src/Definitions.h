#pragma once

// Файл содержит задание используемых типов и функций
// для решения задачи u''xx + u''yy + F = 0

#ifndef __host__
#define __host__
#define __device__
#endif

typedef double NumericType;
#ifdef MPI_VERSION
const MPI_Datatype MpiNumericType = MPI_DOUBLE;
#endif
const NumericType DefaultEps = static_cast<NumericType>( 0.0001 );

#include <MathObjects.h> // CArea

// Область решения задачи.
const CArea Area( -2, 2, -2, 2 );

const size_t BlockSizeX = 32;
const size_t BlockSizeY = 16;

// Правая часть.
__host__ __device__ inline NumericType F( NumericType x, NumericType y )
{
	const NumericType xy2 = ( x + y ) * ( x + y );
	const NumericType f = 4 * ( 1 - 2 * xy2 ) * exp( 1 - xy2 );
	return f;
}

// Граничная функция.
__host__ __device__ inline NumericType Phi( NumericType x, NumericType y )
{
	const NumericType xy2 = ( x + y ) * ( x + y );
	const NumericType phi = exp( 1 - xy2 );
	return phi;
}
