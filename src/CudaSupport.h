#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

///////////////////////////////////////////////////////////////////////////////

#define cudaCheck( expr ) \
	{ \
		const cudaError_t errorCode = ( expr ); \
		if( errorCode != cudaSuccess ) { \
			ostringstream oss; \
			oss << __FILE__ << ":" \
				<< __LINE__ << ": " \
				<< cudaGetErrorString( errorCode ) << ": " \
				<< "`" << #expr << "`"; \
			throw CException( oss.str() ); \
		} \
	}

// selects a cuda device which compute capability is 2.*
void SelectSuitableCudaDevice();

///////////////////////////////////////////////////////////////////////////////
