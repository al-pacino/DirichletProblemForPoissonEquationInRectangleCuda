#include <Std.h>
#include <Errors.h>
#include <CudaSupport.h>
#include <Definitions.h>

///////////////////////////////////////////////////////////////////////////////

void SelectSuitableCudaDevice()
{
	int deviceCount = 0;
	int suitableDevice = -1;
	cudaGetDeviceCount( &deviceCount );
	cerr << "Number of cuda devices: " << deviceCount << endl;
	cerr << endl;
	for( int i = 0; i < deviceCount; i++ ) {
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties( &deviceProp, i );
		cerr << "Device#" << left << setw( 13 ) << i
			<< ": " << deviceProp.name << endl;
		cerr << "Compute capability  : " << deviceProp.major
			<< "." << deviceProp.minor << endl;
		cerr << "Total global memory : " << deviceProp.totalGlobalMem << endl; 
		cerr << endl;

		if( deviceProp.major == 2 /* && deviceProp.minor == 0 */ ) {
			suitableDevice = i;
		}
	}

	if( suitableDevice == -1 ) {
		throw CException( "Suitable cuda device was not found!" );
	}

	cudaCheck( cudaSetDevice( suitableDevice ) );

	cerr << "Device#" << suitableDevice << " has been selected." << endl;
	cerr << endl;
	cerr << string( 64, '-' ) << endl << endl;
}

///////////////////////////////////////////////////////////////////////////////
