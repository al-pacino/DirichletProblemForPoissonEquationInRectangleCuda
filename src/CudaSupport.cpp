#include <Std.h>
#include <Errors.h>
#include <CudaSupport.h>
#include <Definitions.h>

///////////////////////////////////////////////////////////////////////////////

void SelectSuitableCudaDevice( ostream& out )
{
	int deviceCount = 0;
	int suitableDevice = -1;
	cudaCheck( cudaGetDeviceCount( &deviceCount ) );
	out << "Number of cuda devices: " << deviceCount << endl;
	out << endl;
	for( int i = 0; i < deviceCount; i++ ) {
		cudaDeviceProp deviceProp;
		cudaCheck( cudaGetDeviceProperties( &deviceProp, i ) );
		out << "Device#" << left << setw( 13 ) << i
			<< ": " << deviceProp.name << endl;
		out << "Compute capability  : " << deviceProp.major
			<< "." << deviceProp.minor << endl;
		out << "Total global memory : " << deviceProp.totalGlobalMem << endl; 
		out << endl;

		if( deviceProp.major == 2 /* && deviceProp.minor == 0 */ ) {
			suitableDevice = i;
		}
	}

	if( suitableDevice == -1 ) {
		throw CException( "Suitable cuda device was not found!" );
	}
	cudaCheck( cudaSetDevice( suitableDevice ) );

	out << "Device#" << suitableDevice << " has been selected." << endl;
	out << endl;
}

///////////////////////////////////////////////////////////////////////////////
